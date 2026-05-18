"""
rl_training.py
==============
D3PO reinforcement learning pipeline:

  ZXEnv       — ZX-calculus simplification environment
  Buffer      — rollout buffer with GAE computation
  rollout     — collect trajectories
  Training steps:
    train_critic_system   (alternating hypernet+critic_params / critic_net)
    train_actor_system    (alternating hypernet+actor_params / actor_net)

Key fixes vs. original
----------------------
1. log(policy) always clipped to [1e-6, 1.0] before log — prevents -inf/NaN in PPO.
2. kl_discrete / entropy_discrete mask BEFORE normalising — avoids polluting
   the distribution with dummy-entry probability mass.
3. match_matrix indices are 0-based; CCAT shifts internally by +1 when
   looking up into h (which has the virtual node at index 0).
"""

from __future__ import annotations

import pickle
from fractions import Fraction

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
from flax.training import train_state

import pyzx as zx

from graph_utils import (
    pad_graph_like,
    fair_spider_unfusion,
    get_match_data,
)
from data_pipeline import add_virtual_node


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def _sample_circuit_jax(key, n_qubits, depth, p_X, p_Z, p_H, p_CNOT, p_I, p_is_T):
    """JAX-key driven circuit sampler (used inside ZXEnv.reset)."""
    c = zx.Circuit(n_qubits)
    gate_types = jnp.array([0, 1, 2, 3, 4])
    probs = jnp.array([p_X, p_Z, p_H, p_CNOT, p_I])
    probs = probs / probs.sum()

    for _ in range(depth):
        key, layer_key = jax.random.split(key)
        q_key, choice_key, phase_key = jax.random.split(layer_key, 3)
        shuffled = jax.random.permutation(q_key, jnp.arange(n_qubits)).tolist()

        while shuffled:
            choice_key, sub_key = jax.random.split(choice_key)
            gate_idx = int(jax.random.choice(sub_key, gate_types, p=probs))

            if gate_idx == 4 or not shuffled:
                shuffled.pop()
            elif gate_idx == 3:
                if len(shuffled) >= 2:
                    i, j = int(shuffled.pop()), int(shuffled.pop())
                    c.add_gate("CNOT", i, j)
                else:
                    shuffled.pop()
            elif gate_idx == 2:
                c.add_gate("HAD", int(shuffled.pop()))
            else:  # 0 or 1
                q = int(shuffled.pop())
                phase_key, p_sub, t_sub = jax.random.split(phase_key, 3)
                is_t = bool(jax.random.uniform(p_sub) < p_is_T)
                if not is_t:
                    val = int(jax.random.randint(p_sub, (), 1, 4))
                    phase = Fraction(val, 2)
                else:
                    val = int(jax.random.choice(t_sub, jnp.array([1, 3, 5, 7])))
                    phase = Fraction(val, 4)
                name = "ZPhase" if gate_idx == 1 else "XPhase"
                c.add_gate(name, q, phase=phase)
    return c


def _circuit_to_jax_graph(g: zx.Graph) -> dict:
    """Convert a PyZX graph to a dict of JAX arrays (no virtual node yet)."""
    inputs  = set(g.inputs())
    outputs = set(g.outputs())
    boundary = inputs | outputs
    spiders  = sorted(v for v in g.vertices() if v not in boundary)
    node_map = {v: i for i, v in enumerate(spiders)}

    feats = []
    for v in spiders:
        phase = float(g.phase(v)) * np.pi
        is_in  = any(n in inputs  for n in g.neighbors(v))
        is_out = any(n in outputs for n in g.neighbors(v))
        feats.append([
            jnp.cos(phase), jnp.sin(phase),
            float(is_in), float(not is_in and not is_out), float(is_out),
        ])

    senders, receivers = [], []
    for u, v in g.edge_set():
        if u in node_map and v in node_map:
            ui, vi = node_map[u], node_map[v]
            senders.extend([ui, vi]); receivers.extend([vi, ui])

    return {
        "nodes":     jnp.array(feats,      dtype=jnp.float32),
        "senders":   jnp.array(senders,    dtype=jnp.int32),
        "receivers": jnp.array(receivers,  dtype=jnp.int32),
        "n_node":    jnp.array([len(spiders)], dtype=jnp.int32),
        "n_edge":    jnp.array([len(senders)], dtype=jnp.int32),
    }


class ZXEnv:
    """
    ZX-calculus simplification environment.

    Observations contain:
      graph        : dict ready for jraph (with virtual node)
      actor_nodes  : match_matrix (0-based spider indices)
      mask         : mask_matrix
      action_select: segment_ids
      raw_matches  : raw PyZX match objects
    """

    def __init__(
        self,
        key,
        axiom_names: list[str],
        preference_dim: int,
        n_qubits: int   = 5,
        depth: int      = 20,
        fixed_qubit: bool = True,
        fixed_depth: bool = True,
        num_step: int   = 16,
    ):
        self.axiom_names    = axiom_names
        self.preference_dim = preference_dim
        self.num_step       = num_step
        self.setup          = dict(
            max_qubits=n_qubits, max_depth=depth,
            fixed_qubit=fixed_qubit, fixed_depth=fixed_depth,
        )
        self.rule_map = {
            "gadget_fusion":  zx.gadget_simp.apply,
            "id_removal":     zx.id_simp.apply,
            "pivot":          zx.pivot_simp.apply,
            "lcomp":          zx.lcomp_simp.apply,
            "pivot_boundary": zx.pivot_boundary_simp.apply,
            "pivot_gadget":   zx.pivot_gadget_simp.apply,
            "unfusion":       fair_spider_unfusion,
        }

    def reset(self, key, initial_circuit=None, pen: float = 10.0) -> dict:
        if initial_circuit is None:
            k, k1, k2, k3 = jax.random.split(key, 4)
            n_qubits = self.setup["max_qubits"]
            if not self.setup["fixed_qubit"]:
                k, qk = jax.random.split(k)
                n_qubits = int(jax.random.randint(qk, (), 2, n_qubits))
            depth = self.setup["max_depth"]
            if not self.setup["fixed_depth"]:
                k, dk = jax.random.split(k)
                depth = int(jax.random.randint(dk, (), 5, depth))
            probs = jax.random.dirichlet(k1, jnp.ones(5))
            p_X, p_Z, p_H, p_CNOT, p_I = probs
            p_is_T = float(jax.random.uniform(k2))
            initial_circuit = _sample_circuit_jax(
                k3, n_qubits, depth, p_X, p_Z, p_H, p_CNOT, p_I, p_is_T)

        self.initial_circuit = initial_circuit
        self.pen = pen
        self.g   = initial_circuit.copy().to_graph()
        zx.to_graph_like(self.g)
        self.g = pad_graph_like(self.g)

        tc    = initial_circuit.tcount()
        twoq  = initial_circuit.twoqubitcount()
        depth = initial_circuit.depth()
        self.current_metric = np.array([tc, twoq, depth], dtype=np.float32)
        self.init_metric    = self.current_metric.copy()
        self.steps = 0

        return self._make_obs()

    def _make_obs(self) -> dict:
        match_matrix, mask_matrix, segment_ids, raw_matches = \
            get_match_data(self.g.copy(), self.axiom_names)
        # Shift match_matrix +1 so that index 0 = virtual node in h
        # (h has shape [N_spiders+1, d_model] with virtual node at 0)
        shifted_matrix = np.where(mask_matrix, match_matrix + 1, 0)
        graph = _circuit_to_jax_graph(self.g)
        return {
            "graph":        add_virtual_node(graph),
            "actor_nodes":  shifted_matrix,
            "mask":         mask_matrix,
            "action_select": segment_ids,
            "raw_matches":  raw_matches,
        }

    def _reward(self, new_t, new_twoq, new_depth):
        metric = np.array([new_t, new_twoq, new_depth], dtype=np.float32)
        delta  = (self.current_metric - metric) / (self.init_metric + 1.0)
        sign   = np.sign(delta)
        return sign * np.log1p(np.abs(delta))

    def _end_reward(self):
        delta = (self.init_metric - self.current_metric) / (self.init_metric + 1.0)
        return np.sign(delta) * np.log1p(np.abs(delta))

    def _dynamic_penalty(self):
        f_t = (self.steps / self.num_step) * self.pen
        p   = -np.log1p(f_t)
        return np.array([p, p, p], dtype=np.float32)

    def step(self, action_idx: int, raw_matches: list, segment_ids):
        done       = False
        graph_data = None

        # STOP action
        if segment_ids[action_idx] == len(self.axiom_names):
            return graph_data, self._end_reward(), True, {"msg": "stop"}

        match_data = raw_matches[action_idx]
        rule_name  = self.axiom_names[segment_ids[action_idx]]

        g_next = self.g.copy()
        g_next = self._apply_rule(g_next, rule_name, match_data)
        if not zx.is_graph_like(g_next):
            zx.to_graph_like(g_next)

        try:
            c_next     = zx.extract_circuit(g_next.copy()).to_basic_gates()
            c_next     = zx.basic_optimization(c_next)
            new_t      = c_next.tcount()
            new_twoq   = c_next.twoqubitcount()
            new_depth  = c_next.depth()
            new_metric = np.array([new_t, new_twoq, new_depth], dtype=np.float32)

            reward = (self._reward(new_t, new_twoq, new_depth)
                      if self.steps < self.num_step - 1
                      else self._end_reward())

            self.g = pad_graph_like(g_next)
            self.current_metric[:] = new_metric
            info = {"valid": True}

        except Exception as e:
            reward = self._dynamic_penalty()
            info   = {"valid": False, "error": str(e)}

        self.steps += 1
        done = self.steps >= self.num_step

        if not done:
            graph_data = self._make_obs()

        return graph_data, reward, done, info

    def _apply_rule(self, g, name, m):
        if name == "gadget_fusion":
            self.rule_map[name](g, [m])
        elif name == "id_removal":
            self.rule_map[name](g, m)
        elif name == "pivot":
            self.rule_map[name](g, *m)
        elif name in ("lcomp", "pivot_boundary"):
            self.rule_map[name](g, m)
        elif name == "pivot_gadget":
            self.rule_map[name](g, m[0])
        elif name == "unfusion":
            self.rule_map[name](g, m)
        return g


# ---------------------------------------------------------------------------
# Preference sampling
# ---------------------------------------------------------------------------

def sample_preference(key) -> jax.Array:
    return jax.random.dirichlet(key, jnp.ones(3), dtype=jnp.float32)


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """
    Multi-objective GAE.
    rewards, values, next_values: (T, reward_dim)
    dones: (T,)
    """
    dones_exp = dones.astype(jnp.float32)[:, None]
    deltas    = rewards + gamma * next_values * (1.0 - dones_exp) - values

    def scan_fn(carry, xs):
        delta, done = xs
        adv = delta + gamma * lam * (1.0 - done) * carry
        return adv, adv

    _, advantages = jax.lax.scan(
        scan_fn, jnp.zeros_like(values[0]), (deltas, dones.astype(jnp.float32)),
        reverse=True)
    return advantages, advantages + values


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------

class Buffer:
    def __init__(self):
        self.states       = []
        self.actors       = {"actor_nodes": [], "mask": [], "action_select": []}
        self.mask_dummy   = []
        self.raw_matches  = []
        self.preferences  = []
        self.log_policies = []
        self.actions      = []
        self.edges_count  = []
        self.unfusion_id  = []
        self.rewards      = []
        self.values       = []
        self.dones        = []
        self.advantages   = None
        self.targets      = None

    def clear(self):
        self.__init__()

    def compute_gae_targets(self):
        vals   = jnp.array(self.values,  dtype=jnp.float32)
        rews   = jnp.array(self.rewards, dtype=jnp.float32)
        dones  = jnp.array(self.dones,   dtype=jnp.float32)
        dummy  = jnp.zeros((1, vals.shape[1]), dtype=vals.dtype)
        next_v = jnp.concatenate([vals[1:], dummy], axis=0)
        self.advantages, self.targets = compute_gae(rews, vals, next_v, dones)
        self.rewards = []; self.values = []; self.dones = []

    def normalize(self, state_cap, row_cap, col_cap, n_axioms):
        """Pad all variable-length arrays to static shapes for JIT."""
        dummy_axiom_id = n_axioms + 1

        padded_states, padded_nodes, padded_masks, padded_selects, dummy_masks = \
            [], [], [], [], []

        for i, state in enumerate(self.states):
            row_pad = state_cap - state.shape[0]
            padded_states.append(jnp.pad(state, ((0, row_pad), (0, 0))))

        for i in range(len(self.actors["action_select"])):
            nodes  = self.actors["actor_nodes"][i]
            mask   = self.actors["mask"][i]
            select = self.actors["action_select"][i]
            real   = select.shape[0]
            rp     = row_cap - real
            cp_n   = col_cap - nodes.shape[1]
            cp_m   = col_cap - mask.shape[1]

            # Real-vs-dummy indicator
            dm = jnp.concatenate([jnp.ones(real), jnp.zeros(rp)])
            dummy_masks.append(dm)

            padded_nodes.append(jnp.pad(nodes,  ((0, rp), (0, cp_n))))
            padded_masks.append(jnp.pad(mask,   ((0, rp), (0, cp_m))))

            pad_select = jnp.full((rp,), dummy_axiom_id, dtype=select.dtype)
            padded_selects.append(jnp.concatenate([select, pad_select]))

        self.states                    = jnp.stack(padded_states)
        self.mask_dummy                = jnp.stack(dummy_masks)
        self.actors["actor_nodes"]     = jnp.stack(padded_nodes)
        self.actors["mask"]            = jnp.stack(padded_masks)
        self.actors["action_select"]   = jnp.stack(padded_selects)
        self.preferences  = jnp.array(self.preferences,  dtype=jnp.float32)
        self.log_policies = jnp.array(self.log_policies, dtype=jnp.float32)
        self.actions      = jnp.array(self.actions)
        self.edges_count  = jnp.array(self.edges_count)
        self.unfusion_id  = jnp.array(self.unfusion_id)
        self.rewards      = jnp.array(self.rewards,      dtype=jnp.float32)
        self.values       = jnp.array(self.values,       dtype=jnp.float32)
        self.dones        = jnp.array(self.dones,        dtype=jnp.float32)


def _get_buffer_caps(buffer: Buffer):
    state_cap = max(s.shape[0] for s in buffer.states)
    row_cap   = max(m.shape[0] for m in buffer.actors["mask"])
    col_cap   = max(m.shape[1] for m in buffer.actors["mask"])
    def ceil2(n): return 1 << ((n - 1).bit_length()) if n > 1 else 1
    return ceil2(state_cap), ceil2(row_cap), ceil2(col_cap)


# ---------------------------------------------------------------------------
# Policy utilities
# ---------------------------------------------------------------------------

def edit_unfusion_probs(rewrite_probs, group_ids, mask_matrix, edges_count,
                        unfusion_id, alpha=0.9):
    degrees     = jnp.sum(mask_matrix, axis=1) - 1
    densities   = degrees / jnp.maximum(edges_count, 1)
    is_unfusion = (group_ids == unfusion_id)
    unf_dens    = jnp.where(is_unfusion, densities, 0.0)
    norm_dens   = unf_dens / (jnp.sum(unf_dens) + 1e-15)
    return jnp.where(is_unfusion,
                     (1.0 - alpha) * rewrite_probs + alpha * norm_dens,
                     rewrite_probs)


def compute_policy(axiom_probs, rewrite_probs, segment_ids, eps=1e-6):
    """
    Joint probability: P(match, axiom) = P(match|axiom) * P(axiom).
    Result is clipped to [eps, 1] so log(policy) is always finite.
    """
    p_axiom = axiom_probs[segment_ids]
    unorm   = rewrite_probs * p_axiom + 1e-8
    policy  = unorm / jnp.sum(unorm)
    return jnp.clip(policy, eps, 1.0)


def compute_masked_probs(axiom_logits, logits, mask, inf_val=1e12):
    """Mask dummy rewrites and dummy axiom before softmax."""
    masked_logits       = jnp.where(mask == 0.0, -inf_val, logits)
    masked_axiom_logits = axiom_logits.at[-1].set(-inf_val)
    return jax.nn.softmax(masked_axiom_logits), jax.nn.softmax(masked_logits)


# FIX: mask BEFORE normalising to avoid polluting distributions
def kl_discrete(p, q, mask, epsilon=1e-15):
    p_m    = p * mask
    q_m    = q * mask
    p_safe = (p_m + epsilon * mask) / (jnp.sum(p_m) + epsilon)
    q_safe = (q_m + epsilon * mask) / (jnp.sum(q_m) + epsilon)
    return jnp.sum(p_m * (jnp.log(p_safe + epsilon) - jnp.log(q_safe + epsilon)))


def entropy_discrete(p, mask, epsilon=1e-15):
    p_m    = p * mask
    p_safe = (p_m + epsilon * mask) / (jnp.sum(p_m) + epsilon)
    return -jnp.sum(p_m * jnp.log(p_safe + epsilon))


def per_obj_ppo_loss(new_log_probs, old_log_probs, advantages, clip_eps=0.2):
    ratio  = jnp.exp(new_log_probs - old_log_probs)[:, None]
    surr1  = ratio * advantages
    surr2  = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return jnp.minimum(surr1, surr2)


def critic_loss(values, targets):
    return jnp.mean((values - targets) ** 2)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout(
    key, env: ZXEnv, buffer: Buffer,
    encoder, actor_net, critic_net,
    actor_params_net, critic_params_net, hypernet,
    params: dict,
    initial_circuit=None,
    max_iter: int = 256,
):
    key, reset_key = jax.random.split(key)
    obs        = env.reset(reset_key, initial_circuit=initial_circuit)
    graph      = jraph.GraphsTuple(**obs["graph"])
    actor      = {k: obs[k] for k in ("actor_nodes", "mask", "action_select")}
    raw_matches = obs["raw_matches"]

    key, pref_key = jax.random.split(key)
    omega = sample_preference(pref_key)
    done  = False
    vars_ = {"params": None}

    for _ in range(max_iter):
        if done:
            key, pref_key, reset_key = jax.random.split(key, 3)
            omega = sample_preference(pref_key)
            obs   = env.reset(reset_key)
            graph = jraph.GraphsTuple(**obs["graph"])
            actor = {k: obs[k] for k in ("actor_nodes", "mask", "action_select")}
            raw_matches = obs["raw_matches"]

        buffer.preferences.append(omega)
        for k in ("actor_nodes", "mask", "action_select"):
            buffer.actors[k].append(actor[k])

        # Encode
        vars_["params"] = params["encoder"]
        h, _ = encoder.apply(vars_, graph)
        buffer.states.append(h)

        # Shared embedding
        vars_["params"] = params["hypernet"]
        shared = hypernet.apply(vars_, omega)

        # Modulations
        vars_["params"] = params["actor_params"]
        a_mods = actor_params_net.apply(vars_, shared)

        vars_["params"] = params["critic_params"]
        c_mods = critic_params_net.apply(vars_, shared)

        # Critic value
        vars_["params"] = params["critic_net"]
        values = critic_net.apply(
            vars_, h,
            c_mods["q"], c_mods["sk"], c_mods["sv"], c_mods["so"],
            c_mods["s_gate"], c_mods["s_up"], c_mods["s_down"], c_mods["s_out"])
        buffer.values.append(values)

        # Actor logits
        vars_["params"] = params["actor_net"]
        rw = a_mods["rewrites"]; ax = a_mods["axioms"]
        axiom_logits, logits = actor_net.apply(
            vars_, h, actor,
            rw["sk"], rw["so"], rw["sq"], rw["sv"],
            rw["s_gate"], rw["s_up"], rw["s_down"], rw["a"],
            ax["sk"], ax["so"], ax["sv"], ax["q"], ax["a"])

        axiom_logits  = axiom_logits.at[-1].set(-1e12)
        axiom_probs   = jax.nn.softmax(axiom_logits)
        unfusion_id   = 0
        edges_count   = env.g.num_edges()
        rw_probs      = edit_unfusion_probs(
            jax.nn.softmax(logits), actor["action_select"],
            actor["mask"], edges_count, unfusion_id)
        policy = compute_policy(axiom_probs, rw_probs, actor["action_select"])
        action = int(jnp.argmax(policy))

        # FIXED: clip before log to avoid -inf
        log_policy = jnp.log(jnp.clip(policy, 1e-6, 1.0))
        buffer.log_policies.append(log_policy[action])
        buffer.actions.append(action)
        buffer.edges_count.append(edges_count)
        buffer.unfusion_id.append(unfusion_id)

        next_obs, reward, done, info = env.step(action, raw_matches, actor["action_select"])
        if next_obs is not None:
            graph = jraph.GraphsTuple(**next_obs["graph"])
            actor = {k: next_obs[k] for k in ("actor_nodes", "mask", "action_select")}
            raw_matches = next_obs["raw_matches"]
        buffer.rewards.append(reward)
        buffer.dones.append(float(done))

    sc, rc, cc = _get_buffer_caps(buffer)
    buffer.normalize(sc, rc, cc, len(env.axiom_names))
    buffer.compute_gae_targets()


# ---------------------------------------------------------------------------
# Training state creation
# ---------------------------------------------------------------------------

def create_rl_states(
    rng, encoder, hypernet, actor_params_net, critic_params_net,
    actor_net, critic_net,
    dummy_pref, obs, pretrained_encoder_params, lrs,
):
    keys = jax.random.split(rng, 6)
    rng_enc, rng_hn, rng_ap, rng_cp, rng_an, rng_cn = keys

    dummy_graph = jraph.GraphsTuple(**obs["graph"])

    # Encoder (pretrained, low LR to fine-tune)
    _ = encoder.init(rng_enc, dummy_graph)
    enc_state = train_state.TrainState.create(
        apply_fn=encoder.apply, params=pretrained_encoder_params,
        tx=optax.adam(lrs["encoder"]))

    # Hypernet
    hyper_vars  = hypernet.init(rng_hn, dummy_pref)
    hyper_state = train_state.TrainState.create(
        apply_fn=hypernet.apply, params=hyper_vars["params"],
        tx=optax.adam(lrs["hypernet"]))

    shared = hypernet.apply({"params": hyper_state.params}, dummy_pref)

    # Actor params
    ap_vars  = actor_params_net.init(rng_ap, shared)
    ap_state = train_state.TrainState.create(
        apply_fn=actor_params_net.apply, params=ap_vars["params"],
        tx=optax.adam(lrs["actor_params"]))

    # Critic params
    cp_vars  = critic_params_net.init(rng_cp, shared)
    cp_state = train_state.TrainState.create(
        apply_fn=critic_params_net.apply, params=cp_vars["params"],
        tx=optax.adam(lrs["critic_params"]))

    a_mods = actor_params_net.apply({"params": ap_state.params}, shared)
    c_mods = critic_params_net.apply({"params": cp_state.params}, shared)
    h, _   = encoder.apply({"params": enc_state.params}, dummy_graph)

    dummy_actor = {k: obs[k] for k in ("actor_nodes", "mask", "action_select")}
    rw = a_mods["rewrites"]; ax = a_mods["axioms"]

    # Actor net
    an_vars  = actor_net.init(
        rng_an, h, dummy_actor,
        rw["sk"], rw["so"], rw["sq"], rw["sv"],
        rw["s_gate"], rw["s_up"], rw["s_down"], rw["a"],
        ax["sk"], ax["so"], ax["sv"], ax["q"], ax["a"])
    an_state = train_state.TrainState.create(
        apply_fn=actor_net.apply, params=an_vars["params"],
        tx=optax.adam(lrs["actor_net"]))

    # Critic net
    cn_vars  = critic_net.init(
        rng_cn, h,
        c_mods["q"], c_mods["sk"], c_mods["sv"], c_mods["so"],
        c_mods["s_gate"], c_mods["s_up"], c_mods["s_down"], c_mods["s_out"])
    cn_state = train_state.TrainState.create(
        apply_fn=critic_net.apply, params=cn_vars["params"],
        tx=optax.adam(lrs["critic_net"]))

    return {
        "encoder":       enc_state,
        "hypernet":      hyper_state,
        "actor_params":  ap_state,
        "critic_params": cp_state,
        "actor_net":     an_state,
        "critic_net":    cn_state,
    }


def sample_batch(key, buffer: Buffer, batch_size: int = 32) -> dict:
    n   = buffer.states.shape[0]
    idx = jax.random.permutation(key, n)[:batch_size]
    return {
        "states":      buffer.states[idx],
        "advantages":  buffer.advantages[idx],
        "targets":     buffer.targets[idx],
        "preferences": buffer.preferences[idx],
        "log_policies": buffer.log_policies[idx],
        "actions":     buffer.actions[idx],
        "edges_count": buffer.edges_count[idx],
        "unfusion_id": buffer.unfusion_id[idx],
        "actors": {k: buffer.actors[k][idx] for k in buffer.actors},
        "mask_dummy":  buffer.mask_dummy[idx],
    }


# ---------------------------------------------------------------------------
# Batched forward passes
# ---------------------------------------------------------------------------

def _critic_forward_batch(states, preferences, hyper_st, cp_st, cn_st):
    shared     = jax.vmap(lambda o: hyper_st.apply_fn({"params": hyper_st.params}, o))(preferences)
    c_mods_all = jax.vmap(lambda s: cp_st.apply_fn({"params": cp_st.params}, s))(shared)

    return jax.vmap(lambda h, q, sk, sv, so, sg, su, sd, sout:
        cn_st.apply_fn({"params": cn_st.params}, h, q, sk, sv, so, sg, su, sd, sout)
    )(
        states,
        c_mods_all["q"], c_mods_all["sk"], c_mods_all["sv"], c_mods_all["so"],
        c_mods_all["s_gate"], c_mods_all["s_up"], c_mods_all["s_down"], c_mods_all["s_out"],
    )


def _actor_forward_batch(states, actors, mask_dummy, preferences,
                         hyper_st, ap_st, an_st, edges_count, unfusion_id):
    shared     = jax.vmap(lambda o: hyper_st.apply_fn({"params": hyper_st.params}, o))(preferences)
    a_mods_all = jax.vmap(lambda s: ap_st.apply_fn({"params": ap_st.params}, s))(shared)

    def single(h, actor, rw, ax, md, ec, uid):
        al, logits = an_st.apply_fn(
            {"params": an_st.params}, h, actor,
            rw["sk"], rw["so"], rw["sq"], rw["sv"],
            rw["s_gate"], rw["s_up"], rw["s_down"], rw["a"],
            ax["sk"], ax["so"], ax["sv"], ax["q"], ax["a"])
        a_probs, rw_probs = compute_masked_probs(al, logits, md)
        rw_probs = edit_unfusion_probs(rw_probs, actor["action_select"], actor["mask"], ec, uid)
        a_probs_per_match = a_probs[actor["action_select"]]
        policy = (rw_probs * a_probs_per_match) + 1e-8
        policy = policy / jnp.sum(policy)
        return jnp.clip(policy, 1e-6, 1.0)

    return jax.vmap(single)(
        states, actors,
        a_mods_all["rewrites"], a_mods_all["axioms"],
        mask_dummy, edges_count, unfusion_id,
    )


# ---------------------------------------------------------------------------
# Critic training
# ---------------------------------------------------------------------------

@jax.jit
def train_critic_modulators_step(batch, hyper_st, cp_st, cn_st):
    def loss_fn(hp, cpp):
        h_tmp  = hyper_st.replace(params=hp)
        cp_tmp = cp_st.replace(params=cpp)
        vals   = _critic_forward_batch(batch["states"], batch["preferences"], h_tmp, cp_tmp, cn_st)
        return critic_loss(vals, batch["targets"])

    (gh, gcp) = jax.grad(loss_fn, argnums=(0, 1))(hyper_st.params, cp_st.params)
    return hyper_st.apply_gradients(grads=gh), cp_st.apply_gradients(grads=gcp)


@jax.jit
def train_critic_net_step(batch, hyper_st, cp_st, cn_st):
    def loss_fn(cnp):
        cn_tmp = cn_st.replace(params=cnp)
        vals   = _critic_forward_batch(batch["states"], batch["preferences"], hyper_st, cp_st, cn_tmp)
        return critic_loss(vals, batch["targets"])

    grads = jax.grad(loss_fn)(cn_st.params)
    return cn_st.apply_gradients(grads=grads)


def train_critic_system(rng, batch, hyper_st, cp_st, cn_st,
                        num_steps=20, percentage=0.8):
    hyper_steps  = int(num_steps * percentage) + 1
    net_steps    = max(1, num_steps - hyper_steps)
    for _ in range(net_steps):
        cn_st = train_critic_net_step(batch, hyper_st, cp_st, cn_st)
    for _ in range(hyper_steps):
        hyper_st, cp_st = train_critic_modulators_step(batch, hyper_st, cp_st, cn_st)
    return hyper_st, cp_st, cn_st


# ---------------------------------------------------------------------------
# Actor training
# ---------------------------------------------------------------------------

@jax.jit
def train_actor_modulators_step(batch, hyper_st, ap_st, an_st, new_omega, hyperparams, clip_eps=0.2):
    def loss_fn(hp, app):
        h_tmp  = hyper_st.replace(params=hp)
        ap_tmp = ap_st.replace(params=app)

        policies = _actor_forward_batch(
            batch["states"], batch["actors"], batch["mask_dummy"],
            batch["preferences"], h_tmp, ap_tmp, an_st,
            batch["edges_count"], batch["unfusion_id"])

        # FIXED: clip before log
        new_lp  = jnp.take_along_axis(
            jnp.log(jnp.clip(policies, 1e-6, 1.0)),
            batch["actions"][:, None], axis=1).squeeze(-1)
        ppo_vec = per_obj_ppo_loss(new_lp, batch["log_policies"], batch["advantages"], clip_eps)
        scalar  = -jnp.mean(jnp.sum(batch["preferences"] * ppo_vec, axis=-1))

        # Diversity + entropy regularisation
        new_pols = _actor_forward_batch(
            batch["states"], batch["actors"], batch["mask_dummy"],
            new_omega, h_tmp, ap_tmp, an_st,
            batch["edges_count"], batch["unfusion_id"])
        kl_vals  = jax.vmap(kl_discrete)(policies, new_pols, batch["mask_dummy"])
        diff     = jnp.sum(jnp.abs(batch["preferences"] - new_omega), axis=-1)
        div_loss = jnp.mean((kl_vals - hyperparams["alpha"] * diff) ** 2)
        ent      = jax.vmap(entropy_discrete)(policies, batch["mask_dummy"])
        ent_loss = jnp.mean(ent)

        return scalar + hyperparams["diversity"] * div_loss - hyperparams["entropy"] * ent_loss

    gh, gap = jax.grad(loss_fn, argnums=(0, 1))(hyper_st.params, ap_st.params)
    return hyper_st.apply_gradients(grads=gh), ap_st.apply_gradients(grads=gap)


@jax.jit
def train_actor_net_step(batch, hyper_st, ap_st, an_st, new_omega, hyperparams, clip_eps=0.2):
    def loss_fn(anp):
        an_tmp   = an_st.replace(params=anp)
        policies = _actor_forward_batch(
            batch["states"], batch["actors"], batch["mask_dummy"],
            batch["preferences"], hyper_st, ap_st, an_tmp,
            batch["edges_count"], batch["unfusion_id"])
        # FIXED: clip before log
        new_lp  = jnp.take_along_axis(
            jnp.log(jnp.clip(policies, 1e-6, 1.0)),
            batch["actions"][:, None], axis=1).squeeze(-1)
        ppo_vec = per_obj_ppo_loss(new_lp, batch["log_policies"], batch["advantages"], clip_eps)
        return -jnp.mean(jnp.sum(batch["preferences"] * ppo_vec, axis=-1))

    grads = jax.grad(loss_fn)(an_st.params)
    return an_st.apply_gradients(grads=grads)


def train_actor_system(key, batch, hyper_st, ap_st, an_st, hyperparams,
                       num_steps=20, percentage=0.8, clip_eps=0.2, noise=0.05):
    hyper_steps = int(num_steps * percentage) + 1
    net_steps   = max(1, num_steps - hyper_steps)

    for _ in range(net_steps):
        key, sub = jax.random.split(key)
        pert     = noise * jax.random.uniform(sub, batch["preferences"].shape)
        new_omega = batch["preferences"] + pert
        new_omega = new_omega / jnp.sum(new_omega, axis=-1, keepdims=True)
        an_st     = train_actor_net_step(batch, hyper_st, ap_st, an_st, new_omega, hyperparams, clip_eps)

    for _ in range(hyper_steps):
        key, sub = jax.random.split(key)
        pert     = noise * jax.random.uniform(sub, batch["preferences"].shape)
        new_omega = batch["preferences"] + pert
        new_omega = new_omega / jnp.sum(new_omega, axis=-1, keepdims=True)
        hyper_st, ap_st = train_actor_modulators_step(
            batch, hyper_st, ap_st, an_st, new_omega, hyperparams, clip_eps)

    return hyper_st, ap_st, an_st
