"""
rl_model.py
===========
Reinforcement-learning model components:

  Hypernetwork (SiluResMLPNet)
    ↓ shared embedding
  ActorParams   → dynamic weights for ActorNetwork
  CriticParams  → dynamic weights for CriticNetwork

  ActorNetwork:
    CCAT_block (action-centric attention)
    SegmentedPooling_block (pool per axiom)
    → (axiom_logits, rewrite_logits)

  CriticNetwork:
    CPooling_block (global graph pooling)
    → value vector (reward_dim,)

All Cayley / SoftSgTh / Silu FFN layers are defined here alongside
the network modules that use them.

Key fix vs. original:
  CCAT_layer.action_attention uses safe_softmax instead of
  jax.nn.softmax — prevents NaN when all attention logits are -inf
  (i.e. a dummy/padded match with all-zero node mask).
"""

from __future__ import annotations

from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import jraph
import flax.linen as nn

from vae_model import CayleyDense, RowCayleyDense, ColCayleyDense


# ---------------------------------------------------------------------------
# Numerical utilities
# ---------------------------------------------------------------------------

def safe_softmax(logits: jax.Array, axis: int = -1) -> jax.Array:
    """
    Softmax that returns a zero vector (not NaN) when all logits are -inf.
    Standard softmax: softmax([-inf, …]) = nan.
    This version: returns [0, 0, …, 0] for fully-masked rows.
    """
    shifted  = logits - jnp.max(logits, axis=axis, keepdims=True)
    exp_x    = jnp.exp(shifted)
    sum_exp  = jnp.sum(exp_x, axis=axis, keepdims=True)
    safe_sum = jnp.where(sum_exp == 0, 1.0, sum_exp)
    return exp_x / safe_sum


def softSign_tanh(x: jax.Array, slope: float = 1.0, max_val: float = 1.0) -> jax.Array:
    return 0.5 * max_val * (jax.nn.tanh(slope * x) + jax.nn.soft_sign(x))


# ---------------------------------------------------------------------------
# Residual FFN layers (with Cayley skip connections)
# ---------------------------------------------------------------------------

def _make_skip(d_in: int, d_out: int):
    """Return the appropriate Cayley skip layer for the (d_in, d_out) pair."""
    if d_in < d_out:
        return RowCayleyDense(d_in=d_in, d_out=d_out)
    elif d_in == d_out:
        return lambda x: x
    else:
        return ColCayleyDense(d_in=d_in, d_out=d_out)


class SiluFFNLayer(nn.Module):
    d_in:     int
    d_out:    int
    use_bias: bool  = True
    scale:    float = 1.0

    def setup(self):
        self.W    = nn.Dense(self.d_out, use_bias=self.use_bias,
                             kernel_init=nn.initializers.orthogonal(self.scale))
        self.W_up = _make_skip(self.d_in, self.d_out)

    def __call__(self, x):
        return self.W_up(x) + nn.silu(self.W(x))


class GeluFFNLayer(nn.Module):
    d_in:     int
    d_out:    int
    use_bias: bool  = True
    scale:    float = 1.0

    def setup(self):
        self.W    = nn.Dense(self.d_out, use_bias=self.use_bias,
                             kernel_init=nn.initializers.orthogonal(self.scale))
        self.W_up = _make_skip(self.d_in, self.d_out)

    def __call__(self, x):
        return self.W_up(x) + nn.gelu(self.W(x))


class SoftSgThFFNLayer(nn.Module):
    d_in:     int
    d_out:    int
    use_bias: bool  = True
    scale:    float = 1.0

    def setup(self):
        self.W    = nn.Dense(self.d_out, use_bias=self.use_bias,
                             kernel_init=nn.initializers.orthogonal(self.scale))
        self.W_up = _make_skip(self.d_in, self.d_out)

    def __call__(self, x):
        return self.W_up(x) + softSign_tanh(self.W(x))


# ---------------------------------------------------------------------------
# Residual MLP networks
# ---------------------------------------------------------------------------

class SiluResMLPNet(nn.Module):
    """Stack of SiluFFNLayer with arbitrary widths."""
    layers_dim:  Sequence[int]
    use_biases:  Optional[Sequence[bool]]  = None
    scales:      Optional[Sequence[float]] = None

    def setup(self):
        self.ffn_layers = [
            SiluFFNLayer(
                d_in  = self.layers_dim[i],
                d_out = self.layers_dim[i + 1],
                use_bias = self.use_biases[i] if self.use_biases else True,
                scale    = self.scales[i]     if self.scales     else 1.0,
            )
            for i in range(len(self.layers_dim) - 1)
        ]

    def __call__(self, x):
        for layer in self.ffn_layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Gated FFN (Cayley parametrised)
# ---------------------------------------------------------------------------

class CFFNHead(nn.Module):
    """
    SwiGLU-style gated FFN with Cayley weight matrices.
    Handles both expansion (d_in ≤ d_mid) and contraction (d_in > d_mid).
    """
    d_in:  int
    d_mid: int

    def setup(self):
        if self.d_in <= self.d_mid:
            self.U_gate = CayleyDense(self.d_in)
            self.U_up   = CayleyDense(self.d_in)
            self.U_down = ColCayleyDense(self.d_mid, self.d_in)
            self.V_gate = RowCayleyDense(self.d_in, self.d_mid)
            self.V_up   = RowCayleyDense(self.d_in, self.d_mid)
            self.V_down = CayleyDense(self.d_in)
        else:
            self.U_gate = ColCayleyDense(self.d_in, self.d_mid)
            self.U_up   = ColCayleyDense(self.d_in, self.d_mid)
            self.U_down = CayleyDense(self.d_mid)
            self.V_gate = CayleyDense(self.d_mid)
            self.V_up   = CayleyDense(self.d_mid)
            self.V_down = RowCayleyDense(self.d_mid, self.d_in)

    def __call__(self, x, s_gate, s_up, s_down):
        gate = nn.silu(self.V_gate(self.U_gate(x) * s_gate))
        up   = self.V_up(self.U_up(x) * s_up)
        return self.V_down(self.U_down(gate * up) * s_down)


# ---------------------------------------------------------------------------
# Hypernetwork parameter generators
# ---------------------------------------------------------------------------

class ActorParams(nn.Module):
    """Maps a shared preference embedding to all actor modulation weights."""
    d_in:    int
    d_satt:  int
    d_shead: int
    d_a:     int

    def setup(self):
        # Rewrite attention modulators
        self.sk_rw   = SoftSgThFFNLayer(self.d_in, self.d_satt)
        self.so_rw   = SoftSgThFFNLayer(self.d_in, self.d_satt)
        self.sq_rw   = SoftSgThFFNLayer(self.d_in, self.d_satt)
        self.sv_rw   = SoftSgThFFNLayer(self.d_in, self.d_satt)
        self.s_gate  = SoftSgThFFNLayer(self.d_in, self.d_shead)
        self.s_up    = SoftSgThFFNLayer(self.d_in, self.d_shead)
        self.s_down  = SoftSgThFFNLayer(self.d_in, self.d_shead)
        self.a_rw    = SiluFFNLayer(self.d_in, self.d_a)
        # Axiom pooling modulators
        self.q_ax    = SiluFFNLayer(self.d_in, self.d_satt)
        self.sk_ax   = SoftSgThFFNLayer(self.d_in, self.d_satt)
        self.so_ax   = SoftSgThFFNLayer(self.d_in, self.d_satt)
        self.sv_ax   = SoftSgThFFNLayer(self.d_in, self.d_satt)
        self.a_ax    = SiluFFNLayer(self.d_in, self.d_a)

    def __call__(self, shared):
        return {
            "rewrites": {
                "sk": self.sk_rw(shared), "so": self.so_rw(shared),
                "sq": self.sq_rw(shared), "sv": self.sv_rw(shared),
                "s_gate": self.s_gate(shared),
                "s_up":   self.s_up(shared),
                "s_down": self.s_down(shared),
                "a":      self.a_rw(shared),
            },
            "axioms": {
                "q":  self.q_ax(shared),  "sk": self.sk_ax(shared),
                "so": self.so_ax(shared), "sv": self.sv_ax(shared),
                "a":  self.a_ax(shared),
            },
        }


class CriticParams(nn.Module):
    """Maps a shared preference embedding to critic modulation weights."""
    d_in:    int
    d_satt:  int
    d_shead: int
    d_sout:  int

    def setup(self):
        self.q      = SiluFFNLayer(self.d_in, self.d_satt)
        self.sk     = SoftSgThFFNLayer(self.d_in, self.d_satt)
        self.so     = SoftSgThFFNLayer(self.d_in, self.d_satt)
        self.sv     = SoftSgThFFNLayer(self.d_in, self.d_satt)
        self.s_gate = SoftSgThFFNLayer(self.d_in, self.d_shead)
        self.s_up   = SoftSgThFFNLayer(self.d_in, self.d_shead)
        self.s_down = SoftSgThFFNLayer(self.d_in, self.d_shead)
        self.s_out  = SoftSgThFFNLayer(self.d_in, self.d_sout)

    def __call__(self, shared):
        return {
            "q": self.q(shared), "sk": self.sk(shared),
            "so": self.so(shared), "sv": self.sv(shared),
            "s_gate": self.s_gate(shared), "s_up": self.s_up(shared),
            "s_down": self.s_down(shared), "s_out": self.s_out(shared),
        }


# ---------------------------------------------------------------------------
# Actor: CCAT (action-centric cross-attention)
# ---------------------------------------------------------------------------

class CCATLayer(nn.Module):
    """
    Cross-attention from action prototypes into graph node features.

    For each match (a set of nodes), compute a soft attention summary
    weighted by the match-specific action query.

    BUG FIX: uses safe_softmax instead of jax.nn.softmax so that
    fully-masked (dummy) match rows return zeros rather than NaN.
    """
    d_in:       int
    num_actions: int
    num_heads:  int   = 4
    mask_inf:   float = 1e9

    def setup(self):
        assert self.d_in % self.num_heads == 0
        self.d_head = self.d_in // self.num_heads
        oi = nn.initializers.orthogonal()
        self.A  = self.param("action_kernel", oi, (self.num_actions, self.d_in))
        self.Uk = CayleyDense(self.d_in); self.Vk = CayleyDense(self.d_in)
        self.Uq = CayleyDense(self.d_in); self.Vq = CayleyDense(self.d_in)
        self.Uv = CayleyDense(self.d_in); self.Vv = CayleyDense(self.d_in)
        self.Uo = CayleyDense(self.d_in); self.Vo = CayleyDense(self.d_in)

    def __call__(self, nodes, actors, sk, so, sq, sv):
        # Pad A with a zero dummy axiom row
        dummy = jnp.zeros((1, self.d_in), dtype=self.A.dtype)
        padded_A = jnp.concatenate([self.A, dummy], axis=0)

        k = self.Vk(self.Uk(nodes)    * sk)
        q = self.Vq(self.Uq(padded_A) * sq)
        v = self.Vv(self.Uv(nodes)    * sv)

        def action_attention(actor_nodes, nodes_mask, action_idx):
            # actor_nodes: (max_nodes_per_match,) — 0-based spider indices
            # The virtual-node shift (+1) is embedded in actor_nodes already
            # when they come from add_virtual_node-aligned encoder output.
            k_idx = k[actor_nodes].reshape(-1, self.num_heads, self.d_head)
            v_idx = v[actor_nodes].reshape(-1, self.num_heads, self.d_head)
            q_idx = q[action_idx ].reshape(   self.num_heads,  self.d_head)
            scale = 1.0 / jnp.sqrt(self.d_head)
            raw   = jnp.einsum("nhd,hd->nh", k_idx, q_idx) * scale
            logits = jnp.where(nodes_mask[:, None], raw, -self.mask_inf)

            def process_head(hl, hv):
                # FIXED: safe_softmax prevents NaN on all-masked rows
                alpha = safe_softmax(hl)
                return jnp.sum(alpha[..., None] * hv, axis=0)

            s = jax.vmap(process_head, in_axes=(1, 1), out_axes=0)(logits, v_idx)
            return self.Vo(self.Uo(s.ravel()) * so)

        return jax.vmap(action_attention, in_axes=(0, 0, 0))(
            actors["actor_nodes"],
            actors["mask"],
            actors["action_select"],
        )


class CCATBlock(nn.Module):
    d_in:        int
    d_mid:       int
    num_actions: int
    num_heads:   int = 4

    def setup(self):
        self.attention = CCATLayer(self.d_in, self.num_actions, self.num_heads)
        self.ffn       = CFFNHead(self.d_in, self.d_mid)
        self.att_norm  = nn.RMSNorm()
        self.ffn_norm  = nn.RMSNorm()

    def __call__(self, nodes, actors, sk, so, sq, sv, s_gate, s_up, s_down, a):
        h_att  = self.attention(self.att_norm(nodes), actors, sk, so, sq, sv)
        h_ffn  = h_att + self.ffn(self.ffn_norm(h_att), s_gate, s_up, s_down)
        return h_ffn, h_ffn @ a


# ---------------------------------------------------------------------------
# Actor: segmented pooling (one summary vector per axiom group)
# ---------------------------------------------------------------------------

class SegmentedPoolingLayer(nn.Module):
    d_in:      int
    num_heads: int = 4

    def setup(self):
        assert self.d_in % self.num_heads == 0
        self.d_head = self.d_in // self.num_heads
        self.Uk = CayleyDense(self.d_in); self.Vk = CayleyDense(self.d_in)
        self.Uv = CayleyDense(self.d_in); self.Vv = CayleyDense(self.d_in)
        self.Uo = CayleyDense(self.d_in); self.Vo = CayleyDense(self.d_in)

    def __call__(self, nodes, q, sk, sv, so, group_ids, num_groups):
        k = self.Vk(self.Uk(nodes) * sk).reshape(-1, self.num_heads, self.d_head)
        v = self.Vv(self.Uv(nodes) * sv).reshape(-1, self.num_heads, self.d_head)
        scale  = 1.0 / jnp.sqrt(self.d_head)
        logits = jnp.einsum("nhd,hd->nh", k, q.reshape(self.num_heads, self.d_head)) * scale

        def process_head(hl, hv):
            g_max   = jax.ops.segment_max(hl, group_ids, num_groups)
            shifted = hl - g_max[group_ids]
            exp_w   = jnp.exp(shifted)
            norm    = jax.ops.segment_sum(exp_w, group_ids, num_groups)
            alpha   = exp_w / (norm[group_ids] + 1e-10)
            return jax.ops.segment_sum(alpha[..., None] * hv, group_ids, num_groups)

        pooled = jax.vmap(process_head, in_axes=(1, 1), out_axes=0)(logits, v)
        combined = pooled.transpose(1, 0, 2).reshape(num_groups, -1)
        return self.Vo(self.Uo(combined) * so)


class SegmentedPoolingBlock(nn.Module):
    d_in:      int
    num_heads: int

    def setup(self):
        self.layer = SegmentedPoolingLayer(self.d_in, self.num_heads)
        self.norm  = nn.RMSNorm()

    def __call__(self, x, q, sk, sv, so, group_ids, num_groups, a):
        return self.layer(self.norm(x), q, sk, sv, so, group_ids, num_groups) @ a


# ---------------------------------------------------------------------------
# Actor network
# ---------------------------------------------------------------------------

class ActorNetwork(nn.Module):
    d_in:        int
    d_mid:       int
    num_actions: int
    num_heads:   int = 4

    def setup(self):
        self.attention  = CCATBlock(self.d_in, self.d_mid, self.num_actions, self.num_heads)
        self.pool_axiom = SegmentedPoolingBlock(self.d_in, self.num_heads)

    def __call__(
        self, nodes, actors,
        sk_att, so_att, sq_att, sv_att,
        s_gate, s_up, s_down, a_att,
        sk_pool, so_pool, sv_pool, q_pool, a_pool,
    ):
        actions, logits = self.attention(
            nodes, actors,
            sk_att, so_att, sq_att, sv_att,
            s_gate, s_up, s_down, a_att,
        )
        group_ids   = actors["action_select"]
        num_groups  = self.num_actions + 1
        axiom_logits = self.pool_axiom(
            actions, q_pool, sk_pool, sv_pool, so_pool,
            group_ids, num_groups, a_pool,
        )
        return axiom_logits, logits


# ---------------------------------------------------------------------------
# Critic network
# ---------------------------------------------------------------------------

class CPoolingLayer(nn.Module):
    d_in:      int
    num_heads: int = 4

    def setup(self):
        assert self.d_in % self.num_heads == 0
        self.d_head = self.d_in // self.num_heads
        self.Uk = CayleyDense(self.d_in); self.Vk = CayleyDense(self.d_in)
        self.Uv = CayleyDense(self.d_in); self.Vv = CayleyDense(self.d_in)
        self.Uo = CayleyDense(self.d_in); self.Vo = CayleyDense(self.d_in)

    def __call__(self, nodes, q, sk, sv, so):
        k = self.Vk(self.Uk(nodes) * sk).reshape(-1, self.num_heads, self.d_head)
        v = self.Vv(self.Uv(nodes) * sv).reshape(-1, self.num_heads, self.d_head)
        scale  = 1.0 / jnp.sqrt(self.d_head)
        logits = jnp.einsum("nhd,hd->nh", k, q.reshape(self.num_heads, self.d_head)) * scale

        def process_head(hl, hv):
            alpha = jax.nn.softmax(hl)
            return jnp.sum(alpha[..., None] * hv, axis=0)

        s = jax.vmap(process_head, in_axes=(1, 1), out_axes=0)(logits, v)
        return self.Vo(self.Uo(s.ravel()) * so)


class CPoolingBlock(nn.Module):
    d_in:      int
    d_mid:     int
    num_heads: int = 4

    def setup(self):
        self.att_norm = nn.RMSNorm()
        self.attention = CPoolingLayer(self.d_in, self.num_heads)
        self.ffn_norm  = nn.RMSNorm()
        self.ffn       = CFFNHead(self.d_in, self.d_mid)

    def __call__(self, x, q, sk, sv, so, s_gate, s_up, s_down):
        pooled = self.attention(self.att_norm(x), q, sk, sv, so)
        return self.ffn(self.ffn_norm(pooled), s_gate, s_up, s_down)


class CriticNetwork(nn.Module):
    d_in:      int
    d_mid:     int
    d_out:     int
    num_heads: int = 4

    def setup(self):
        self.pooling = CPoolingBlock(self.d_in, self.d_mid, self.num_heads)
        self.W_out   = _make_skip(self.d_mid, self.d_out)   # reuse skip helper

    def __call__(self, nodes, q, sk, sv, so, s_gate, s_up, s_down, s_out):
        out = self.pooling(nodes, q, sk, sv, so, s_gate, s_up, s_down)
        return self.W_out(out) * s_out
