"""
vae_model.py
============
Graph VAE (GMAT encoder + decoder) implemented in Flax.

Architecture
------------
Encoder:
  Input_block  →  stack of GMAT_block  →  (mu, sigma)

Decoder:
  z  →  FFN heads  →  (link_emb, phase_emb, role_logits)

Cayley layers replace standard Dense layers for near-orthogonal weight
matrices, improving training stability with Cayley-transform parametrisation.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import jraph
import flax.linen as nn


# ---------------------------------------------------------------------------
# Orthogonal / Cayley weight layers
# ---------------------------------------------------------------------------

class CayleyDense(nn.Module):
    """Square orthogonal layer via Cayley transform: W = (I+A)^{-1}(I-A)."""
    d_in: int

    @nn.compact
    def __call__(self, x):
        M = self.param("cayley_param", nn.initializers.normal(0.02), (self.d_in, self.d_in))
        A = M - M.T
        I = jnp.eye(self.d_in, dtype=x.dtype)
        W = jnp.linalg.solve(I + A, I - A)
        return x @ W


class RowCayleyDense(nn.Module):
    """Rectangular expansion layer (d_in → d_out, d_out ≥ d_in)."""
    d_in: int
    d_out: int

    def setup(self):
        assert self.d_out >= self.d_in, "RowCayleyDense requires d_out >= d_in"
        self.M = self.param("cayley_param", nn.initializers.normal(0.02),
                            (self.d_out, self.d_out))

    def __call__(self, x):
        A = self.M - self.M.T
        I = jnp.eye(self.d_out, dtype=x.dtype)
        Q = jnp.linalg.solve(I + A, I - A)
        W = Q[:self.d_in, :]   # (d_in, d_out)
        return x @ W


class ColCayleyDense(nn.Module):
    """Rectangular contraction layer (d_in → d_out, d_in ≥ d_out)."""
    d_in: int
    d_out: int

    def setup(self):
        assert self.d_in >= self.d_out, "ColCayleyDense requires d_in >= d_out"
        self.M = self.param("cayley_param", nn.initializers.normal(0.02),
                            (self.d_in, self.d_in))

    def __call__(self, x):
        A = self.M - self.M.T
        I = jnp.eye(self.d_in, dtype=x.dtype)
        Q = jnp.linalg.solve(I + A, I - A)
        W = Q[:, :self.d_out]  # (d_in, d_out)
        return x @ W


# ---------------------------------------------------------------------------
# FFN building blocks
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward: W_down(silu(W_gate(x)) * W_up(x))."""
    d_in:  int
    d_mid: int

    def setup(self):
        init = nn.initializers.he_normal()
        self.W_up   = nn.Dense(self.d_mid, use_bias=False, kernel_init=init)
        self.W_gate = nn.Dense(self.d_mid, use_bias=False, kernel_init=init)
        self.W_down = nn.Dense(self.d_in,  use_bias=False, kernel_init=init)

    def __call__(self, x):
        return self.W_down(nn.silu(self.W_gate(x)) * self.W_up(x))


# ---------------------------------------------------------------------------
# Graph multi-head attention
# ---------------------------------------------------------------------------

class GMATLayer(nn.Module):
    """
    Graph Multi-head Attention Transform.
    Computes segment-softmax attention over edges for each head,
    aggregates via segment-sum, then projects to d_in.
    """
    d_in:      int
    num_heads: int

    def setup(self):
        assert self.d_in % self.num_heads == 0
        self.d_k = self.d_in // self.num_heads
        oi = nn.initializers.orthogonal()
        self.Wq = nn.Dense(self.d_in, use_bias=False, kernel_init=oi)
        self.Wk = nn.Dense(self.d_in, use_bias=False, kernel_init=oi)
        self.Wv = nn.Dense(self.d_in, use_bias=False, kernel_init=oi)
        self.Wo = nn.Dense(self.d_in, use_bias=False,
                           kernel_init=nn.initializers.glorot_normal())

    def __call__(self, graph: jraph.GraphsTuple) -> jnp.ndarray:
        h = graph.nodes
        N = h.shape[0]
        scale = 1.0 / jnp.sqrt(self.d_k)

        q = self.Wq(h).reshape(N, self.num_heads, self.d_k)
        k = self.Wk(h).reshape(N, self.num_heads, self.d_k)
        v = self.Wv(h).reshape(N, self.num_heads, self.d_k)

        q_i = q[graph.receivers]
        k_j = k[graph.senders]
        v_j = v[graph.senders]

        logits = jnp.sum(q_i * k_j, axis=-1) * scale  # (E, H)

        def process_head(head_logits, head_v_j):
            alpha = jraph.segment_softmax(head_logits, graph.receivers, N)
            return jraph.segment_sum(alpha[..., None] * head_v_j, graph.receivers, N)

        aggregated = jax.vmap(process_head, in_axes=1, out_axes=1)(logits, v_j)
        return self.Wo(aggregated.reshape(N, self.d_in))


class GMATBlock(nn.Module):
    """Pre-norm residual block: GMAT attention + SwiGLU FFN."""
    d_in:      int
    d_mid:     int
    num_heads: int

    def setup(self):
        self.attention = GMATLayer(self.d_in, self.num_heads)
        self.ffn       = SwiGLUFFN(self.d_in, self.d_mid)
        self.att_norm  = nn.RMSNorm()
        self.ffn_norm  = nn.RMSNorm()

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        h = graph.nodes
        h = h + self.attention(graph._replace(nodes=self.att_norm(h)))
        h = h + self.ffn(self.ffn_norm(h))
        return graph._replace(nodes=h)


# ---------------------------------------------------------------------------
# Input block (expansion + virtual-node initialisation)
# ---------------------------------------------------------------------------

class ExpandLayer(nn.Module):
    d_out: int

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        h = nn.Dense(self.d_out, use_bias=True,
                     kernel_init=nn.initializers.he_normal())(graph.nodes)
        return graph._replace(nodes=nn.silu(h))


class AveragingLayer(nn.Module):
    """Learned weighted average pooling per graph."""
    d_in: int

    def setup(self):
        self.C = self.param("kernel", nn.initializers.normal(0.02), (self.d_in,))

    def __call__(self, graph: jraph.GraphsTuple) -> jnp.ndarray:
        num_graphs = graph.n_node.shape[0]
        node_graph_idx = jnp.repeat(
            jnp.arange(num_graphs), graph.n_node,
            total_repeat_length=graph.nodes.shape[0])
        logits  = jnp.dot(graph.nodes, self.C)
        weights = jraph.segment_softmax(logits, node_graph_idx, num_graphs)
        return jraph.segment_sum(
            graph.nodes * weights[:, None], node_graph_idx, num_graphs)


class InputBlock(nn.Module):
    """Expand features to d_out, then seed the virtual node via pooling."""
    d_out: int

    def setup(self):
        self.expansion = ExpandLayer(self.d_out)
        self.averaging = AveragingLayer(self.d_out)

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        graph = self.expansion(graph)
        virtual_emb = self.averaging(graph)          # (n_graphs, d_out)

        # Virtual node is the first node of each graph
        offsets = jnp.cumsum(jnp.concatenate([
            jnp.array([0]), graph.n_node[:-1]]))
        new_nodes = graph.nodes.at[offsets].set(virtual_emb)
        return graph._replace(nodes=new_nodes)


# ---------------------------------------------------------------------------
# Variance network (learnable per-dim sigma)
# ---------------------------------------------------------------------------

class VariancesNetwork(nn.Module):
    """
    Produces a bounded per-dimension variance vector in [min_v, max_v].
    Uses soft_sign + tanh composite for smooth saturation.
    """
    d_sigma:  int
    min_v:    float = 0.2
    max_v:    float = 1.0

    def setup(self):
        self.sigma_param = self.param(
            "sigma_param",
            nn.initializers.truncated_normal(stddev=0.02),
            (self.d_sigma,))

    def __call__(self) -> jnp.ndarray:
        y = 0.5 * (jax.nn.soft_sign(self.sigma_param) + 1.0)
        return self.min_v + (self.max_v - self.min_v) * y


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    d_model:    int
    d_mid:      int
    num_heads:  int
    num_layers: int   = 4
    min_sigma:  float = 0.2
    max_sigma:  float = 1.0

    def setup(self):
        self.input_layer    = InputBlock(d_out=self.d_model)
        self.sigma_network  = VariancesNetwork(
            self.d_model, min_v=self.min_sigma, max_v=self.max_sigma)
        self.blocks = [
            GMATBlock(self.d_model, self.d_mid, self.num_heads, name=f"block_{i}")
            for i in range(self.num_layers)
        ]

    def __call__(self, graph: jraph.GraphsTuple):
        graph = self.input_layer(graph)
        for block in self.blocks:
            graph = block(graph)
        return graph.nodes, self.sigma_network()   # (N, d_model), (d_model,)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    Three-head decoder operating on node embeddings z:
      link_head  → embeddings for dot-product link prediction
      phase_head → 2-dim embeddings for Von Mises phase prediction
      role_head  → 3-dim logits for in/inner/out classification
    """
    d_mid:   int
    d_link:  int
    d_phase: int
    d_role:  int

    def setup(self):
        self.pre_norm       = nn.RMSNorm()
        self.link_head      = SwiGLUFFN(self.d_link,  self.d_mid)
        self.phase_head     = SwiGLUFFN(self.d_phase, self.d_mid)
        self.role_head      = SwiGLUFFN(self.d_role,  self.d_mid)
        self.post_norm_link  = nn.RMSNorm()
        self.post_norm_phase = nn.RMSNorm()
        self.post_norm_role  = nn.RMSNorm()

    def __call__(self, z: jnp.ndarray):
        z = self.pre_norm(z)
        h_link  = self.post_norm_link (self.link_head (z))
        h_phase = self.post_norm_phase(self.phase_head(z))
        h_role  = self.post_norm_role (self.role_head (z))
        return h_link, h_phase, h_role
