"""
vae_training.py
===============
Loss functions, metrics, and training/validation steps for the
Graph VAE (pre-training phase).

All losses are written to be NaN-free:
  - kl_loss: correct KL(N(mu,diag(sigma2)) || N(0,I)) formula
  - predict_phase: safe normalisation with eps=1e-6
  - log operations are always guarded with jnp.clip
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import jraph
from flax.training import train_state


# ---------------------------------------------------------------------------
# Reparametrisation sampling
# ---------------------------------------------------------------------------

def sample_embeddings(
    key: jax.Array,
    mu: jax.Array,
    sigma: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    """
    z = mu + eps * sigma, eps ~ N(0,I).
    Noise is applied only to real (masked) nodes.
    """
    eps = jax.random.normal(key, shape=mu.shape) * sigma
    return mu + eps * mask[:, None]


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def kl_loss(mu: jax.Array, sigma2: jax.Array, mask: jax.Array) -> jax.Array:
    """
    KL( N(mu, diag(sigma2)) || N(0, I) )
    = 0.5 * ( tr(sigma2) + ||mu||^2 - d - log det(sigma2) )

    Parameters
    ----------
    mu     : (N, d)  — mean embeddings
    sigma2 : (d,)    — per-dim variances from VariancesNetwork
    mask   : (N,)    — 1 for real nodes, 0 for padding
    """
    d = sigma2.shape[0]
    # Guard against non-positive sigma2 (shouldn't happen with
    # VariancesNetwork but clip defensively)
    sigma2_safe = jnp.clip(sigma2, 1e-6, None)

    log_det = jnp.sum(jnp.log(sigma2_safe))
    tr      = jnp.sum(sigma2_safe)

    n_real   = jnp.maximum(1.0, jnp.sum(mask))
    total_mu2 = jnp.sum((mu * mask[:, None]) ** 2)

    return 0.5 * (tr + total_mu2 - d - log_det)


def adjacency_loss(
    p_logits: jax.Array,
    n_logits: jax.Array,
    links: dict,
) -> jax.Array:
    """
    Balanced BCE link-prediction loss.
    Positive and negative contributions are averaged separately then combined.
    """
    pos_loss = -jax.nn.log_sigmoid( p_logits) * links["pos_mask"]
    neg_loss = -jax.nn.log_sigmoid(-n_logits) * links["neg_mask"]

    pos_mean = jnp.sum(pos_loss) / jnp.maximum(1.0, jnp.sum(links["pos_mask"]))
    neg_mean = jnp.sum(neg_loss) / jnp.maximum(1.0, jnp.sum(links["neg_mask"]))

    return 0.4 * pos_mean + 0.6 * neg_mean


def phase_loss(
    preds: jax.Array,
    targets: jax.Array,
    mask: jax.Array,
    kappa: float = 1.0,
) -> jax.Array:
    """
    Von Mises loss: kappa * mean(1 - cos(theta_pred - theta_target)).
    preds and targets are 2D unit vectors [cos, sin].
    Virtual / padding nodes are excluded via mask.
    """
    cos_sim = jnp.sum(preds * targets, axis=-1)
    loss    = (1.0 - cos_sim) * mask
    return kappa * jnp.sum(loss) / jnp.maximum(1.0, jnp.sum(mask))


def role_loss(
    logits: jax.Array,
    targets: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    """
    Softmax cross-entropy for in/inner/out classification.
    targets: (N, 3) one-hot.  mask: (N,).
    """
    individual = optax.softmax_cross_entropy(logits, targets)
    return jnp.sum(individual * mask) / jnp.maximum(1.0, jnp.sum(mask))


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_link(embeddings: jax.Array, links: dict):
    """Dot-product similarity for positive and negative node pairs."""
    p_logits = jnp.sum(embeddings[links["pos_s"]] * embeddings[links["pos_r"]], axis=-1)
    n_logits = jnp.sum(embeddings[links["neg_s"]] * embeddings[links["neg_r"]], axis=-1)
    return p_logits, n_logits


def predict_phase(embeddings: jax.Array, mask: jax.Array) -> jax.Array:
    """
    L2-normalise embeddings to unit vectors for Von Mises phase prediction.
    Uses eps=1e-6 to avoid exploding gradients near zero (1e-15 gives
    d/dx sqrt(x) ~ 1.6e7 which destabilises training).
    Masked (padding) nodes output zero vectors.
    """
    norm2 = jnp.sum(embeddings ** 2, axis=-1)
    norm  = jnp.sqrt(jnp.maximum(norm2, 1e-6))
    return (embeddings / norm[:, None]) * mask[:, None]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def adjacency_metric(p_logits, n_logits, links):
    p_correct = (p_logits >= 0) * links["pos_mask"]
    n_correct = (n_logits <= 0) * links["neg_mask"]
    p_acc = 100.0 * jnp.sum(p_correct) / jnp.maximum(1.0, jnp.sum(links["pos_mask"]))
    n_acc = 100.0 * jnp.sum(n_correct) / jnp.maximum(1.0, jnp.sum(links["neg_mask"]))
    return p_acc, n_acc


def phase_metric(preds, targets, mask):
    sims     = jnp.sum(preds * targets, axis=1)
    mask_sum = jnp.maximum(1.0, jnp.sum(mask))
    avg      = jnp.sum(sims * mask) / mask_sum
    var      = jnp.sum(((sims - avg) ** 2) * mask) / mask_sum
    return avg, jnp.sqrt(var)


def role_metric(logits, targets, mask):
    preds = jnp.argmax(logits, axis=-1)
    if targets.ndim == 2:
        targets = jnp.argmax(targets, axis=-1)
    accs = []
    for i in range(3):
        role_mask = (targets == i) * mask
        correct   = (preds == i) * role_mask
        accs.append(100.0 * jnp.sum(correct) / jnp.maximum(1.0, jnp.sum(role_mask)))
    return jnp.array(accs)


# ---------------------------------------------------------------------------
# Training steps (EM / coordinate-ascent)
# ---------------------------------------------------------------------------

def create_vae_states(
    rng,
    encoder_model,
    decoder_model,
    dummy_graph: jraph.GraphsTuple,
    d_model: int,
    lrs: dict,
):
    """Initialise encoder and decoder TrainStates."""
    import jax.numpy as jnp

    rng_enc, rng_dec = jax.random.split(rng)

    enc_vars = encoder_model.init(rng_enc, dummy_graph)
    enc_state = train_state.TrainState.create(
        apply_fn=encoder_model.apply,
        params=enc_vars["params"],
        tx=optax.adam(lrs["enc"]),
    )

    dummy_z   = jnp.zeros((dummy_graph.nodes.shape[0], d_model))
    dec_vars  = decoder_model.init(rng_dec, dummy_z)
    dec_state = train_state.TrainState.create(
        apply_fn=decoder_model.apply,
        params=dec_vars["params"],
        tx=optax.adam(lrs["dec"]),
    )

    return {"encoder": enc_state, "decoder": dec_state}


@jax.jit
def e_step(states, batch, lambdas, key, kappa):
    """Update encoder only (decoder frozen)."""
    enc, dec = states["encoder"], states["decoder"]

    def loss_fn(enc_params):
        mu, sigma = enc.apply_fn({"params": enc_params}, batch["graph"])
        z = sample_embeddings(key, mu, sigma, batch["node_mask"])

        h_link, h_phase, h_role = dec.apply_fn({"params": dec.params}, z)

        p_logits, n_logits = predict_link(h_link, batch["link"])
        phase_preds        = predict_phase(h_phase, batch["node_mask"])

        l_adj  = adjacency_loss(p_logits, n_logits, batch["link"])
        l_ph   = phase_loss(phase_preds, batch["phase"], batch["node_mask"], kappa)
        l_ro   = role_loss(h_role, batch["role"], batch["node_mask"])
        l_kl   = kl_loss(mu, sigma, batch["node_mask"])

        return (l_adj
                + lambdas["phase"] * l_ph
                + lambdas["role"]  * l_ro
                + lambdas["kl"]    * l_kl)

    loss, grads = jax.value_and_grad(loss_fn)(enc.params)
    new_enc     = enc.apply_gradients(grads=grads)
    return {**states, "encoder": new_enc}, loss


@jax.jit
def m_step(states, batch, lambdas, key, kappa):
    """Update decoder only (encoder frozen)."""
    enc, dec = states["encoder"], states["decoder"]

    mu, sigma = enc.apply_fn({"params": enc.params}, batch["graph"])
    l_kl      = kl_loss(mu, sigma, batch["node_mask"])

    def loss_fn(dec_params):
        z = sample_embeddings(key, mu, sigma, batch["node_mask"])
        h_link, h_phase, h_role = dec.apply_fn({"params": dec_params}, z)

        p_logits, n_logits = predict_link(h_link, batch["link"])
        phase_preds        = predict_phase(h_phase, batch["node_mask"])

        l_adj = adjacency_loss(p_logits, n_logits, batch["link"])
        l_ph  = phase_loss(phase_preds, batch["phase"], batch["node_mask"], kappa)
        l_ro  = role_loss(h_role, batch["role"], batch["node_mask"])

        return (l_adj
                + lambdas["phase"] * l_ph
                + lambdas["role"]  * l_ro
                + lambdas["kl"]    * l_kl)

    loss, grads = jax.value_and_grad(loss_fn)(dec.params)
    new_dec     = dec.apply_gradients(grads=grads)
    return {**states, "decoder": new_dec}, loss


@jax.jit
def validation_step(states, batch, lambdas, key, kappa):
    enc, dec = states["encoder"], states["decoder"]

    mu, sigma = enc.apply_fn({"params": enc.params}, batch["graph"])
    z         = mu   # deterministic at validation time

    h_link, h_phase, h_role = dec.apply_fn({"params": dec.params}, z)

    p_logits, n_logits = predict_link(h_link, batch["link"])
    phase_preds        = predict_phase(h_phase, batch["node_mask"])

    l_adj = adjacency_loss(p_logits, n_logits, batch["link"])
    l_ph  = phase_loss(phase_preds, batch["phase"], batch["node_mask"], kappa)
    l_ro  = role_loss(h_role, batch["role"], batch["node_mask"])
    l_kl  = kl_loss(mu, sigma, batch["node_mask"])
    total = l_adj + lambdas["phase"]*l_ph + lambdas["role"]*l_ro + lambdas["kl"]*l_kl

    p_acc, n_acc       = adjacency_metric(p_logits, n_logits, batch["link"])
    ph_avg, ph_std     = phase_metric(phase_preds, batch["phase"], batch["node_mask"])
    role_accs          = role_metric(h_role, batch["role"], batch["node_mask"])

    return {
        "loss": total, "l_adj": l_adj, "l_ph": l_ph, "l_ro": l_ro, "l_kl": l_kl,
        "adj_p_acc": p_acc,   "adj_n_acc": n_acc,
        "phase_sim": ph_avg,  "phase_std": ph_std,
        "role_in":   role_accs[0], "role_mid": role_accs[1], "role_out": role_accs[2],
    }


def train_epoch(states, dataset, caps, lambdas, key, kappa, n_e, n_m, batch_size):
    from data_pipeline import data_loader
    for batch in data_loader(dataset, caps, batch_size):
        for _ in range(n_e):
            key, sub = jax.random.split(key)
            states, _ = e_step(states, batch, lambdas, sub, kappa)
        for _ in range(n_m):
            key, sub = jax.random.split(key)
            states, _ = m_step(states, batch, lambdas, sub, kappa)
    return states


def validate_epoch(states, dataset, caps, lambdas, key, kappa, batch_size):
    from data_pipeline import data_loader
    metrics = []
    for batch in data_loader(dataset, caps, batch_size):
        key, sub = jax.random.split(key)
        metrics.append(validation_step(states, batch, lambdas, sub, kappa))
    return {k: jnp.mean(jnp.array([m[k] for m in metrics])) for k in metrics[0]}
