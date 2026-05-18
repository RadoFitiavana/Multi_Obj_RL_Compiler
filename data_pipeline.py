"""
data_pipeline.py
================
JAX / jraph data pipeline:
  - add_virtual_node
  - build_links (for link-prediction pre-training)
  - prepare_first_pass, get_caps, prepare_batch, data_loader, split_dataset
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import jraph


# ---------------------------------------------------------------------------
# Virtual node injection
# ---------------------------------------------------------------------------

def add_virtual_node(graph: dict) -> dict:
    """
    Prepend a zero-feature virtual node at index 0 and connect it
    bi-directionally to every real node.

    Existing edge indices are shifted by +1 so real spiders start at 1.
    The match_matrix produced by get_match_data stores 0-based spider
    indices; CCAT_layer will look up k[match_matrix] in the encoder
    output h which includes this virtual node at index 0.
    To align properly, match_matrix values should be shifted +1 here —
    but we do that inside CCAT at index time to keep match_matrix clean.

    NOTE: the virtual node's embedding is initialised to zero;
    Input_block.inject_virtual_nodes will overwrite it with a pooled
    graph summary before the encoder processes the graph.
    """
    nodes = graph["nodes"]
    n_real = nodes.shape[0]
    n_feat = nodes.shape[1]

    virtual_feat = np.zeros((1, n_feat), dtype=np.float32)
    new_nodes = np.vstack([virtual_feat, nodes])

    real_idx = np.arange(1, n_real + 1)
    virt_idx = np.zeros(n_real, dtype=np.int32)

    # Virtual ↔ real edges
    v_send = np.concatenate([virt_idx, real_idx])
    v_recv = np.concatenate([real_idx, virt_idx])

    # Shift existing edges
    shifted_send = graph["senders"] + 1
    shifted_recv = graph["receivers"] + 1

    final_send = np.concatenate([v_send, shifted_send]).astype(np.int32)
    final_recv = np.concatenate([v_recv, shifted_recv]).astype(np.int32)

    return {
        "nodes":     new_nodes,
        "edges":     None,
        "senders":   final_send,
        "receivers": final_recv,
        "n_node":    np.array([new_nodes.shape[0]], dtype=np.int32),
        "n_edge":    np.array([final_recv.shape[0]], dtype=np.int32),
        "globals":   None,
    }


# ---------------------------------------------------------------------------
# Link building (used in VAE pre-training)
# ---------------------------------------------------------------------------

def build_links(graph: dict) -> dict:
    """
    Classify every pair of real spider nodes as a positive (edge exists)
    or negative (no edge) link.  Indices are shifted +1 for virtual-node
    alignment.
    """
    n_atoms = int(graph["n_node"])

    existing = set()
    for s, r in zip(graph["senders"], graph["receivers"]):
        existing.add((min(s, r), max(s, r)))

    pos_s, pos_r, neg_s, neg_r = [], [], [], []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if (i, j) in existing:
                pos_s.append(i); pos_r.append(j)
            else:
                neg_s.append(i); neg_r.append(j)

    return {
        "pos_senders":   np.array(pos_s, dtype=np.int32) + 1,
        "pos_receivers": np.array(pos_r, dtype=np.int32) + 1,
        "neg_senders":   np.array(neg_s, dtype=np.int32) + 1,
        "neg_receivers": np.array(neg_r, dtype=np.int32) + 1,
        "n_pos": len(pos_s),
        "n_neg": len(neg_s),
    }


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_first_pass(raw_graphs: list) -> dict:
    """
    Add virtual nodes and build link data for a list of raw numpy graph dicts.
    Returns {"graph_dataset": [...], "link_dataset": [...]}.
    """
    graphs, links = [], []
    for g in raw_graphs:
        graphs.append(add_virtual_node(g))
        links.append(build_links(g))
    return {"graph_dataset": graphs, "link_dataset": links}


def get_caps(graphs: list, links: list) -> dict:
    """Compute padding caps across the whole dataset."""
    max_nodes = max(g["nodes"].shape[0] for g in graphs)
    max_edges = max(g["senders"].shape[0] for g in graphs)
    max_pos   = max(l["n_pos"] for l in links)
    max_neg   = max(l["n_neg"] for l in links)
    return {
        "max_nodes": max_nodes,
        "max_edges": max_edges,
        "max_pos":   max_pos,
        "max_neg":   max_neg,
    }


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------

def prepare_batch(
    graphs: list,
    links: list,
    caps: dict,
    batch_size: int,
) -> dict:
    """
    Collate a list of jraph.GraphsTuple objects and their link data
    into a single padded batch dict.
    """
    batched = jraph.batch(graphs)
    offsets = np.cumsum(np.array(batched.n_node)) - np.array(batched.n_node)

    total_pos = batch_size * caps["max_pos"]
    total_neg = batch_size * caps["max_neg"]

    pos_s    = np.zeros(total_pos, dtype=np.int32)
    pos_r    = np.zeros(total_pos, dtype=np.int32)
    pos_mask = np.zeros(total_pos, dtype=np.float32)
    neg_s    = np.zeros(total_neg, dtype=np.int32)
    neg_r    = np.zeros(total_neg, dtype=np.int32)
    neg_mask = np.zeros(total_neg, dtype=np.float32)

    hp, hn = 0, 0
    for i, lk in enumerate(links):
        off = int(offsets[i])

        np_ = len(lk["pos_senders"])
        pos_s   [hp : hp + np_] = lk["pos_senders"]   + off
        pos_r   [hp : hp + np_] = lk["pos_receivers"]  + off
        pos_mask[hp : hp + np_] = 1.0
        hp += caps["max_pos"]

        nn_ = len(lk["neg_senders"])
        neg_s   [hn : hn + nn_] = lk["neg_senders"]   + off
        neg_r   [hn : hn + nn_] = lk["neg_receivers"]  + off
        neg_mask[hn : hn + nn_] = 1.0
        hn += caps["max_neg"]

    pad_n_node = caps["max_nodes"] * (batch_size + 1)
    pad_n_edge = caps["max_edges"] * (batch_size + 1)
    padded = jraph.pad_with_graphs(batched, pad_n_node, pad_n_edge, batch_size + 1)

    phases = np.array(padded.nodes[:, :2])
    roles  = np.array(padded.nodes[:, 2:])

    n_real = int(jnp.sum(padded.n_node[:-1]))
    node_mask = np.zeros(pad_n_node, dtype=np.int32)
    node_mask[:n_real] = 1

    return {
        "graph": padded,
        "link": {
            "pos_s": pos_s, "pos_r": pos_r, "pos_mask": pos_mask,
            "neg_s": neg_s, "neg_r": neg_r, "neg_mask": neg_mask,
        },
        "phase":     phases,
        "role":      roles,
        "node_mask": node_mask,
    }


def data_loader(dataset: dict, caps: dict, batch_size: int = 16):
    """Yield padded batches from the dataset."""
    raw_graphs = dataset["graph_dataset"]
    links      = dataset["link_dataset"]

    for i in range(0, len(raw_graphs), batch_size):
        g_raw = raw_graphs[i : i + batch_size]
        l_raw = links     [i : i + batch_size]

        if len(g_raw) < batch_size:
            break

        g_jraph = [
            jraph.GraphsTuple(**g) if isinstance(g, dict) else g
            for g in g_raw
        ]
        yield prepare_batch(g_jraph, l_raw, caps, batch_size)


def split_dataset(dataset: dict, train_ratio: float = 0.8, seed: int = 42) -> tuple:
    """Split dataset into train / val dicts with matching indices."""
    n = len(dataset["graph_dataset"])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    split = int(n * train_ratio)
    train_idx, val_idx = idx[:split], idx[split:]

    def _subset(idx_list):
        return {
            "graph_dataset": [dataset["graph_dataset"][i] for i in idx_list],
            "link_dataset":  [dataset["link_dataset"][i]  for i in idx_list],
        }

    return _subset(train_idx), _subset(val_idx)
