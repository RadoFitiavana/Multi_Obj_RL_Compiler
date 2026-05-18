"""
graph_utils.py
==============
PyZX graph construction, rewriting utilities, and match-data extraction.

All functions here are pure Python / NumPy — no JAX dependency.
"""

from fractions import Fraction

import numpy as np
import pyzx as zx


# ---------------------------------------------------------------------------
# Circuit sampling
# ---------------------------------------------------------------------------

def set_numpy_reproducibility(seed: int = 42) -> None:
    np.random.seed(seed)


def sample_circuit(
    n_qubits: int,
    depth: int,
    p_X: float,
    p_Z: float,
    p_H: float,
    p_CNOT: float,
    p_I: float,
    p_is_T: float,
) -> zx.Circuit:
    """
    Sample a random ZX circuit using NumPy RNG.
    Gate weights are normalised internally so they don't need to sum to 1.
    """
    c = zx.Circuit(n_qubits)
    gate_types = ["X", "Z", "H", "CNOT", "I"]
    weights = np.array([p_X, p_Z, p_H, p_CNOT, p_I], dtype=float)
    weights /= weights.sum()

    for _ in range(depth):
        available = np.random.permutation(n_qubits).tolist()

        while available:
            choice = np.random.choice(gate_types, p=weights)

            if choice == "I" or not available:
                available.pop()

            elif choice == "CNOT":
                if len(available) >= 2:
                    i, j = available.pop(), available.pop()
                    c.add_gate("CNOT", i, j)
                else:
                    available.pop()

            elif choice == "H":
                c.add_gate("HAD", available.pop())

            elif choice in ("X", "Z"):
                q = available.pop()
                if np.random.random() > p_is_T:
                    phase = Fraction(int(np.random.randint(1, 4)), 2)
                else:
                    phase = Fraction(int(np.random.choice([1, 3, 5, 7])), 4)
                gate = "ZPhase" if choice == "Z" else "XPhase"
                c.add_gate(gate, q, phase=phase)

    return c


def generate_circuit_set(
    min_n_qubits: int,
    max_n_qubits: int,
    min_depth: int,
    max_depth: int,
    n_circuits: int,
) -> list:
    """
    Generate a balanced set of random circuits across the qubit range.
    Returns a list of zx.Circuit objects.
    """
    qubit_range = np.arange(min_n_qubits, max_n_qubits + 1)
    per_class = n_circuits // len(qubit_range)
    remainder = n_circuits % len(qubit_range)

    dataset = []
    for n_qubits in qubit_range:
        count = per_class + (1 if remainder > 0 else 0)
        remainder -= 1
        depths = np.random.randint(min_depth, max_depth + 1, size=count)

        for depth in depths:
            probs = np.random.dirichlet([1, 1, 1, 1, 1])
            p_X, p_Z, p_H, p_CNOT, p_I = probs
            p_is_T = np.random.random()
            circ = sample_circuit(n_qubits, int(depth), p_X, p_Z, p_H, p_CNOT, p_I, p_is_T)
            circ.name = f"Q{n_qubits}_D{depth}_T{p_is_T:.2f}"
            dataset.append(circ)

    np.random.shuffle(dataset)
    return dataset


# ---------------------------------------------------------------------------
# Graph preprocessing
# ---------------------------------------------------------------------------

def pad_graph_like(g: zx.Graph) -> zx.Graph:
    """
    Ensure every boundary edge is a SIMPLE edge.
    Inserts identity (0-phase Z) spiders on Hadamard boundary edges.
    """
    boundary = list(g.inputs()) + list(g.outputs())

    for b in boundary:
        for s in list(g.neighbors(b)):
            if not g.connected(b, s):
                continue
            if g.edge_type(g.edge(b, s)) == zx.EdgeType.HADAMARD:
                new_v = g.add_vertex(
                    ty=zx.VertexType.Z,
                    phase=0,
                    qubit=g.qubit(b),
                    row=(g.row(b) + g.row(s)) / 2,
                )
                g.remove_edge(g.edge(b, s))
                g.add_edge((b, new_v), edgetype=zx.EdgeType.SIMPLE)
                g.add_edge((new_v, s), edgetype=zx.EdgeType.HADAMARD)

    return g


def circuit_to_numpy_graph(g: zx.Graph) -> dict:
    """
    Convert a PyZX graph-like diagram to a dict of NumPy arrays
    suitable for wrapping in jraph.

    Node features (5-dim):
        [cos(phase), sin(phase), is_input_neighbour, is_inner, is_output_neighbour]

    Returns keys: nodes, senders, receivers, n_node, n_edge
    """
    inputs = set(g.inputs())
    outputs = set(g.outputs())
    boundary = inputs | outputs
    spiders = sorted(v for v in g.vertices() if v not in boundary)
    node_map = {v: i for i, v in enumerate(spiders)}

    node_features = []
    for v in spiders:
        phase_rad = float(g.phase(v)) * np.pi
        is_in = any(n in inputs for n in g.neighbors(v))
        is_out = any(n in outputs for n in g.neighbors(v))
        is_inner = not is_in and not is_out
        node_features.append([
            np.cos(phase_rad),
            np.sin(phase_rad),
            float(is_in),
            float(is_inner),
            float(is_out),
        ])

    senders, receivers = [], []
    for u, v in g.edge_set():
        if u in node_map and v in node_map:
            ui, vi = node_map[u], node_map[v]
            senders.extend([ui, vi])
            receivers.extend([vi, ui])

    return {
        "nodes": np.array(node_features, dtype=np.float32),
        "senders": np.array(senders, dtype=np.int32),
        "receivers": np.array(receivers, dtype=np.int32),
        "n_node": np.array([len(spiders)], dtype=np.int32),
        "n_edge": np.array([len(senders)], dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# Random rewriting (for dataset augmentation)
# ---------------------------------------------------------------------------

def apply_random_rewrites(g: zx.Graph, max_steps: int = 10) -> zx.Graph:
    """Apply a random walk of simplification rules to g (in-place)."""
    g.track_phases = False
    rules = [
        zx.spider_simp,
        zx.id_simp,
        zx.pivot_simp,
        zx.lcomp_simp,
        zx.gadget_simp,
        zx.clifford_simp,
    ]
    n_steps = np.random.randint(1, max_steps + 1)
    for _ in range(n_steps):
        if np.random.random() > 0.9:
            zx.full_reduce(g)
        else:
            rules[np.random.randint(len(rules))](g)
    return g


def generate_graph_variants(base_circuit: zx.Circuit, num_variants: int = 10):
    """
    Produce `num_variants` semantically equivalent graph representations
    of `base_circuit`.  The first variant is the clean graph-like form;
    the rest are random rewrites.

    Returns (pyzx_graphs, numpy_graph_dicts).
    """
    g_start = base_circuit.to_graph()
    zx.to_graph_like(g_start)

    pyzx_variants, numpy_variants = [], []
    for i in range(num_variants):
        g = g_start.copy()
        g.track_phases = False
        if i > 0:
            g = apply_random_rewrites(g)
            zx.to_graph_like(g)
        g = pad_graph_like(g)
        pyzx_variants.append(g)
        numpy_variants.append(circuit_to_numpy_graph(g))

    return pyzx_variants, numpy_variants


# ---------------------------------------------------------------------------
# Match-data extraction
# ---------------------------------------------------------------------------

def get_match_data(g: zx.Graph, axiom_names: list):
    """
    Extract structured match data for every applicable rewrite in `g`.

    Parameters
    ----------
    g            : PyZX graph (will not be modified).
    axiom_names  : ordered list of rule names to query.

    Returns
    -------
    match_matrix : int32 (n_matches, num_v)
                   Row i lists the 0-based spider indices involved in match i,
                   padded with 0 on the right.
                   Values are 0-based (NOT +1 offset).
    mask_matrix  : int32 (n_matches, num_v)
                   1 where match_matrix[i,j] is a real index, 0 for padding.
    segment_ids  : int32 (n_matches,)
                   axiom index for each match row; len(axiom_names) = STOP.
    raw_matches  : list
                   Original PyZX match objects (for apply_rule).

    NOTE: match_matrix stores 0-based indices that align directly with
    the node matrix produced by circuit_to_numpy_graph / the encoder h.
    The old code stored +1 offsets intended for add_virtual_node indexing,
    which caused OOB gather reads inside CCAT.  Fixed here.
    """
    boundary = list(g.inputs()) + list(g.outputs())
    spiders = sorted(v for v in g.vertices() if v not in boundary)
    v_map = {v: i for i, v in enumerate(spiders)}
    num_v = len(spiders)

    rule_map = {
        "id_removal":     zx.id_simp.find_all_matches,
        "gadget_fusion":  zx.merge_phase_gadget_rule.match_phase_gadgets,
        "pivot":          zx.pivot_simp.find_all_matches,
        "lcomp":          zx.lcomp_simp.find_all_matches,
        "pivot_boundary": zx.pivot_rule.match_pivot_boundary,
        "pivot_gadget":   zx.pivot_rule.match_pivot_gadget,
    }

    all_involved: list[list[int]] = []
    raw_matches: list = []
    segment_ids: list[int] = []

    for axiom_idx, name in enumerate(axiom_names):
        if name == "unfusion":
            for v in spiders:
                if g.vertex_degree(v) > 2:
                    involved = [v] + [n for n in g.neighbors(v) if n in v_map]
                    all_involved.append(sorted(set(involved) & v_map.keys()))
                    raw_matches.append(v)
                    segment_ids.append(axiom_idx)
            continue

        rule_fn = rule_map.get(name)
        if rule_fn is None:
            continue

        for m in rule_fn(g):
            raw_matches.append(m)
            segment_ids.append(axiom_idx)

            if name == "id_removal":
                involved = {m} | set(g.neighbors(m))

            elif name == "pivot":
                u, v = m
                involved = {u, v} | set(g.neighbors(u)) | set(g.neighbors(v))

            elif name == "pivot_boundary":
                (u, v), (bounds, targets) = m
                involved = (
                    {u, v}
                    | set(g.neighbors(u))
                    | set(g.neighbors(v))
                    | set(bounds)
                    | set(targets)
                )

            elif name == "pivot_gadget":
                (u, v), lists = m
                involved = {u, v} | set(g.neighbors(u)) | set(g.neighbors(v))
                for sub in lists:
                    involved.update(sub)

            elif name == "lcomp":
                involved = {m} | set(g.neighbors(m))

            elif name == "gadget_fusion":
                involved = set(m)
                for ps in m:
                    hubs = list(g.neighbors(ps))
                    involved.update(hubs)
                    for h in hubs:
                        involved.update(g.neighbors(h))
            else:
                involved = set()

            filtered = sorted(involved & v_map.keys())
            if filtered:
                all_involved.append(filtered)

    # STOP action — all spiders are "involved"
    all_involved.append(list(range(num_v)))
    segment_ids.append(len(axiom_names))
    raw_matches.append(None)

    n_matches = len(all_involved)
    match_matrix = np.zeros((n_matches, num_v), dtype=np.int32)
    mask_matrix = np.zeros((n_matches, num_v), dtype=np.int32)

    for i, nodes in enumerate(all_involved):
        for j, vid in enumerate(nodes):
            # 0-based index — aligns directly with encoder node matrix
            match_matrix[i, j] = v_map[vid]
            mask_matrix[i, j] = 1

    return match_matrix, mask_matrix, np.array(segment_ids, dtype=np.int32), raw_matches


# ---------------------------------------------------------------------------
# Spider unfusion
# ---------------------------------------------------------------------------

def fair_spider_unfusion(g: zx.Graph, v: int) -> zx.Graph:
    """
    Tensor-preserving, graph-like spider unfusion.
    Splits v's neighbours 50/50 and bridges via H-[0-phase]-H.
    """
    neighbors = list(g.neighbors(v))
    if len(neighbors) <= 2:
        return g

    count_to_move = len(neighbors) // 2
    to_move = neighbors[:count_to_move]

    v_new = g.add_vertex(zx.VertexType.Z, phase=0)
    for n in to_move:
        e = g.edge(v, n)
        et = g.edge_type(e)
        g.remove_edge(e)
        g.add_edge(g.edge(v_new, n), et)

    v_mid = g.add_vertex(zx.VertexType.Z, phase=0)
    g.add_edge(g.edge(v, v_mid), zx.EdgeType.HADAMARD)
    g.add_edge(g.edge(v_new, v_mid), zx.EdgeType.HADAMARD)
    return g
