"""Hardware graph construction from FakeBackendV2.

Extracts noise properties from backend and builds a PyG Data object
with z-score normalized node and edge features.

Supports two backend types:
  - Qiskit FakeBackendV2: loaded via qiskit_ibm_runtime.fake_provider
  - Synthetic backends: loaded from JSON files in data/circuits/backends/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.sparse.csgraph import floyd_warshall
from torch_geometric.data import Data

from data.normalization import zscore_normalize

# Module-level configuration for HW feature dimensionality.
_include_t1_t2: bool = False
_exclude_degree: bool = False


def configure_hw_features(
    include_t1_t2: bool = False,
    exclude_degree: bool = False,
) -> None:
    """Set hardware feature mode. Call before building any hardware graphs.

    Args:
        include_t1_t2: If True, include raw t1/t2 in z-scored features (+2 dim).
        exclude_degree: If True, exclude degree from z-scored features (-1 dim).
    """
    global _include_t1_t2, _exclude_degree
    _include_t1_t2 = include_t1_t2
    _exclude_degree = exclude_degree

# Backend name -> class name mapping for qiskit_ibm_runtime.fake_provider
BACKEND_REGISTRY: dict[str, str] = {
    # --- 5Q backends ---
    "athens": "FakeAthensV2",
    "belem": "FakeBelemV2",
    "bogota": "FakeBogotaV2",
    "burlington": "FakeBurlingtonV2",
    "essex": "FakeEssexV2",
    "lima": "FakeLimaV2",
    "london": "FakeLondonV2",
    "manila": "FakeManilaV2",
    "ourense": "FakeOurenseV2",
    "quito": "FakeQuitoV2",
    "rome": "FakeRomeV2",
    "santiago": "FakeSantiagoV2",
    "valencia": "FakeValenciaV2",
    "vigo": "FakeVigoV2",
    "yorktown": "FakeYorktownV2",
    # --- 7Q backends ---
    "casablanca": "FakeCasablancaV2",
    "jakarta": "FakeJakartaV2",
    "lagos": "FakeLagosV2",
    "nairobi": "FakeNairobiV2",
    "oslo": "FakeOslo",
    "perth": "FakePerth",
    # --- 15-16Q backends ---
    "melbourne": "FakeMelbourneV2",
    "guadalupe": "FakeGuadalupeV2",
    # --- 20Q backends ---
    "almaden": "FakeAlmadenV2",
    "boeblingen": "FakeBoeblingenV2",
    "johannesburg": "FakeJohannesburgV2",
    "poughkeepsie": "FakePoughkeepsieV2",
    "singapore": "FakeSingaporeV2",
    # --- 27-28Q backends ---
    "algiers": "FakeAlgiers",
    "auckland": "FakeAuckland",
    "cairo": "FakeCairoV2",
    "cambridge": "FakeCambridgeV2",
    "geneva": "FakeGeneva",
    "hanoi": "FakeHanoiV2",
    "kolkata": "FakeKolkataV2",
    "montreal": "FakeMontrealV2",
    "mumbai": "FakeMumbaiV2",
    "paris": "FakeParisV2",
    "peekskill": "FakePeekskill",
    "sydney": "FakeSydneyV2",
    "toronto": "FakeTorontoV2",
    # --- 33Q backends ---
    "prague": "FakePrague",
    # --- 53Q backends ---
    "rochester": "FakeRochesterV2",
    # --- 65Q backends ---
    "brooklyn": "FakeBrooklynV2",
    "manhattan": "FakeManhattanV2",
    # --- 127Q backends ---
    "brisbane": "FakeBrisbane",
    "cusco": "FakeCusco",
    "kawasaki": "FakeKawasaki",
    "kyiv": "FakeKyiv",
    "kyoto": "FakeKyoto",
    "osaka": "FakeOsaka",
    "quebec": "FakeQuebec",
    "sherbrooke": "FakeSherbrooke",
    "washington": "FakeWashingtonV2",
    # --- 133Q backends ---
    "torino": "FakeTorino",
}


def get_backend(name: str) -> Any:
    """Instantiate a FakeBackendV2 by short name.

    Args:
        name: Backend short name (e.g. 'manila', 'toronto').

    Returns:
        Instantiated FakeBackendV2 object.
    """
    import qiskit_ibm_runtime.fake_provider as fp

    name_lower = name.lower()
    if name_lower not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend '{name}'. Available: {list(BACKEND_REGISTRY.keys())}"
        )
    cls_name = BACKEND_REGISTRY[name_lower]
    cls = getattr(fp, cls_name)
    return cls()


def extract_qubit_properties(backend: Any) -> dict[str, np.ndarray]:
    """Extract per-qubit noise properties from a backend.

    Args:
        backend: A FakeBackendV2 instance.

    Returns:
        Dict with keys: t1, t2, readout_error, single_qubit_error, degree,
        t1_cx_ratio, t2_cx_ratio.
        Each value is a 1D numpy array of shape (num_qubits,).
    """
    target = backend.target
    num_qubits = target.num_qubits

    t1 = np.zeros(num_qubits)
    t2 = np.zeros(num_qubits)
    readout_error = np.zeros(num_qubits)
    single_qubit_error = np.zeros(num_qubits)

    for q in range(num_qubits):
        qp = target.qubit_properties[q]
        t1[q] = qp.t1 if qp.t1 is not None else 0.0
        t2[q] = qp.t2 if qp.t2 is not None else 0.0

        # Readout error from measure gate
        if "measure" in target.operation_names:
            meas_props = target["measure"]
            if (q,) in meas_props and meas_props[(q,)] is not None:
                readout_error[q] = meas_props[(q,)].error or 0.0

        # Average single-qubit gate error (average over sx, x)
        sq_errors = []
        for gate_name in ["sx", "x"]:
            if gate_name in target.operation_names:
                gate_props = target[gate_name]
                if (q,) in gate_props and gate_props[(q,)] is not None:
                    err = gate_props[(q,)].error
                    if err is not None:
                        sq_errors.append(err)
        single_qubit_error[q] = np.mean(sq_errors) if sq_errors else 0.0

    # Degree from coupling map
    coupling_map = backend.coupling_map
    degree = np.zeros(num_qubits)
    neighbor_map: list[set[int]] = [set() for _ in range(num_qubits)]
    for edge in coupling_map.get_edges():
        neighbor_map[edge[0]].add(edge[1])
        neighbor_map[edge[1]].add(edge[0])
    for q in range(num_qubits):
        degree[q] = len(neighbor_map[q])

    # Per-qubit mean cx_duration (average over all connected edges)
    gate_name = _get_two_qubit_gate_name(backend)
    cx_props = target[gate_name]
    cx_duration_sum = np.zeros(num_qubits)
    cx_duration_count = np.zeros(num_qubits)
    for qargs, props in cx_props.items():
        if props is None or props.duration is None:
            continue
        p, q = qargs
        cx_duration_sum[p] += props.duration
        cx_duration_sum[q] += props.duration
        cx_duration_count[p] += 1
        cx_duration_count[q] += 1
    mean_cx_duration = np.where(
        cx_duration_count > 0,
        cx_duration_sum / cx_duration_count,
        1.0,  # fallback: avoid division by zero
    )

    # T1/cx_duration and T2/cx_duration: how many 2Q gates fit before decoherence
    t1_cx_ratio = np.where(mean_cx_duration > 0, t1 / mean_cx_duration, 0.0)
    t2_cx_ratio = np.where(mean_cx_duration > 0, t2 / mean_cx_duration, 0.0)

    # T2/T1 ratio: decoherence type indicator (dimensionless, not z-scored)
    # T2/T1 ≈ 2 → relaxation-limited, T2/T1 ≈ 1 → dephasing-dominated
    # Clipped to [0, 2] — theoretical bound T2 ≤ 2*T1; outliers from noisy calibration data
    t2_t1_ratio = np.clip(np.where(t1 > 0, t2 / t1, 0.0), 0.0, 2.0)

    return {
        "t1": t1,
        "t2": t2,
        "readout_error": readout_error,
        "single_qubit_error": single_qubit_error,
        "degree": degree,
        "t1_cx_ratio": t1_cx_ratio,
        "t2_cx_ratio": t2_cx_ratio,
        "t2_t1_ratio": t2_t1_ratio,
    }


def _get_two_qubit_gate_name(backend: Any) -> str:
    """Detect the native 2-qubit gate name for a backend.

    Checks for 'cx', 'ecr', and 'cz' in order of preference.

    Args:
        backend: A FakeBackendV2 instance.

    Returns:
        Name of the 2-qubit gate ('cx', 'ecr', or 'cz').

    Raises:
        ValueError: If no supported 2-qubit gate is found.
    """
    target = backend.target
    for gate_name in ("cx", "ecr", "cz"):
        if gate_name in target.operation_names:
            return gate_name
    raise ValueError(
        f"No supported 2-qubit gate (cx, ecr, cz) found in backend. "
        f"Available operations: {target.operation_names}"
    )


def extract_edge_properties(
    backend: Any,
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    """Extract per-edge 2-qubit gate properties from a backend.

    Supports backends with cx, ecr, or cz as the native 2-qubit gate.

    Args:
        backend: A FakeBackendV2 instance.

    Returns:
        Tuple of (edge_list, edge_error, edge_duration).
        edge_list: List of (src, dst) tuples (undirected, each pair once).
        edge_error: Array of shape (num_edges,) with 2q gate error.
        edge_duration: Array of shape (num_edges,) with 2q gate duration.
    """
    target = backend.target
    gate_name = _get_two_qubit_gate_name(backend)
    cx_props = target[gate_name]

    # Collect undirected edges (deduplicate by sorting, average both directions)
    edge_dict: dict[tuple[int, int], dict[str, list[float]]] = {}
    for qargs, props in cx_props.items():
        if props is None:
            continue
        a, b = qargs
        key = (min(a, b), max(a, b))
        error = props.error if props.error is not None else 0.0
        duration = props.duration if props.duration is not None else 0.0
        if key not in edge_dict:
            edge_dict[key] = {"errors": [error], "durations": [duration]}
        else:
            edge_dict[key]["errors"].append(error)
            edge_dict[key]["durations"].append(duration)

    edge_list = sorted(edge_dict.keys())
    edge_error = np.array(
        [np.mean(edge_dict[e]["errors"]) for e in edge_list], dtype=np.float32
    )
    edge_duration = np.array(
        [np.mean(edge_dict[e]["durations"]) for e in edge_list], dtype=np.float32
    )

    return edge_list, edge_error, edge_duration


def build_hardware_graph(backend: Any, eps: float = 1e-8) -> Data:
    """Build a PyG Data object for a hardware backend.

    Node features (6 or 8):
      z-scored: [t1, t2,] readout_error, single_qubit_error,
                degree, t1_cx_ratio, t2_cx_ratio
      raw:      t2_t1_ratio
    Set via configure_hw_features(include_t1_t2=True) for 8dim (default 6dim).

    Edge features (2):
      z-scored: 2q_gate_error
      raw:      edge_coherence_ratio (cx_duration / min(T1,T2) of endpoints)

    Args:
        backend: A FakeBackendV2 instance.
        eps: Epsilon for z-score normalization.

    Returns:
        PyG Data object with x, edge_index, edge_attr, and num_qubits.
    """
    qubit_props = extract_qubit_properties(backend)
    edge_list, edge_error, edge_duration = extract_edge_properties(backend)

    # -- Node features --
    # Z-scored features: (num_qubits, 5 or 7)
    zscore_feat_list = []
    if _include_t1_t2:
        zscore_feat_list.extend([qubit_props["t1"], qubit_props["t2"]])
    zscore_feat_list.extend([
        qubit_props["readout_error"],
        qubit_props["single_qubit_error"],
    ])
    if not _exclude_degree:
        zscore_feat_list.append(qubit_props["degree"])
    zscore_feat_list.extend([
        qubit_props["t1_cx_ratio"],
        qubit_props["t2_cx_ratio"],
    ])
    node_zscore = np.stack(zscore_feat_list, axis=1).astype(np.float32)
    node_zscore = zscore_normalize(torch.from_numpy(node_zscore), dim=0, eps=eps)

    # Raw features: (num_qubits, 1) — T2/T1 ratio (dimensionless, not z-scored)
    node_raw = torch.from_numpy(
        qubit_props["t2_t1_ratio"].reshape(-1, 1).astype(np.float32)
    )

    x = torch.cat([node_zscore, node_raw], dim=1)

    # -- Edge features --
    num_qubits = backend.target.num_qubits
    if len(edge_list) > 0:
        src = [e[0] for e in edge_list] + [e[1] for e in edge_list]
        dst = [e[1] for e in edge_list] + [e[0] for e in edge_list]
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        # Z-scored: 2q_gate_error
        edge_error_t = torch.from_numpy(edge_error.reshape(-1, 1))
        edge_error_t = torch.cat([edge_error_t, edge_error_t], dim=0)
        edge_error_t = zscore_normalize(edge_error_t, dim=0, eps=eps)

        # Raw: edge_coherence_ratio = cx_duration / min(T1_u, T1_v, T2_u, T2_v)
        t1 = qubit_props["t1"]
        t2 = qubit_props["t2"]
        coherence_ratio = np.zeros(len(edge_list), dtype=np.float32)
        for i, (u, v) in enumerate(edge_list):
            min_coherence = min(t1[u], t1[v], t2[u], t2[v])
            if min_coherence > 0:
                coherence_ratio[i] = edge_duration[i] / min_coherence
        coherence_t = torch.from_numpy(coherence_ratio.reshape(-1, 1))
        coherence_t = torch.cat([coherence_t, coherence_t], dim=0)

        edge_attr = torch.cat([edge_error_t, coherence_t], dim=1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.float32)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_qubits=num_qubits,
    )


def precompute_c_eff(backend: Any) -> np.ndarray:
    """Precompute unified effective cost matrix C_eff.

    C_eff encodes the total noise cost of executing a 2Q gate between two
    physical qubits, accounting for SWAP overhead (each SWAP = 3 CX gates):

        C_eff(j,k) = ε₂(j,k)      if (j,k) adjacent
        C_eff(j,k) = D_swap(j,k)   otherwise

    where D_swap is the shortest path on a graph weighted by 3·ε₂ per edge.
    All values are in additive -log(fidelity) units.

    Args:
        backend: A FakeBackendV2 instance.

    Returns:
        h×h numpy array of effective costs.
    """
    target = backend.target
    h = target.num_qubits
    gate_name = _get_two_qubit_gate_name(backend)
    cx_props = target[gate_name]

    # Build raw error matrix for adjacent pairs
    raw_error = np.full((h, h), np.inf)
    np.fill_diagonal(raw_error, 0)

    # Build SWAP-cost weighted adjacency for Floyd-Warshall
    swap_adj = np.full((h, h), np.inf)
    np.fill_diagonal(swap_adj, 0)

    for qargs, props in cx_props.items():
        if props is None:
            continue
        p, q = qargs
        error = props.error if props.error is not None else 0.0
        raw_error[p][q] = min(raw_error[p][q], error)
        raw_error[q][p] = min(raw_error[q][p], error)
        swap_cost = 3.0 * error  # SWAP = 3 CX gates
        swap_adj[p][q] = min(swap_adj[p][q], swap_cost)
        swap_adj[q][p] = min(swap_adj[q][p], swap_cost)

    # Floyd-Warshall on SWAP-cost graph
    d_swap = floyd_warshall(swap_adj).astype(np.float32)

    # Build C_eff: adjacent uses raw error, non-adjacent uses SWAP distance
    c_eff = d_swap.copy()
    adjacent = np.isfinite(raw_error) & (raw_error > 0)
    c_eff[adjacent] = raw_error[adjacent]
    np.fill_diagonal(c_eff, 0)

    return c_eff


def precompute_error_distance(backend: Any) -> np.ndarray:
    """Precompute error-weighted shortest path distances via Floyd-Warshall.

    Used for L_surr loss in Stage 2 training.

    Args:
        backend: A FakeBackendV2 instance.

    Returns:
        h×h numpy array of error-accumulated shortest path distances.
    """
    target = backend.target
    h = target.num_qubits
    gate_name = _get_two_qubit_gate_name(backend)
    cx_props = target[gate_name]

    adj = np.full((h, h), np.inf)
    np.fill_diagonal(adj, 0)

    for qargs, props in cx_props.items():
        if props is None:
            continue
        p, q = qargs
        error = props.error if props.error is not None else 0.0
        # Use minimum error for undirected
        adj[p][q] = min(adj[p][q], error)
        adj[q][p] = min(adj[q][p], error)

    d_error = floyd_warshall(adj)
    return d_error.astype(np.float32)


def precompute_hop_distance(backend: Any) -> np.ndarray:
    """Precompute unweighted shortest path (hop count) distances.

    Used for L_hop loss in Stage 2 training.

    Args:
        backend: A FakeBackendV2 instance.

    Returns:
        h×h numpy array of hop distances.
    """
    h = backend.target.num_qubits
    adj = np.full((h, h), np.inf)
    np.fill_diagonal(adj, 0)
    for edge in backend.coupling_map.get_edges():
        p, q = edge
        adj[p][q] = 1.0
        adj[q][p] = 1.0
    return floyd_warshall(adj).astype(np.float32)


def precompute_grama_W(backend: Any) -> np.ndarray:
    """Precompute the GraMA weighted-distance matrix W (Eq. 5 in Piao et al. 2026).

    The hardware coupling graph is weighted with -log(1 - eps_2q) per edge,
    so that all-pairs shortest paths produced by Floyd-Warshall correspond to
    the negative log of the multiplicative two-qubit success probability of
    routing between two physical qubits along the optimal path.

    For unconnected pairs, returns the path through the graph (FW infinity is
    propagated to inf, then later replaced with the max finite value to avoid
    NaN gradients in the loss).

    Args:
        backend: A FakeBackendV2 instance.

    Returns:
        h×h numpy float32 array. Diagonal is 0; off-diagonal entries are
        non-negative shortest -log(1-eps) sums.
    """
    target = backend.target
    h = target.num_qubits
    gate_name = _get_two_qubit_gate_name(backend)
    cx_props = target[gate_name]

    adj = np.full((h, h), np.inf)
    np.fill_diagonal(adj, 0)

    for qargs, props in cx_props.items():
        if props is None:
            continue
        p, q = qargs
        eps = props.error if props.error is not None else 0.0
        eps = float(np.clip(eps, 0.0, 0.999999))
        w = -np.log1p(-eps)  # = -log(1 - eps)
        # Use minimum weight for undirected
        adj[p][q] = min(adj[p][q], w)
        adj[q][p] = min(adj[q][p], w)

    W = floyd_warshall(adj).astype(np.float32)
    # Replace inf (disconnected) with max finite value to keep gradients finite.
    finite = W[np.isfinite(W)]
    if finite.size:
        max_w = float(finite.max())
        W[~np.isfinite(W)] = max_w
    else:
        W[~np.isfinite(W)] = 0.0
    return W


def precompute_grama_single_qubit_costs(backend: Any) -> dict[str, np.ndarray]:
    """Precompute raw per-qubit single-qubit cost vectors for GraMA Eq. (6).

    Per the paper:
        S = s_read · 1_k^T + s_gate · g^T

    where s_read[i] is the readout error of physical qubit i (raw, not z-scored),
    and s_gate[i] is the *highest* single-qubit gate error among all 1Q gates on
    physical qubit i.

    The g vector (per-logical 1Q gate count) is built per-circuit at load time.

    Args:
        backend: A FakeBackendV2 instance.

    Returns:
        Dict with keys 's_read' and 's_gate', each (h,) float32 numpy arrays.
    """
    target = backend.target
    num_qubits = target.num_qubits

    s_read = np.zeros(num_qubits, dtype=np.float32)
    s_gate = np.zeros(num_qubits, dtype=np.float32)

    if "measure" in target.operation_names:
        meas_props = target["measure"]
        for q in range(num_qubits):
            if (q,) in meas_props and meas_props[(q,)] is not None:
                s_read[q] = float(meas_props[(q,)].error or 0.0)

    for q in range(num_qubits):
        max_err = 0.0
        for gate_name in ("sx", "x", "rz", "id"):
            if gate_name in target.operation_names:
                gate_props = target[gate_name]
                if (q,) in gate_props and gate_props[(q,)] is not None:
                    err = gate_props[(q,)].error
                    if err is not None and err > max_err:
                        max_err = float(err)
        s_gate[q] = max_err

    return {"s_read": s_read, "s_gate": s_gate}


def get_hw_node_features(backend: Any) -> np.ndarray:
    """Get quality-score input features for a Qiskit FakeBackendV2.

    Returns (h, 6 or 8) array depending on configure_hw_features().
    Z-scored: [t1, t2,] readout_error, sq_error, degree, t1_cx_ratio, t2_cx_ratio
    Raw (appended): t2_t1_ratio

    Args:
        backend: A FakeBackendV2 instance.

    Returns:
        (h, 6 or 8) numpy array for QualityScore module input.
    """
    props = extract_qubit_properties(backend)
    zscore_list = []
    if _include_t1_t2:
        zscore_list.extend([props["t1"], props["t2"]])
    zscore_list.extend([
        props["readout_error"],
        props["single_qubit_error"],
    ])
    if not _exclude_degree:
        zscore_list.append(props["degree"])
    zscore_list.extend([
        props["t1_cx_ratio"],
        props["t2_cx_ratio"],
    ])
    zscore_feats = np.stack(zscore_list, axis=1).astype(np.float32)

    mean = zscore_feats.mean(axis=0, keepdims=True)
    std = zscore_feats.std(axis=0, keepdims=True) + 1e-8
    zscore_feats = (zscore_feats - mean) / std

    # Append raw T2/T1 ratio (dimensionless, not z-scored)
    raw_feats = props["t2_t1_ratio"].reshape(-1, 1).astype(np.float32)

    return np.concatenate([zscore_feats, raw_feats], axis=1).astype(np.float32)

