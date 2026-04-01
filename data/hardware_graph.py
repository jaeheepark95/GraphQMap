"""Hardware graph construction from FakeBackendV2 and synthetic backends.

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

SYNTHETIC_BACKENDS_DIR = Path(__file__).parent / "circuits" / "backends"

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


def _load_synthetic_backend(name: str) -> dict:
    """Load a synthetic backend definition from JSON.

    Args:
        name: Synthetic backend name (e.g. 'queko_aspen4', 'mlqd_grid5x5').

    Returns:
        Dict with backend_name, num_qubits, coupling_map,
        qubit_properties, edge_properties.
    """
    path = SYNTHETIC_BACKENDS_DIR / f"{name}.json"
    if not path.exists():
        raise ValueError(f"Synthetic backend file not found: {path}")
    with open(path) as f:
        return json.load(f)


def is_synthetic_backend(name: str) -> bool:
    """Check if a backend name refers to a synthetic backend."""
    return (SYNTHETIC_BACKENDS_DIR / f"{name}.json").exists()


def get_backend(name: str) -> Any:
    """Instantiate a FakeBackendV2 by short name.

    For Qiskit FakeBackendV2 backends only. For synthetic backends,
    use build_hardware_graph_from_synthetic() instead.

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

    return {
        "t1": t1,
        "t2": t2,
        "readout_error": readout_error,
        "single_qubit_error": single_qubit_error,
        "degree": degree,
        "t1_cx_ratio": t1_cx_ratio,
        "t2_cx_ratio": t2_cx_ratio,
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


def extract_edge_properties(backend: Any) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Extract per-edge 2-qubit gate properties from a backend.

    Supports backends with cx, ecr, or cz as the native 2-qubit gate.

    Args:
        backend: A FakeBackendV2 instance.

    Returns:
        Tuple of (edge_list, edge_features).
        edge_list: List of (src, dst) tuples (undirected, each pair once).
        edge_features: Array of shape (num_edges, 1) with [2q_error].
    """
    target = backend.target
    gate_name = _get_two_qubit_gate_name(backend)
    cx_props = target[gate_name]

    # Collect undirected edges (deduplicate by sorting)
    edge_dict: dict[tuple[int, int], list[float]] = {}
    for qargs, props in cx_props.items():
        if props is None:
            continue
        a, b = qargs
        key = (min(a, b), max(a, b))
        cx_error = props.error if props.error is not None else 0.0
        if key not in edge_dict:
            edge_dict[key] = [cx_error]
        else:
            # Average the two directions
            edge_dict[key][0] = (edge_dict[key][0] + cx_error) / 2

    edge_list = sorted(edge_dict.keys())
    edge_features = np.array([edge_dict[e] for e in edge_list], dtype=np.float32)
    if edge_features.ndim == 1:
        edge_features = edge_features.reshape(-1, 1)

    return edge_list, edge_features


def build_hardware_graph(backend: Any, eps: float = 1e-8) -> Data:
    """Build a PyG Data object for a hardware backend.

    Node features (5): readout_error, single_qubit_error, degree,
                       t1_cx_ratio, t2_cx_ratio
    Edge features (1): cx_error
    All features z-score normalized within the backend.

    Args:
        backend: A FakeBackendV2 instance.
        eps: Epsilon for z-score normalization.

    Returns:
        PyG Data object with x, edge_index, edge_attr, and num_qubits.
    """
    qubit_props = extract_qubit_properties(backend)
    edge_list, edge_feats = extract_edge_properties(backend)

    # Stack node features: (num_qubits, 5)
    node_features = np.stack([
        qubit_props["readout_error"],
        qubit_props["single_qubit_error"],
        qubit_props["degree"],
        qubit_props["t1_cx_ratio"],
        qubit_props["t2_cx_ratio"],
    ], axis=1).astype(np.float32)

    x = torch.from_numpy(node_features)
    # Z-score normalize within backend (along dim=0, across qubits)
    x = zscore_normalize(x, dim=0, eps=eps)

    # Build edge_index: undirected -> add both directions
    if len(edge_list) > 0:
        src = [e[0] for e in edge_list] + [e[1] for e in edge_list]
        dst = [e[1] for e in edge_list] + [e[0] for e in edge_list]
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        # Edge features: duplicate for both directions
        edge_attr = torch.from_numpy(edge_feats)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        # Z-score normalize edge features within backend
        edge_attr = zscore_normalize(edge_attr, dim=0, eps=eps)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float32)

    num_qubits = backend.target.num_qubits

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_qubits=num_qubits,
    )


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


def get_hw_node_features(backend: Any) -> np.ndarray:
    """Get quality-score input features for a Qiskit FakeBackendV2.

    Returns (h, 5) array: [readout_error, single_qubit_error, degree,
                           t1_cx_ratio, t2_cx_ratio].
    Values are z-score normalized within the backend.

    Args:
        backend: A FakeBackendV2 instance.

    Returns:
        (h, 5) numpy array for QualityScore module input.
    """
    props = extract_qubit_properties(backend)
    raw = np.stack([
        props["readout_error"],
        props["single_qubit_error"],
        props["degree"],
        props["t1_cx_ratio"],
        props["t2_cx_ratio"],
    ], axis=1).astype(np.float32)

    mean = raw.mean(axis=0, keepdims=True)
    std = raw.std(axis=0, keepdims=True) + 1e-8
    return ((raw - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Synthetic backend support
# ---------------------------------------------------------------------------


def build_hardware_graph_from_synthetic(name: str, eps: float = 1e-8) -> Data:
    """Build a PyG Data object from a synthetic backend JSON file.

    Same output format as build_hardware_graph() for FakeBackendV2.

    Args:
        name: Synthetic backend name (e.g. 'queko_aspen4').
        eps: Epsilon for z-score normalization.

    Returns:
        PyG Data object with x, edge_index, edge_attr, and num_qubits.
    """
    profile = _load_synthetic_backend(name)
    n = profile["num_qubits"]
    coupling_map = [tuple(e) for e in profile["coupling_map"]]
    qprops = profile["qubit_properties"]
    eprops = profile["edge_properties"]

    # Node features: (n, 5) — readout_error, sq_gate_error, degree,
    #                           t1_cx_ratio, t2_cx_ratio
    degree = np.zeros(n)
    neighbor_map: list[set[int]] = [set() for _ in range(n)]
    for p, q in coupling_map:
        neighbor_map[p].add(q)
        neighbor_map[q].add(p)
    for i in range(n):
        degree[i] = len(neighbor_map[i])

    # Per-qubit mean cx_duration for ratio features
    cx_duration_sum = np.zeros(n)
    cx_duration_count = np.zeros(n)
    for p, q in coupling_map:
        key = f"({p}, {q})"
        ep = eprops.get(key) or eprops.get(f"({q}, {p})")
        if ep and ep.get("cx_duration"):
            cx_duration_sum[p] += ep["cx_duration"]
            cx_duration_sum[q] += ep["cx_duration"]
            cx_duration_count[p] += 1
            cx_duration_count[q] += 1
    mean_cx_duration = np.where(cx_duration_count > 0, cx_duration_sum / cx_duration_count, 1.0)

    t1_arr = np.array([qprops[str(i)]["t1"] for i in range(n)], dtype=np.float32)
    t2_arr = np.array([qprops[str(i)]["t2"] for i in range(n)], dtype=np.float32)
    t1_cx_ratio = np.where(mean_cx_duration > 0, t1_arr / mean_cx_duration, 0.0)
    t2_cx_ratio = np.where(mean_cx_duration > 0, t2_arr / mean_cx_duration, 0.0)

    node_features = np.zeros((n, 5), dtype=np.float32)
    for i in range(n):
        qp = qprops[str(i)]
        node_features[i] = [
            qp["readout_error"], qp["sq_gate_error"], degree[i],
            t1_cx_ratio[i], t2_cx_ratio[i],
        ]

    x = torch.from_numpy(node_features)
    x = zscore_normalize(x, dim=0, eps=eps)

    # Edge features: (num_edges*2, 1) — cx_error only
    edge_list = coupling_map
    if len(edge_list) > 0:
        src = [e[0] for e in edge_list] + [e[1] for e in edge_list]
        dst = [e[1] for e in edge_list] + [e[0] for e in edge_list]
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        edge_feats = []
        for p, q in edge_list:
            key = f"({p}, {q})"
            ep = eprops[key]
            edge_feats.append([ep["cx_error"]])
        edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        edge_attr = zscore_normalize(edge_attr, dim=0, eps=eps)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_qubits=n)


def precompute_error_distance_synthetic(name: str) -> np.ndarray:
    """Precompute error-weighted shortest path distances for a synthetic backend.

    Args:
        name: Synthetic backend name (e.g. 'queko_aspen4').

    Returns:
        h×h numpy array of error-accumulated shortest path distances.
    """
    profile = _load_synthetic_backend(name)
    h = profile["num_qubits"]
    eprops = profile["edge_properties"]
    coupling_map = [tuple(e) for e in profile["coupling_map"]]

    adj = np.full((h, h), np.inf)
    np.fill_diagonal(adj, 0)

    for p, q in coupling_map:
        key = f"({p}, {q})"
        error = eprops[key]["cx_error"]
        adj[p][q] = min(adj[p][q], error)
        adj[q][p] = min(adj[q][p], error)

    return floyd_warshall(adj).astype(np.float32)


def precompute_hop_distance_synthetic(name: str) -> np.ndarray:
    """Precompute hop distances for a synthetic backend.

    Args:
        name: Synthetic backend name.

    Returns:
        h×h numpy array of hop distances.
    """
    profile = _load_synthetic_backend(name)
    h = profile["num_qubits"]
    coupling_map = [tuple(e) for e in profile["coupling_map"]]
    adj = np.full((h, h), np.inf)
    np.fill_diagonal(adj, 0)
    for p, q in coupling_map:
        adj[p][q] = 1.0
        adj[q][p] = 1.0
    return floyd_warshall(adj).astype(np.float32)


def get_hw_node_features_synthetic(name: str) -> np.ndarray:
    """Get quality-score input features for a synthetic backend.

    Returns (h, 5) array: [readout_error, single_qubit_error, degree,
                           t1_cx_ratio, t2_cx_ratio].
    Values are z-score normalized within the backend.

    Args:
        name: Synthetic backend name.

    Returns:
        (h, 5) numpy array for QualityScore module input.
    """
    profile = _load_synthetic_backend(name)
    n = profile["num_qubits"]
    qprops = profile["qubit_properties"]
    coupling_map = [tuple(e) for e in profile["coupling_map"]]
    eprops = profile["edge_properties"]

    # Degree from coupling map
    degree = np.zeros(n)
    neighbor_map: list[set[int]] = [set() for _ in range(n)]
    for p, q in coupling_map:
        neighbor_map[p].add(q)
        neighbor_map[q].add(p)
    for i in range(n):
        degree[i] = len(neighbor_map[i])

    # Per-qubit mean cx_duration for ratio features
    cx_duration_sum = np.zeros(n)
    cx_duration_count = np.zeros(n)
    for p, q in coupling_map:
        key = f"({p}, {q})"
        ep = eprops.get(key) or eprops.get(f"({q}, {p})")
        if ep and ep.get("cx_duration"):
            cx_duration_sum[p] += ep["cx_duration"]
            cx_duration_sum[q] += ep["cx_duration"]
            cx_duration_count[p] += 1
            cx_duration_count[q] += 1
    mean_cx_duration = np.where(cx_duration_count > 0, cx_duration_sum / cx_duration_count, 1.0)

    t1_arr = np.array([qprops[str(i)]["t1"] for i in range(n)], dtype=np.float32)
    t2_arr = np.array([qprops[str(i)]["t2"] for i in range(n)], dtype=np.float32)
    t1_cx_ratio = np.where(mean_cx_duration > 0, t1_arr / mean_cx_duration, 0.0)
    t2_cx_ratio = np.where(mean_cx_duration > 0, t2_arr / mean_cx_duration, 0.0)

    raw = np.zeros((n, 5), dtype=np.float32)
    for i in range(n):
        qp = qprops[str(i)]
        raw[i] = [
            qp["readout_error"], qp["sq_gate_error"], degree[i],
            t1_cx_ratio[i], t2_cx_ratio[i],
        ]

    # Z-score normalize within backend
    mean = raw.mean(axis=0, keepdims=True)
    std = raw.std(axis=0, keepdims=True) + 1e-8
    return ((raw - mean) / std).astype(np.float32)
