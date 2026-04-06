"""Circuit graph construction from quantum circuits.

Parses .qasm files or QuantumCircuit objects and builds PyG Data objects
with z-score normalized node and edge features.

Node features are configurable via a feature registry. Available features:
  - gate_count: total gates on this qubit
  - two_qubit_gate_count: 2-qubit gates involving this qubit
  - degree: number of distinct qubits interacting via 2-qubit gates
  - depth_participation: fraction of circuit depth where qubit is active
  - weighted_degree: sum of interaction counts across all connected edges
  - single_qubit_gate_ratio: fraction of gates that are single-qubit
  - critical_path_fraction: fraction of DAG critical path involving this qubit
  - interaction_entropy: Shannon entropy of per-neighbor interaction distribution

Edge features (per qubit pair with 2-qubit interaction):
  - interaction_count: number of 2-qubit gates between pair
  - earliest_interaction: normalized time of first gate
  - latest_interaction: normalized time of last gate
  - interaction_span: latest - earliest (temporal duration)
  - interaction_density: count / (span + eps) (burstiness)

Positional encodings (appended after node features):
  - rwpe: Random Walk Positional Encoding (k-step self-return probabilities)
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from torch_geometric.data import Data

from data.normalization import zscore_normalize

# Default feature set (backward compatible)
DEFAULT_NODE_FEATURES: list[str] = [
    "gate_count",
    "two_qubit_gate_count",
    "single_qubit_gate_ratio",
    "critical_path_fraction",
]


def load_circuit(path: str | Path) -> QuantumCircuit:
    """Load a quantum circuit from a .qasm file.

    Args:
        path: Path to a .qasm file.

    Returns:
        Parsed QuantumCircuit.
    """
    return QuantumCircuit.from_qasm_file(str(path))


def extract_circuit_features(circuit: QuantumCircuit) -> dict[str, Any]:
    """Extract ALL available node and edge features from a quantum circuit.

    Computes every registered node feature so that feature selection
    can happen downstream without re-parsing the circuit.

    Node features (per logical qubit):
        - gate_count: total gates on this qubit
        - two_qubit_gate_count: 2-qubit gates involving this qubit
        - degree: number of distinct qubits interacting via 2-qubit gates
        - depth_participation: fraction of depth where qubit is active
        - weighted_degree: sum of interaction counts with all neighbors
        - single_qubit_gate_ratio: (gate_count - 2q_count) / gate_count
        - critical_path_fraction: fraction of critical path length on this qubit
        - interaction_entropy: -Σ p_ij log p_ij over neighbor interaction distribution

    Edge features (per qubit pair with 2-qubit interaction):
        - interaction_count: number of 2-qubit gates between pair
        - earliest_interaction: normalized time (0~1) of first gate
        - latest_interaction: normalized time (0~1) of last gate
        - interaction_span: latest - earliest (temporal duration of interaction)
        - interaction_density: count / (span + eps) (burstiness of interaction)

    Args:
        circuit: A QuantumCircuit instance.

    Returns:
        Dict with keys: node_features (dict[str, list[float]]),
        edge_list, edge_features, num_qubits.
    """
    dag = circuit_to_dag(circuit)
    num_qubits = circuit.num_qubits

    # Per-qubit gate counts
    gate_count = [0] * num_qubits
    two_qubit_gate_count = [0] * num_qubits
    neighbors: list[set[int]] = [set() for _ in range(num_qubits)]

    # Depth participation: track which layers each qubit is active in
    layers = list(dag.layers())
    total_depth = max(len(layers), 1)
    active_layers: list[set[int]] = [set() for _ in range(num_qubits)]

    # Edge interaction tracking
    pair_interactions: dict[tuple[int, int], list[int]] = defaultdict(list)

    for layer_idx, layer in enumerate(layers):
        for op in layer["graph"].op_nodes():
            qubit_indices = [circuit.qubits.index(q) for q in op.qargs]

            for qi in qubit_indices:
                gate_count[qi] += 1
                active_layers[qi].add(layer_idx)

            if len(qubit_indices) == 2:
                q0, q1 = qubit_indices
                two_qubit_gate_count[q0] += 1
                two_qubit_gate_count[q1] += 1
                neighbors[q0].add(q1)
                neighbors[q1].add(q0)

                key = (min(q0, q1), max(q0, q1))
                pair_interactions[key].append(layer_idx)

    # --- Node features ---
    degree = [len(n) for n in neighbors]
    depth_participation = [len(al) / total_depth for al in active_layers]

    # Weighted degree: sum of interaction counts per qubit
    weighted_degree = [0.0] * num_qubits
    for (q0, q1), interactions in pair_interactions.items():
        count = len(interactions)
        weighted_degree[q0] += count
        weighted_degree[q1] += count

    # Single-qubit gate ratio
    single_qubit_gate_ratio = [
        (gate_count[i] - two_qubit_gate_count[i]) / max(gate_count[i], 1)
        for i in range(num_qubits)
    ]

    # Critical path fraction: per-qubit participation in DAG critical path
    # Skip for very large circuits (>5000 ops) to avoid O(n^2) traversal
    op_count = dag.size()
    if op_count <= 5000:
        critical_path_fraction = _compute_critical_path_fraction(dag, circuit, num_qubits)
    else:
        critical_path_fraction = [1.0 / max(num_qubits, 1)] * num_qubits

    # Interaction entropy: H_i = -Σ p_ij log p_ij over neighbors
    # Measures whether interactions are concentrated on few neighbors or spread evenly
    interaction_entropy = [0.0] * num_qubits
    for qi in range(num_qubits):
        # Collect interaction counts with each neighbor
        neighbor_counts: list[int] = []
        for nj in neighbors[qi]:
            key = (min(qi, nj), max(qi, nj))
            neighbor_counts.append(len(pair_interactions[key]))
        total = sum(neighbor_counts)
        if total > 0 and len(neighbor_counts) > 1:
            probs = [c / total for c in neighbor_counts]
            interaction_entropy[qi] = -sum(
                p * math.log(p) for p in probs if p > 0
            )

    # Store all node features as a dict of lists
    node_features_dict: dict[str, list[float]] = {
        "gate_count": [float(x) for x in gate_count],
        "two_qubit_gate_count": [float(x) for x in two_qubit_gate_count],
        "degree": [float(x) for x in degree],
        "depth_participation": depth_participation,
        "weighted_degree": weighted_degree,
        "single_qubit_gate_ratio": single_qubit_gate_ratio,
        "critical_path_fraction": critical_path_fraction,
        "interaction_entropy": interaction_entropy,
    }

    # Edge features
    edge_list = sorted(pair_interactions.keys())
    edge_features_list = []
    for pair in edge_list:
        interactions = pair_interactions[pair]
        count = len(interactions)
        earliest = min(interactions) / total_depth
        latest = max(interactions) / total_depth
        span = latest - earliest
        density = count / (span + 1e-8) if span > 1e-10 else float(count)
        edge_features_list.append([count, earliest, latest, span, density])

    edge_features = (
        torch.tensor(edge_features_list, dtype=torch.float32)
        if edge_features_list
        else torch.zeros((0, 5), dtype=torch.float32)
    )

    return {
        "node_features_dict": node_features_dict,
        "edge_list": edge_list,
        "edge_features": edge_features,
        "num_qubits": num_qubits,
    }


def _compute_critical_path_fraction(
    dag: Any, circuit: QuantumCircuit, num_qubits: int,
) -> list[float]:
    """Compute per-qubit participation in the DAG critical path.

    Uses a per-qubit depth tracking approach: for each qubit, tracks the
    cumulative depth of operations. The critical path length is the maximum
    depth across all qubits. Each qubit's fraction is its max depth / critical path.

    This is O(n) in the number of gates, avoiding expensive DAG traversal.

    Args:
        dag: DAGCircuit from circuit_to_dag.
        circuit: Original QuantumCircuit.
        num_qubits: Number of logical qubits.

    Returns:
        List of floats (one per qubit), each in [0, 1].
    """
    # Per-qubit depth: track how deep each qubit is in the circuit
    qubit_depth = [0] * num_qubits

    for node in dag.topological_op_nodes():
        qubit_indices = [circuit.qubits.index(q) for q in node.qargs]
        # This op starts after the latest qubit it touches is free
        start = max(qubit_depth[qi] for qi in qubit_indices)
        new_depth = start + 1
        for qi in qubit_indices:
            qubit_depth[qi] = new_depth

    critical_path_length = max(qubit_depth) if qubit_depth else 1

    return [d / max(critical_path_length, 1) for d in qubit_depth]


def compute_rwpe(
    edge_list: list[tuple[int, int]],
    num_nodes: int,
    k: int,
    start_step: int = 2,
) -> torch.Tensor:
    """Compute Random Walk Positional Encoding.

    RWPE_i = [RW^s_ii, RW^(s+1)_ii, ..., RW^(s+k-1)_ii]
    where RW is the random walk transition matrix (row-normalized adjacency)
    and s = start_step.

    Step 1 (1-step self-return) is structurally zero for graphs without
    self-loops, so start_step defaults to 2 to skip dead dimensions.

    Args:
        edge_list: List of undirected edges (i, j).
        num_nodes: Number of nodes in the graph.
        k: Number of RWPE dimensions to output.
        start_step: First random walk step to include (default 2).

    Returns:
        Tensor of shape (num_nodes, k) with self-return probabilities.
    """
    if num_nodes == 0 or k == 0:
        return torch.zeros((num_nodes, k), dtype=torch.float32)

    # Build adjacency matrix
    A = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    for i, j in edge_list:
        A[i, j] = 1.0
        A[j, i] = 1.0

    # Row-normalize to get transition matrix
    D = A.sum(dim=1, keepdim=True).clamp(min=1.0)
    M = A / D

    # Advance to start_step
    Mk = M.clone()
    for _ in range(start_step - 1):
        Mk = Mk @ M

    # Collect k dimensions starting from start_step
    rwpe = torch.zeros(num_nodes, k, dtype=torch.float32)
    for step in range(k):
        rwpe[:, step] = Mk.diagonal()
        if step < k - 1:
            Mk = Mk @ M

    return rwpe


def build_circuit_graph(
    circuit: QuantumCircuit,
    node_feature_names: list[str] | None = None,
    rwpe_k: int = 0,
    edge_dim: int | None = None,
    eps: float = 1e-8,
) -> Data:
    """Build a PyG Data object for a quantum circuit.

    Node features are selected by name from the feature registry,
    z-score normalized within the circuit, then optionally concatenated
    with RWPE (not z-score normalized).

    Args:
        circuit: A QuantumCircuit instance.
        node_feature_names: List of feature names to include.
            Defaults to DEFAULT_NODE_FEATURES if None.
        rwpe_k: Number of RWPE steps (0 = disabled).
        edge_dim: Number of edge feature dimensions (None = use all 5).
        eps: Epsilon for z-score normalization.

    Returns:
        PyG Data with x, edge_index, edge_attr, num_qubits.
    """
    if node_feature_names is None:
        node_feature_names = DEFAULT_NODE_FEATURES

    feats = extract_circuit_features(circuit)
    return build_circuit_graph_from_raw(
        node_features_dict=feats["node_features_dict"],
        edge_list=feats["edge_list"],
        edge_features=feats["edge_features"],
        num_qubits=feats["num_qubits"],
        node_feature_names=node_feature_names,
        rwpe_k=rwpe_k,
        edge_dim=edge_dim,
        eps=eps,
    )


def _extend_edge_features(edge_features: torch.Tensor) -> torch.Tensor:
    """Extend 3-dim edge features to 5-dim by computing span and density.

    Backward compatibility for old cache format with only
    (interaction_count, earliest_interaction, latest_interaction).

    Args:
        edge_features: (num_edges, 3) tensor.

    Returns:
        (num_edges, 5) tensor with span and density appended.
    """
    if edge_features.shape[0] == 0:
        return torch.zeros((0, 5), dtype=torch.float32)
    count = edge_features[:, 0]
    earliest = edge_features[:, 1]
    latest = edge_features[:, 2]
    span = latest - earliest
    density = torch.where(span > 1e-10, count / (span + 1e-8), count)
    return torch.cat([edge_features, span.unsqueeze(1), density.unsqueeze(1)], dim=1)


def build_circuit_graph_from_raw(
    node_features_dict: dict[str, list[float]],
    edge_list: list[tuple[int, int]],
    edge_features: torch.Tensor,
    num_qubits: int,
    node_feature_names: list[str] | None = None,
    rwpe_k: int = 0,
    edge_dim: int | None = None,
    eps: float = 1e-8,
) -> Data:
    """Build a PyG Data object from pre-extracted raw features.

    This enables feature selection at load time without re-parsing QASM.

    Args:
        node_features_dict: Dict mapping feature name -> list of values per qubit.
        edge_list: List of (src, dst) edge pairs.
        edge_features: Raw edge feature tensor (num_edges, 3 or 5).
        num_qubits: Number of logical qubits.
        node_feature_names: Which node features to include.
        rwpe_k: Number of RWPE steps (0 = disabled).
        edge_dim: Number of edge feature dimensions to use (None = use all).
            If edge_features has fewer dims, extends automatically.
        eps: Epsilon for z-score normalization.

    Returns:
        PyG Data with x, edge_index, edge_attr, num_qubits.
    """
    if node_feature_names is None:
        node_feature_names = DEFAULT_NODE_FEATURES

    # Validate feature names
    available = set(node_features_dict.keys())
    for name in node_feature_names:
        if name not in available:
            raise ValueError(
                f"Unknown circuit node feature '{name}'. "
                f"Available: {sorted(available)}"
            )

    # Select and stack node features
    selected = [node_features_dict[name] for name in node_feature_names]
    x = torch.tensor(
        list(zip(*selected)),
        dtype=torch.float32,
    )  # (num_qubits, num_features)

    # Z-score normalize node features within circuit
    x = zscore_normalize(x, dim=0, eps=eps)

    # Append RWPE if requested (not z-score normalized)
    if rwpe_k > 0:
        rwpe = compute_rwpe(edge_list, num_qubits, rwpe_k)
        x = torch.cat([x, rwpe], dim=1)

    # Prepare edge features: extend if needed, then slice to edge_dim
    edge_features = edge_features.clone() if isinstance(edge_features, torch.Tensor) else torch.tensor(edge_features, dtype=torch.float32)
    if edge_features.ndim == 2 and edge_features.shape[1] < 5 and edge_features.shape[0] > 0:
        edge_features = _extend_edge_features(edge_features)
    if edge_dim is not None and edge_features.ndim == 2 and edge_features.shape[1] > edge_dim:
        edge_features = edge_features[:, :edge_dim]

    # Build edge_index: undirected -> both directions
    if len(edge_list) > 0:
        src = [e[0] for e in edge_list] + [e[1] for e in edge_list]
        dst = [e[1] for e in edge_list] + [e[0] for e in edge_list]
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        edge_attr = torch.cat([edge_features, edge_features], dim=0)  # duplicate for both dirs
        edge_attr = zscore_normalize(edge_attr, dim=0, eps=eps)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        empty_dim = edge_dim if edge_dim is not None else (edge_features.shape[1] if edge_features.ndim == 2 else 5)
        edge_attr = torch.zeros((0, empty_dim), dtype=torch.float32)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_qubits=num_qubits,
    )
