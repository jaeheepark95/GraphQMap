"""Multi-programming circuit graph merging for GraphQMap.

Merges multiple circuit graphs into a single disconnected PyG graph
with global summary features for multi-programming scenarios.
"""

from __future__ import annotations

import torch
from qiskit import QuantumCircuit
from torch_geometric.data import Batch, Data

from data.circuit_graph import build_circuit_graph, extract_circuit_features
from data.normalization import zscore_normalize


def merge_circuits(
    circuits: list[QuantumCircuit],
    summary_stats: dict[str, tuple[float, float]] | None = None,
    eps: float = 1e-8,
) -> Data:
    """Merge multiple circuits into a single disconnected PyG graph.

    Each node receives its circuit's global summary features (z-score normalized
    across the training dataset if summary_stats are provided).

    Args:
        circuits: List of QuantumCircuit objects to merge.
        summary_stats: Optional dict mapping summary feature names to (mean, std)
                       tuples for dataset-level z-score normalization.
                       Keys: 'total_qubits', 'total_2q_gates', 'total_depth', 'gate_density'.
                       If None, raw summary values are used (no cross-dataset normalization).
        eps: Epsilon for normalization.

    Returns:
        Merged PyG Data object with:
        - x: (total_qubits, 8) node features (4 local + 4 global summary)
        - edge_index, edge_attr: merged edges with adjusted indices
        - num_qubits: total logical qubit count
        - circuit_sizes: list of per-circuit qubit counts
        - circuit_ids: (total_qubits,) tensor mapping each node to its circuit index
    """
    graph_list: list[Data] = []
    circuit_sizes: list[int] = []

    for circuit in circuits:
        feats = extract_circuit_features(circuit)
        global_summary = feats["global_summary"]  # (4,)

        # Normalize global summary across dataset if stats provided
        if summary_stats is not None:
            names = ["total_qubits", "total_2q_gates", "total_depth", "gate_density"]
            normalized = []
            for i, name in enumerate(names):
                mean, std = summary_stats[name]
                normalized.append((global_summary[i] - mean) / (std + eps))
            global_summary = torch.tensor(normalized, dtype=torch.float32)

        graph = build_circuit_graph(circuit, eps=eps, global_summary=global_summary)
        graph_list.append(graph)
        circuit_sizes.append(circuit.num_qubits)

    # Merge into a single disconnected graph using PyG Batch
    merged = Batch.from_data_list(graph_list)

    # Build circuit_ids tensor: which circuit each node belongs to
    circuit_ids = torch.cat([
        torch.full((size,), i, dtype=torch.long)
        for i, size in enumerate(circuit_sizes)
    ])

    # Convert Batch back to plain Data (remove batch-specific attributes)
    data = Data(
        x=merged.x,
        edge_index=merged.edge_index,
        edge_attr=merged.edge_attr,
        num_qubits=sum(circuit_sizes),
        circuit_sizes=circuit_sizes,
        circuit_ids=circuit_ids,
    )

    return data


def validate_multi_programming(
    circuits: list[QuantumCircuit],
    num_physical_qubits: int,
    occupancy_max: float = 0.75,
) -> bool:
    """Check if circuits can be combined on the target hardware.

    Args:
        circuits: List of circuits to combine.
        num_physical_qubits: Number of physical qubits on target backend.
        occupancy_max: Maximum occupancy ratio.

    Returns:
        True if the combination is valid.
    """
    total_logical = sum(c.num_qubits for c in circuits)
    return total_logical < num_physical_qubits and (
        total_logical / num_physical_qubits <= occupancy_max
    )
