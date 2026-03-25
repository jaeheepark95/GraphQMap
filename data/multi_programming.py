"""Multi-programming circuit graph merging for GraphQMap.

Merges multiple circuit graphs into a single disconnected PyG graph.
Uses the same 4-dim node features as single-circuit (no global summary).
The GNN naturally distinguishes circuits through disconnected components.
"""

from __future__ import annotations

import torch
from qiskit import QuantumCircuit
from torch_geometric.data import Batch, Data

from data.circuit_graph import build_circuit_graph
from data.normalization import renormalize_group_edges


def merge_circuits(
    circuits: list[QuantumCircuit],
    eps: float = 1e-8,
) -> Data:
    """Merge multiple circuits into a single disconnected PyG graph.

    Node features remain 4-dimensional (same as single-circuit), keeping
    the model architecture unified for both single and multi-programming.

    Edge features are re-normalized at the group level (across all circuits)
    when multiple circuits are merged. Node features remain per-circuit normalized.

    Args:
        circuits: List of QuantumCircuit objects to merge.
        eps: Epsilon for normalization.

    Returns:
        Merged PyG Data object with:
        - x: (total_qubits, 4) node features (per-circuit normalized)
        - edge_index, edge_attr: merged edges with group-level normalized features
        - num_qubits: total logical qubit count
        - circuit_sizes: list of per-circuit qubit counts
        - circuit_ids: (total_qubits,) tensor mapping each node to its circuit index
    """
    graph_list: list[Data] = []
    circuit_sizes: list[int] = []

    for circuit in circuits:
        graph = build_circuit_graph(circuit, eps=eps)
        graph_list.append(graph)
        circuit_sizes.append(circuit.num_qubits)

    # Re-normalize edge features at group level for multi-programming
    if len(graph_list) > 1:
        graph_list = renormalize_group_edges(graph_list, eps=eps)

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
    return total_logical <= num_physical_qubits and (
        total_logical / num_physical_qubits <= occupancy_max
    )
