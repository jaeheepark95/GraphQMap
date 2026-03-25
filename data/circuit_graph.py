"""Circuit graph construction from quantum circuits.

Parses .qasm files or QuantumCircuit objects and builds PyG Data objects
with z-score normalized node and edge features.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from torch_geometric.data import Data

from data.normalization import zscore_normalize


def load_circuit(path: str | Path) -> QuantumCircuit:
    """Load a quantum circuit from a .qasm file.

    Args:
        path: Path to a .qasm file.

    Returns:
        Parsed QuantumCircuit.
    """
    return QuantumCircuit.from_qasm_file(str(path))


def extract_circuit_features(circuit: QuantumCircuit) -> dict[str, Any]:
    """Extract node and edge features from a quantum circuit.

    Node features (per logical qubit):
        - gate_count: total gates on this qubit
        - two_qubit_gate_count: 2-qubit gates involving this qubit
        - degree: number of distinct qubits interacting via 2-qubit gates
        - circuit_depth_participation: fraction of depth where qubit is active

    Edge features (per qubit pair with 2-qubit interaction):
        - interaction_count: number of 2-qubit gates between pair
        - earliest_interaction: normalized time (0~1) of first gate
        - latest_interaction: normalized time (0~1) of last gate

    Args:
        circuit: A QuantumCircuit instance.

    Returns:
        Dict with keys: node_features, edge_list, edge_features,
        num_qubits.
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

    # Node features
    degree = [len(n) for n in neighbors]
    depth_participation = [len(al) / total_depth for al in active_layers]

    node_features = torch.tensor(
        [[gate_count[i], two_qubit_gate_count[i], degree[i], depth_participation[i]]
         for i in range(num_qubits)],
        dtype=torch.float32,
    )

    # Edge features
    edge_list = sorted(pair_interactions.keys())
    edge_features_list = []
    for pair in edge_list:
        interactions = pair_interactions[pair]
        count = len(interactions)
        earliest = min(interactions) / total_depth
        latest = max(interactions) / total_depth
        edge_features_list.append([count, earliest, latest])

    edge_features = (
        torch.tensor(edge_features_list, dtype=torch.float32)
        if edge_features_list
        else torch.zeros((0, 3), dtype=torch.float32)
    )

    return {
        "node_features": node_features,
        "edge_list": edge_list,
        "edge_features": edge_features,
        "num_qubits": num_qubits,
    }


def build_circuit_graph(
    circuit: QuantumCircuit,
    eps: float = 1e-8,
) -> Data:
    """Build a PyG Data object for a quantum circuit.

    Node features (4-dim) are z-score normalized within the circuit.
    Same dimensions for both single-circuit and multi-programming.

    Args:
        circuit: A QuantumCircuit instance.
        eps: Epsilon for z-score normalization.

    Returns:
        PyG Data with x, edge_index, edge_attr, num_qubits.
    """
    feats = extract_circuit_features(circuit)
    x = feats["node_features"]  # (num_qubits, 4)

    # Z-score normalize node features within circuit (across qubits, dim=0)
    x = zscore_normalize(x, dim=0, eps=eps)

    # Build edge_index: undirected -> both directions
    edge_list = feats["edge_list"]
    if len(edge_list) > 0:
        src = [e[0] for e in edge_list] + [e[1] for e in edge_list]
        dst = [e[1] for e in edge_list] + [e[0] for e in edge_list]
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        edge_attr = feats["edge_features"]
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)  # duplicate for both dirs
        # Z-score normalize edge features within circuit (across edges, dim=0)
        edge_attr = zscore_normalize(edge_attr, dim=0, eps=eps)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 3), dtype=torch.float32)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_qubits=feats["num_qubits"],
    )
