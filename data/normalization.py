"""Z-score normalization utilities for GraphQMap."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

EPSILON = 1e-8


def zscore_normalize(tensor: torch.Tensor, dim: int = 0, eps: float = EPSILON) -> torch.Tensor:
    """Z-score normalize along the given dimension.

    Args:
        tensor: Input tensor to normalize.
        dim: Dimension along which to compute mean/std.
        eps: Small value added to std to avoid division by zero.

    Returns:
        Normalized tensor with zero mean and unit variance along dim.
    """
    mean = tensor.mean(dim=dim, keepdim=True)
    std = tensor.std(dim=dim, keepdim=True, correction=0)
    return (tensor - mean) / (std + eps)


def renormalize_group_edges(graphs: list[Data], eps: float = EPSILON) -> list[Data]:
    """Re-normalize edge features across a group of circuit graphs.

    For multi-programming, multiple circuits form a single disconnected graph.
    Edge features should be z-score normalized at the group level (across all
    edges in all circuits) rather than per-circuit.

    Node features remain per-circuit normalized (each circuit's nodes have
    independent feature scales).

    Args:
        graphs: List of PyG Data objects (circuit graphs with per-circuit
                normalized edge features).
        eps: Epsilon for z-score normalization.

    Returns:
        List of PyG Data objects with group-level normalized edge features.
        Original graphs are not modified; new Data objects are returned.
    """
    if len(graphs) <= 1:
        return graphs

    # Collect all edge features
    all_edges = [g.edge_attr for g in graphs if g.edge_attr.size(0) > 0]
    if not all_edges:
        return graphs

    combined = torch.cat(all_edges, dim=0)
    mean = combined.mean(dim=0, keepdim=True)
    std = combined.std(dim=0, keepdim=True, correction=0)

    result = []
    for g in graphs:
        if g.edge_attr.size(0) > 0:
            new_edge_attr = (g.edge_attr - mean) / (std + eps)
            result.append(Data(
                x=g.x, edge_index=g.edge_index, edge_attr=new_edge_attr,
                num_qubits=g.num_qubits,
            ))
        else:
            result.append(g)
    return result
