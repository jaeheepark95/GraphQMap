"""Hungarian algorithm decoder for inference.

Converts doubly stochastic matrix P to a discrete one-to-one mapping.
Only the top l rows (actual logical qubits) are used.
"""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment


def hungarian_decode(P: torch.Tensor, num_logical: int) -> dict[int, int]:
    """Decode P matrix to a discrete layout via Hungarian algorithm.

    Args:
        P: (h, h) doubly stochastic matrix (single sample, no batch dim).
        num_logical: Number of actual logical qubits (l).

    Returns:
        Layout dict {logical_qubit: physical_qubit}.
    """
    cost = 1.0 - P.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)

    layout = {}
    for i in range(num_logical):
        layout[i] = int(col_ind[i])
    return layout


def hungarian_decode_batch(
    P: torch.Tensor,
    num_logical: int,
) -> list[dict[int, int]]:
    """Decode a batch of P matrices.

    Args:
        P: (batch, h, h) doubly stochastic matrices.
        num_logical: Number of logical qubits (same for all samples in batch).

    Returns:
        List of layout dicts.
    """
    batch_size = P.shape[0]
    P_np = P.detach().cpu().numpy()
    layouts = []

    for b in range(batch_size):
        cost = 1.0 - P_np[b]
        row_ind, col_ind = linear_sum_assignment(cost)
        layout = {i: int(col_ind[i]) for i in range(num_logical)}
        layouts.append(layout)

    return layouts
