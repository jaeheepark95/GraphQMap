"""Z-score normalization utilities for GraphQMap."""

from __future__ import annotations

import torch

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
