"""Score normalization layers for GraphQMap.

Provides SoftmaxNorm (row-wise softmax) as the primary normalization.
Sinkhorn kept for reference/experimentation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxNorm(nn.Module):
    """Row-wise softmax normalization for score matrix.

    Converts S (batch, l, h) to P (batch, l, h) where each row sums to 1.
    No dummy padding needed — avoids the l << h instability of Sinkhorn.

    Args:
        max_iter: Unused, kept for API compatibility with SinkhornLayer.
        tol: Unused, kept for API compatibility with SinkhornLayer.
    """

    def __init__(self, max_iter: int = 20, tol: float = 1e-6) -> None:
        super().__init__()

    def forward(
        self,
        S: torch.Tensor,
        num_logical: int,
        num_physical: int,
        tau: float,
    ) -> torch.Tensor:
        """Apply row-wise softmax with temperature scaling.

        Args:
            S: Score matrix (batch, l, h) where l = num_logical, h = num_physical.
            num_logical: Number of logical qubits (l). Unused, kept for API compat.
            num_physical: Number of physical qubits (h). Unused, kept for API compat.
            tau: Temperature for softmax (lower = sharper).

        Returns:
            P: Row-stochastic matrix (batch, l, h).
        """
        return F.softmax(S / tau, dim=-1)


def log_sinkhorn(
    log_alpha: torch.Tensor,
    max_iter: int = 20,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Log-domain Sinkhorn normalization.

    Args:
        log_alpha: (batch, h, h) or (h, h) matrix in log domain (= S_padded / τ).
        max_iter: Maximum Sinkhorn iterations.
        tol: Early stopping tolerance.

    Returns:
        P: Doubly stochastic matrix of same shape.
    """
    for _ in range(max_iter):
        # Row normalization in log domain
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        # Column normalization in log domain
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)

        # Early stopping: check row sums
        row_sums = log_alpha.exp().sum(dim=-1)
        if (row_sums - 1).abs().max() < tol:
            break

    return log_alpha.exp()


class SinkhornLayer(nn.Module):
    """Sinkhorn layer with dummy padding (legacy).

    Pads the score matrix from (batch, l, h) to (batch, h, h) with dummy rows,
    then applies log-domain Sinkhorn normalization.

    Args:
        max_iter: Maximum Sinkhorn iterations.
        tol: Early stopping tolerance.
    """

    def __init__(self, max_iter: int = 20, tol: float = 1e-6) -> None:
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol

    def forward(
        self,
        S: torch.Tensor,
        num_logical: int,
        num_physical: int,
        tau: float,
    ) -> torch.Tensor:
        """Apply dummy padding and Sinkhorn.

        Args:
            S: Score matrix (batch, l, h) where l = num_logical, h = num_physical.
            num_logical: Number of logical qubits (l).
            num_physical: Number of physical qubits (h).
            tau: Sinkhorn temperature.

        Returns:
            P: Doubly stochastic matrix (batch, h, h).
        """
        batch_size = S.shape[0]
        dummy_rows = num_physical - num_logical

        if dummy_rows > 0:
            dummy = torch.zeros(
                batch_size, dummy_rows, num_physical,
                device=S.device, dtype=S.dtype,
            )
            S_padded = torch.cat([S, dummy], dim=1)  # (batch, h, h)
        else:
            S_padded = S

        # Scale by temperature and apply log-domain Sinkhorn
        log_alpha = S_padded / tau
        P = log_sinkhorn(log_alpha, max_iter=self.max_iter, tol=self.tol)

        return P
