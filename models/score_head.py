"""Score Head for GraphQMap.

Computes compatibility scores between circuit and hardware embeddings:
  S_ij = (C'_i · W_q)^T · (H'_j · W_k) / sqrt(d_k)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ScoreHead(nn.Module):
    """Learned projection score head.

    Args:
        d_model: Input embedding dimension.
        d_k: Projection dimension for scoring.
    """

    def __init__(self, d_model: int = 64, d_k: int = 64) -> None:
        super().__init__()
        self.d_k = d_k
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.scale = math.sqrt(d_k)

    def forward(self, C: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Compute score matrix.

        Args:
            C: Circuit embeddings (batch, l, d).
            H: Hardware embeddings (batch, h, d).

        Returns:
            Score matrix S (batch, l, h).
        """
        Q = self.W_q(C)  # (batch, l, d_k)
        K = self.W_k(H)  # (batch, h, d_k)
        S = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (batch, l, h)
        return S
