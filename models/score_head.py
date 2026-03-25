"""Score Head for GraphQMap.

Computes compatibility scores between circuit and hardware embeddings:
  S_ij = (C'_i · W_q)^T · (H'_j · W_k) / sqrt(d_k) + noise_bias_j

The noise_bias encourages mapping to low-error physical qubits,
similar to QAP's readout/gate error cost terms.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ScoreHead(nn.Module):
    """Learned projection score head with optional noise-aware bias.

    Args:
        d_model: Input embedding dimension.
        d_k: Projection dimension for scoring.
        noise_bias_dim: Number of hardware node features for bias (0 to disable).
    """

    def __init__(
        self, d_model: int = 64, d_k: int = 64, noise_bias_dim: int = 7,
    ) -> None:
        super().__init__()
        self.d_k = d_k
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.scale = math.sqrt(d_k)

        if noise_bias_dim > 0:
            self.noise_proj = nn.Linear(noise_bias_dim, 1, bias=True)
        else:
            self.noise_proj = None

    def forward(
        self,
        C: torch.Tensor,
        H: torch.Tensor,
        hw_node_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute score matrix.

        Args:
            C: Circuit embeddings (batch, l, d).
            H: Hardware embeddings (batch, h, d).
            hw_node_features: Hardware noise features (h, num_features).
                If provided and noise_proj exists, adds per-qubit bias.

        Returns:
            Score matrix S (batch, l, h).
        """
        Q = self.W_q(C)  # (batch, l, d_k)
        K = self.W_k(H)  # (batch, h, d_k)
        S = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (batch, l, h)

        if self.noise_proj is not None and hw_node_features is not None:
            # bias: (h,) → (1, 1, h) broadcast to all logical qubits
            bias = self.noise_proj(hw_node_features).squeeze(-1)  # (h,)
            S = S + bias.unsqueeze(0).unsqueeze(0)

        return S
