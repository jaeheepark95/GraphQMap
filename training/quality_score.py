"""Learnable quality score for physical qubits.

q_score(p) = sigmoid(w1*T1 + w2*T2 + w3*(1-readout_err) + w4*(1-sq_err) + w5*freq + bias)

Weights are learnable parameters — after training, inspecting them reveals
which noise factors matter most for PST.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QualityScore(nn.Module):
    """Learnable weighted quality score for physical qubits.

    Input features should be z-score normalized within backend,
    with error features inverted (1 - error) so "higher = better".

    Args:
        num_features: Number of input features (default 5:
            T1_norm, T2_norm, (1-readout_err_norm), (1-sq_err_norm), freq_norm).
        init_weight: Initial value for all weights.
    """

    def __init__(self, num_features: int = 5, init_weight: float = 0.2) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_features) * init_weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """Compute quality scores.

        Args:
            node_features: (h, num_features) tensor of hardware node features.

        Returns:
            (h,) tensor of quality scores in [0, 1].
        """
        score = torch.sigmoid(
            (node_features * self.weights).sum(dim=-1) + self.bias
        )
        return score
