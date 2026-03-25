"""Learnable quality score for physical qubits.

Uses a 2-layer MLP to capture nonlinear relationships between
hardware noise features and qubit quality. Weights are learnable
parameters — after training, inspecting them reveals which noise
factors matter most for PST.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QualityScore(nn.Module):
    """Learnable MLP quality score for physical qubits.

    Uses a 2-layer MLP to capture nonlinear relationships between
    hardware noise features and qubit quality.

    Args:
        num_features: Number of input features (default 7:
            T1, T2, readout_error, single_qubit_error, degree, t1_cx_ratio, t2_cx_ratio).
        hidden_dim: Hidden layer dimension.
    """

    def __init__(self, num_features: int = 7, hidden_dim: int = 16) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """Compute quality scores.

        Args:
            node_features: (h, num_features) tensor of hardware node features.

        Returns:
            (h,) tensor of quality scores in [0, 1].
        """
        return torch.sigmoid(self.mlp(node_features).squeeze(-1))
