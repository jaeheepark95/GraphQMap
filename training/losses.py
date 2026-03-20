"""Loss functions for GraphQMap training.

Stage 1: L_sup  — Cross-entropy between P and ground-truth permutation matrix Y.
Stage 2: L_surr — Error-aware edge quality loss.
         L_node — NISQ node quality loss (uses learnable QualityScore).
         L_sep  — Multi-programming separation loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from training.quality_score import QualityScore


# ---------------------------------------------------------------------------
# Stage 1: Supervised Cross-Entropy Loss
# ---------------------------------------------------------------------------

class SupervisedCELoss(nn.Module):
    """Cross-entropy loss between Sinkhorn output P and label permutation Y.

    L_sup = -sum_{i,j} Y_ij * log(P_ij)

    Args:
        eps: Small value to clamp P for numerical stability in log.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, P: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss.

        Args:
            P: (batch, h, h) doubly stochastic matrix from Sinkhorn.
            Y: (batch, h, h) binary permutation matrix (ground truth).

        Returns:
            Scalar loss.
        """
        log_P = torch.log(P.clamp(min=self.eps))
        loss = -(Y * log_P).sum(dim=(-2, -1)).mean()
        return loss


# ---------------------------------------------------------------------------
# Stage 2: Surrogate Losses
# ---------------------------------------------------------------------------

class ErrorAwareEdgeLoss(nn.Module):
    """L_surr: Error-aware edge quality loss.

    Uses precomputed error-accumulated shortest path distances.

    L_surr = (1/|E_circuit|) * sum_{(i,j) in E_circuit} sum_{p,q} P_ip * P_jq * d_error(p,q)

    Fully differentiable w.r.t. P (d_error is a precomputed constant).
    """

    def forward(
        self,
        P: torch.Tensor,
        d_error: torch.Tensor,
        circuit_edge_pairs: list[tuple[int, int]],
        num_logical: int,
    ) -> torch.Tensor:
        """Compute error-aware edge loss.

        Args:
            P: (batch, h, h) doubly stochastic matrix.
            d_error: (h, h) precomputed error distance matrix for the backend.
            circuit_edge_pairs: List of (i, j) logical qubit pairs with 2-qubit gates.
            num_logical: Number of logical qubits.

        Returns:
            Scalar loss (per-pair normalized).
        """
        if not circuit_edge_pairs:
            return torch.tensor(0.0, device=P.device, requires_grad=True)

        # P_logical: (batch, l, h) — only the logical qubit rows
        P_logical = P[:, :num_logical, :]

        total_loss = torch.tensor(0.0, device=P.device)

        for i, j in circuit_edge_pairs:
            # P_i: (batch, h), P_j: (batch, h)
            P_i = P_logical[:, i, :]
            P_j = P_logical[:, j, :]
            # sum_{p,q} P_ip * P_jq * d_error(p,q)
            # = P_i @ d_error @ P_j^T  → (batch,) diagonal
            cost = (P_i @ d_error * P_j).sum(dim=-1)  # (batch,)
            total_loss = total_loss + cost.mean()

        return total_loss / len(circuit_edge_pairs)


class NodeQualityLoss(nn.Module):
    """L_node: NISQ node quality loss.

    Drives important logical qubits to high-quality physical qubits.

    L_node = -sum_i w_norm(i) * sum_p P_ip * q_score(p)

    Where w_norm is qubit_importance normalized to sum to 1 (probability
    distribution). This ensures L_node is a weighted average of q_scores,
    bounded in [-1, 0] regardless of circuit size or gate count.

    Args:
        quality_score: Learnable QualityScore module.
    """

    def __init__(self, quality_score: QualityScore) -> None:
        super().__init__()
        self.quality_score = quality_score

    def forward(
        self,
        P: torch.Tensor,
        hw_node_features: torch.Tensor,
        qubit_importance: torch.Tensor,
        num_logical: int,
    ) -> torch.Tensor:
        """Compute node quality loss.

        Args:
            P: (batch, h, h) doubly stochastic matrix.
            hw_node_features: (h, 5) hardware node features
                [T1_norm, T2_norm, (1-readout_err_norm), (1-sq_err_norm), freq_norm].
            qubit_importance: (l,) importance weights per logical qubit
                (= number of 2-qubit gates involving each qubit).
            num_logical: Number of logical qubits.

        Returns:
            Scalar loss in [-1, 0].
        """
        # q_score: (h,) quality scores in [0, 1]
        q_scores = self.quality_score(hw_node_features)

        # P_logical: (batch, l, h)
        P_logical = P[:, :num_logical, :]

        # Expected quality per logical qubit: (batch, l)
        expected_quality = (P_logical * q_scores.unsqueeze(0).unsqueeze(0)).sum(dim=-1)

        # Normalize importance to probability distribution (sum = 1)
        w = qubit_importance.unsqueeze(0)  # (1, l)
        w_sum = w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        w_norm = w / w_sum  # (1, l), sums to 1

        # Weighted average of expected quality: bounded in [0, 1]
        weighted = (w_norm * expected_quality).sum(dim=-1)  # (batch,)

        # Negative: higher quality → lower loss → bounded in [-1, 0]
        loss = -weighted.mean()
        return loss


class SeparationLoss(nn.Module):
    """L_sep: Multi-programming separation loss.

    Encourages physical distance between qubits of different circuits.

    L_sep = (1/|E_cross|) * (-sum_{(i,j) in cross-circuit} sum_{p,q} P_ip * P_jq * d_hw_norm(p,q))

    d_hw is normalized by its maximum value to [0, 1], ensuring L_sep is
    bounded in [-1, 0] and comparable in scale to L_surr and L_node.

    Automatically equals 0 for single-circuit scenarios.
    """

    def forward(
        self,
        P: torch.Tensor,
        d_hw: torch.Tensor,
        cross_circuit_pairs: list[tuple[int, int]],
        num_logical: int,
    ) -> torch.Tensor:
        """Compute separation loss.

        Args:
            P: (batch, h, h) doubly stochastic matrix.
            d_hw: (h, h) hardware distance matrix (hop count or error-weighted).
            cross_circuit_pairs: List of (i, j) pairs from different circuits.
            num_logical: Total number of logical qubits across all circuits.

        Returns:
            Scalar loss in [-1, 0] (per-pair normalized).
        """
        if not cross_circuit_pairs:
            return torch.tensor(0.0, device=P.device, requires_grad=True)

        P_logical = P[:, :num_logical, :]

        # Normalize d_hw to [0, 1] for scale stability
        d_max = d_hw.max().clamp(min=1e-8)
        d_hw_norm = d_hw / d_max

        total_dist = torch.tensor(0.0, device=P.device)

        for i, j in cross_circuit_pairs:
            P_i = P_logical[:, i, :]
            P_j = P_logical[:, j, :]
            dist = (P_i @ d_hw_norm * P_j).sum(dim=-1)  # (batch,)
            total_dist = total_dist + dist.mean()

        # Negative: larger distance → lower loss
        loss = -total_dist / len(cross_circuit_pairs)
        return loss


# ---------------------------------------------------------------------------
# Combined Stage 2 Loss
# ---------------------------------------------------------------------------

class Stage2Loss(nn.Module):
    """Combined Stage 2 loss: L_2 = L_surr + alpha * L_node + lambda * L_sep.

    Args:
        quality_score: Learnable QualityScore module.
        alpha: Weight for L_node.
        lambda_sep: Weight for L_sep.
    """

    def __init__(
        self,
        quality_score: QualityScore,
        alpha: float = 0.3,
        lambda_sep: float = 0.1,
    ) -> None:
        super().__init__()
        self.l_surr = ErrorAwareEdgeLoss()
        self.l_node = NodeQualityLoss(quality_score)
        self.l_sep = SeparationLoss()
        self.alpha = alpha
        self.lambda_sep = lambda_sep

    def forward(
        self,
        P: torch.Tensor,
        d_error: torch.Tensor,
        d_hw: torch.Tensor,
        hw_node_features: torch.Tensor,
        circuit_edge_pairs: list[tuple[int, int]],
        cross_circuit_pairs: list[tuple[int, int]],
        qubit_importance: torch.Tensor,
        num_logical: int,
    ) -> dict[str, torch.Tensor]:
        """Compute combined Stage 2 loss.

        Returns:
            Dict with 'total', 'l_surr', 'l_node', 'l_sep' loss tensors.
        """
        loss_surr = self.l_surr(P, d_error, circuit_edge_pairs, num_logical)
        loss_node = self.l_node(P, hw_node_features, qubit_importance, num_logical)
        loss_sep = self.l_sep(P, d_hw, cross_circuit_pairs, num_logical)

        total = loss_surr + self.alpha * loss_node + self.lambda_sep * loss_sep

        return {
            "total": total,
            "l_surr": loss_surr,
            "l_node": loss_node,
            "l_sep": loss_sep,
        }
