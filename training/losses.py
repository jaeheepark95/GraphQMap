"""Loss functions for GraphQMap training.

Stage 1: L_sup  — Cross-entropy between P and ground-truth permutation matrix Y.
Stage 2: Configurable loss components via registry pattern.
         Available components:
         - error_distance: Error-weighted shortest path distance (L_surr)
         - adjacency: Adjacency matching with gate-frequency weighting (L_adj)
         - hop_distance: Hop distance tiebreaker (L_hop)
         - node_quality: NISQ node quality via learnable MLP (L_node)
         - separation: Multi-programming separation (L_sep)
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

from training.quality_score import QualityScore


# ---------------------------------------------------------------------------
# Loss Registry
# ---------------------------------------------------------------------------

# Maps component name -> (class, requires_quality_score)
_LOSS_REGISTRY: dict[str, tuple[type[nn.Module], bool]] = {}


def register_loss(name: str, requires_quality_score: bool = False) -> Callable:
    """Decorator to register a loss component."""
    def decorator(cls: type[nn.Module]) -> type[nn.Module]:
        _LOSS_REGISTRY[name] = (cls, requires_quality_score)
        return cls
    return decorator


def get_available_losses() -> list[str]:
    """Return list of registered loss component names."""
    return list(_LOSS_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Stage 1: Supervised Cross-Entropy Loss
# ---------------------------------------------------------------------------

class SupervisedCELoss(nn.Module):
    """Cross-entropy loss between softmax output P and label matrix Y.

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
            P: (batch, l, h) row-stochastic matrix from softmax.
            Y: (batch, l, h) binary label matrix (ground truth).

        Returns:
            Scalar loss.
        """
        log_P = torch.log(P.clamp(min=self.eps))
        loss = -(Y * log_P).sum(dim=(-2, -1)).mean()
        return loss


# ---------------------------------------------------------------------------
# Stage 2: Surrogate Loss Components
# ---------------------------------------------------------------------------

@register_loss("error_distance")
class ErrorAwareEdgeLoss(nn.Module):
    """L_surr: Error-aware edge quality loss.

    Uses precomputed error-accumulated shortest path distances,
    weighted by gate frequency per edge.

    L_surr = (1/W) * sum_{(i,j) in E_circuit} f_ij * sum_{p,q} P_ip * P_jq * d_error(p,q)

    where f_ij is the 2Q gate count for pair (i,j) and W = sum f_ij.
    Fully differentiable w.r.t. P (d_error is a precomputed constant).
    """

    def forward(self, P: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Compute error-aware edge loss.

        Required kwargs: d_error, circuit_edge_pairs.
        Optional kwargs: circuit_edge_weights (gate frequency per edge).
        """
        d_error = kwargs["d_error"]
        circuit_edge_pairs = kwargs["circuit_edge_pairs"]
        circuit_edge_weights = kwargs.get("circuit_edge_weights", [])

        if not circuit_edge_pairs:
            return torch.tensor(0.0, device=P.device, requires_grad=True)

        if not circuit_edge_weights:
            circuit_edge_weights = [1.0] * len(circuit_edge_pairs)

        total_loss = torch.tensor(0.0, device=P.device)
        total_weight = 0.0

        for (i, j), w in zip(circuit_edge_pairs, circuit_edge_weights):
            P_i = P[:, i, :]
            P_j = P[:, j, :]
            cost = (P_i @ d_error * P_j).sum(dim=-1)
            total_loss = total_loss + w * cost.mean()
            total_weight += w

        return total_loss / max(total_weight, 1e-8)


@register_loss("adjacency")
class AdjacencyMatchingLoss(nn.Module):
    """L_adj: Adjacency matching loss with gate-frequency weighting.

    L_adj = -(1/W) * sum_{(i,j) in E_circuit} w_ij * sum_{p,q} P_ip * P_jq * A_hw(p,q)

    Output is bounded in [-1, 0]: -1 when all edges map to adjacent qubits.
    """

    def forward(self, P: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Compute adjacency matching loss.

        Required kwargs: d_hw, circuit_edge_pairs.
        Optional kwargs: circuit_edge_weights.
        """
        d_hw = kwargs["d_hw"]
        circuit_edge_pairs = kwargs["circuit_edge_pairs"]
        circuit_edge_weights = kwargs.get("circuit_edge_weights", [])

        if not circuit_edge_pairs:
            return torch.tensor(0.0, device=P.device, requires_grad=True)

        A_hw = (d_hw == 1).float()

        if not circuit_edge_weights:
            circuit_edge_weights = [1.0] * len(circuit_edge_pairs)

        total_adj = torch.tensor(0.0, device=P.device)
        total_weight = 0.0

        for (i, j), w in zip(circuit_edge_pairs, circuit_edge_weights):
            P_i = P[:, i, :]
            P_j = P[:, j, :]
            adj_score = (P_i @ A_hw * P_j).sum(dim=-1)
            total_adj = total_adj + w * adj_score.mean()
            total_weight += w

        loss = -total_adj / max(total_weight, 1e-8)
        return loss


@register_loss("hop_distance")
class HopDistanceLoss(nn.Module):
    """L_hop: Hop distance penalty for physical proximity.

    L_hop = (1/|E|) * sum_{(i,j) in E} sum_{p,q} P_ip * P_jq * d_hop_norm(p,q)

    d_hop is normalized by max value to [0, 1].
    """

    def forward(self, P: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Compute hop distance loss.

        Required kwargs: d_hw, circuit_edge_pairs.
        """
        d_hw = kwargs["d_hw"]
        circuit_edge_pairs = kwargs["circuit_edge_pairs"]

        if not circuit_edge_pairs:
            return torch.tensor(0.0, device=P.device, requires_grad=True)

        d_max = d_hw.max().clamp(min=1e-8)
        d_hop_norm = d_hw / d_max

        total_loss = torch.tensor(0.0, device=P.device)

        for i, j in circuit_edge_pairs:
            P_i = P[:, i, :]
            P_j = P[:, j, :]
            cost = (P_i @ d_hop_norm * P_j).sum(dim=-1)
            total_loss = total_loss + cost.mean()

        return total_loss / len(circuit_edge_pairs)


@register_loss("node_quality", requires_quality_score=True)
class NodeQualityLoss(nn.Module):
    """L_node: NISQ node quality loss.

    L_node = -sum_i w_norm(i) * sum_p P_ip * q_score(p)

    Bounded in [-1, 0].

    Args:
        quality_score: Learnable QualityScore module.
    """

    def __init__(self, quality_score: QualityScore) -> None:
        super().__init__()
        self.quality_score = quality_score

    def forward(self, P: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Compute node quality loss.

        Required kwargs: hw_node_features, qubit_importance.
        """
        hw_node_features = kwargs["hw_node_features"]
        qubit_importance = kwargs["qubit_importance"]

        q_scores = self.quality_score(hw_node_features)
        expected_quality = (P * q_scores.unsqueeze(0).unsqueeze(0)).sum(dim=-1)

        w = qubit_importance.unsqueeze(0)
        w_sum = w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        w_norm = w / w_sum

        weighted = (w_norm * expected_quality).sum(dim=-1)
        loss = -weighted.mean()
        return loss


@register_loss("separation")
class SeparationLoss(nn.Module):
    """L_sep: Multi-programming separation loss.

    Encourages physical distance between qubits of different circuits.
    Bounded in [-1, 0]. Automatically equals 0 for single-circuit scenarios.
    """

    def forward(self, P: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Compute separation loss.

        Required kwargs: d_hw, cross_circuit_pairs.
        """
        d_hw = kwargs["d_hw"]
        cross_circuit_pairs = kwargs["cross_circuit_pairs"]

        if not cross_circuit_pairs:
            return torch.tensor(0.0, device=P.device, requires_grad=True)

        d_max = d_hw.max().clamp(min=1e-8)
        d_hw_norm = d_hw / d_max

        total_dist = torch.tensor(0.0, device=P.device)

        for i, j in cross_circuit_pairs:
            P_i = P[:, i, :]
            P_j = P[:, j, :]
            dist = (P_i @ d_hw_norm * P_j).sum(dim=-1)
            total_dist = total_dist + dist.mean()

        loss = -total_dist / len(cross_circuit_pairs)
        return loss


@register_loss("exclusion")
class ExclusionLoss(nn.Module):
    """L_excl: Column-wise exclusion loss for one-to-one mapping.

    Penalizes multiple logical qubits mapping to the same physical qubit.

    L_excl = (1/h) * sum_j (sum_i P_ij)^2

    When P is a perfect one-to-one mapping, each column sum is 0 or 1,
    giving L_excl = l/h. Values above l/h indicate column collisions.
    Bounded in [l/h, l] (l/h = best, l = all logical map to same physical).
    """

    def forward(self, P: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Compute exclusion loss.

        Args:
            P: (batch, l, h) row-stochastic matrix.
        """
        col_sums = P.sum(dim=1)  # (batch, h)
        loss = (col_sums ** 2).mean(dim=-1).mean()
        return loss


# ---------------------------------------------------------------------------
# Combined Stage 2 Loss (built from config)
# ---------------------------------------------------------------------------

class Stage2Loss(nn.Module):
    """Combined Stage 2 loss built from config components.

    Each component is a registered loss with a weight. The total loss is
    the weighted sum of all active components.

    Args:
        components: List of dicts with 'name' and 'weight' keys.
        quality_score: Learnable QualityScore module (required if any
            component needs it).
    """

    def __init__(
        self,
        components: list[dict[str, Any]],
        quality_score: QualityScore | None = None,
    ) -> None:
        super().__init__()
        self.component_names: list[str] = []
        self.component_weights: list[float] = []
        self.losses = nn.ModuleDict()

        for comp in components:
            name = comp["name"]
            weight = comp.get("weight", 1.0)

            if name not in _LOSS_REGISTRY:
                raise ValueError(
                    f"Unknown loss component '{name}'. "
                    f"Available: {get_available_losses()}"
                )

            cls, needs_qs = _LOSS_REGISTRY[name]
            if needs_qs:
                if quality_score is None:
                    raise ValueError(
                        f"Loss component '{name}' requires quality_score but none provided."
                    )
                self.losses[name] = cls(quality_score=quality_score)
            else:
                self.losses[name] = cls()

            self.component_names.append(name)
            self.component_weights.append(weight)

    def forward(self, P: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Compute combined Stage 2 loss.

        All kwargs are passed through to each component.

        Returns:
            Dict with 'total' and per-component loss tensors (keyed by name).
        """
        result: dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=P.device)

        for name, weight in zip(self.component_names, self.component_weights):
            loss_val = self.losses[name](P, **kwargs)
            result[name] = loss_val
            total = total + weight * loss_val

        result["total"] = total
        return result
