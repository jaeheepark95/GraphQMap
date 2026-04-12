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


@register_loss("adjacency_error_aware")
class AdjacencyErrorAwareLoss(nn.Module):
    """L_adj_err: Error-aware adjacency matching with gate-frequency weighting.

    L_adj_err = -(1/W) * sum_{(i,j)} f_ij * sum_{p,q} P_ip * P_jq * A_hw(p,q) * (1 - eps_2Q(p,q))

    Extends L_adj by weighting adjacent edges by their fidelity (1 - 2Q error).
    Adjacent edges with lower error get higher reward.

    For non-adjacent pairs, A_hw=0 so the fidelity weight doesn't matter.
    For adjacent pairs, d_error(p,q) equals raw 2Q error (Floyd-Warshall
    shortest path on direct neighbors = single edge weight).

    Output bounded in [-1, 0]: -1 when all edges map to perfect adjacent qubits.
    """

    def forward(self, P: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Compute error-aware adjacency loss.

        Required kwargs: d_hw, d_error, circuit_edge_pairs.
        Optional kwargs: circuit_edge_weights.
        """
        d_hw = kwargs["d_hw"]
        d_error = kwargs["d_error"]
        circuit_edge_pairs = kwargs["circuit_edge_pairs"]
        circuit_edge_weights = kwargs.get("circuit_edge_weights", [])

        if not circuit_edge_pairs:
            return torch.tensor(0.0, device=P.device, requires_grad=True)

        # A_hw * (1 - eps_2Q): adjacent edges weighted by fidelity
        A_hw = (d_hw == 1).float()
        # For adjacent pairs, d_error gives raw 2Q error
        fidelity = (1.0 - d_error).clamp(min=0.0)
        A_fidelity = A_hw * fidelity

        if not circuit_edge_weights:
            circuit_edge_weights = [1.0] * len(circuit_edge_pairs)

        total_adj = torch.tensor(0.0, device=P.device)
        total_weight = 0.0

        for (i, j), w in zip(circuit_edge_pairs, circuit_edge_weights):
            P_i = P[:, i, :]
            P_j = P[:, j, :]
            adj_score = (P_i @ A_fidelity * P_j).sum(dim=-1)
            total_adj = total_adj + w * adj_score.mean()
            total_weight += w

        loss = -total_adj / max(total_weight, 1e-8)
        return loss


@register_loss("adjacency_size_aware")
class SizeAwareAdjacencyLoss(AdjacencyMatchingLoss):
    """L_adj with backend-size-dependent piecewise weighting.

    Multiplies the base adjacency loss by a per-backend multiplier based on
    physical qubit count h (read from d_hw.shape[-1]):
        - small  (h <= threshold_small):                    weight_small
        - medium (threshold_small < h <= threshold_large):  weight_medium
        - large  (h > threshold_large):                     weight_large

    Motivation (Phase D-2 diagnosis): the constant L_adj weight has a
    different optimum on different backend sizes. C1 (weight=1.0) showed
    +0.37 PST on Torino (133Q) but -0.42 on Toronto (27Q). Piecewise
    weighting lets each size class get the adjacency pressure it needs.

    Defaults match the C2 sweep target: 0.3 / 0.5 / 1.0 with thresholds
    aligned to the test backends (Toronto 27Q, Brooklyn 65Q, Torino 133Q).
    """

    def __init__(
        self,
        weight_small: float = 0.3,
        weight_medium: float = 0.5,
        weight_large: float = 1.0,
        threshold_small: int = 40,
        threshold_large: int = 80,
    ) -> None:
        super().__init__()
        self.weight_small = weight_small
        self.weight_medium = weight_medium
        self.weight_large = weight_large
        self.threshold_small = threshold_small
        self.threshold_large = threshold_large

    def forward(self, P: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        loss = super().forward(P, **kwargs)
        h = kwargs["d_hw"].shape[-1]
        if h <= self.threshold_small:
            mult = self.weight_small
        elif h <= self.threshold_large:
            mult = self.weight_medium
        else:
            mult = self.weight_large
        return loss * mult


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


@register_loss("swap_count")
class SwapCountLoss(nn.Module):
    """L_swap: SWAP count estimation loss with gate-frequency weighting.

    L_swap = (1/W) * sum_{(i,j) in E} f_ij * sum_{p,q} P_ip * P_jq * d_swap(p,q)

    where d_swap(p,q) = 3 * max(d_hop(p,q) - 1, 0).
    Factor 3 because each SWAP decomposes into 3 CX gates.
    Adjacent qubits have d_swap=0 (no SWAP needed).

    Args:
        normalize: If True, divide d_swap by its max value to normalize to [0, 1].
            This brings the loss scale in line with adjacency (~0.0-0.3 range)
            instead of raw CX counts (0-60+ range).
    """

    def __init__(self, normalize: bool = False) -> None:
        super().__init__()
        self.normalize = normalize

    def forward(self, P: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Compute SWAP count loss.

        Required kwargs: d_hw, circuit_edge_pairs.
        Optional kwargs: circuit_edge_weights (gate frequency per edge).
        """
        d_hw = kwargs["d_hw"]
        circuit_edge_pairs = kwargs["circuit_edge_pairs"]
        circuit_edge_weights = kwargs.get("circuit_edge_weights", [])

        if not circuit_edge_pairs:
            return torch.tensor(0.0, device=P.device, requires_grad=True)

        # d_swap = 3 * max(hop - 1, 0): adjacent=0, 2-hop=3, 3-hop=6, ...
        d_swap = 3.0 * (d_hw - 1).clamp(min=0)

        if self.normalize:
            d_max = d_swap.max().clamp(min=1e-8)
            d_swap = d_swap / d_max

        if not circuit_edge_weights:
            circuit_edge_weights = [1.0] * len(circuit_edge_pairs)

        total_loss = torch.tensor(0.0, device=P.device)
        total_weight = 0.0

        for (i, j), w in zip(circuit_edge_pairs, circuit_edge_weights):
            P_i = P[:, i, :]
            P_j = P[:, j, :]
            cost = (P_i @ d_swap * P_j).sum(dim=-1)
            total_loss = total_loss + w * cost.mean()
            total_weight += w

        return total_loss / max(total_weight, 1e-8)


@register_loss("soft_proximity")
class SoftProximityLoss(nn.Module):
    """L_soft: Exponential decay proximity reward with gate-frequency weighting.

    L_soft = -(1/W) * sum_{(i,j) in E} f_ij * sum_{p,q} P_ip * P_jq * reward(p,q)

    where reward(p,q) = exp(-alpha * max(d_hop(p,q) - 1, 0)).
    Adjacent: reward=1.0, 2-hop: reward=exp(-alpha), 3-hop: reward=exp(-2*alpha).

    Unlike adjacency (binary 0/1), provides non-zero gradient for non-adjacent
    qubits. alpha controls decay: alpha->inf recovers adjacency, alpha->0 gives
    uniform reward.

    Bounded in [-1, 0].

    Args:
        alpha: Exponential decay rate (default: 2.0).
    """

    def __init__(self, alpha: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, P: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Compute soft proximity loss.

        Required kwargs: d_hw, circuit_edge_pairs.
        Optional kwargs: circuit_edge_weights (gate frequency per edge).
        """
        d_hw = kwargs["d_hw"]
        circuit_edge_pairs = kwargs["circuit_edge_pairs"]
        circuit_edge_weights = kwargs.get("circuit_edge_weights", [])

        if not circuit_edge_pairs:
            return torch.tensor(0.0, device=P.device, requires_grad=True)

        # reward(p,q) = exp(-alpha * max(hop - 1, 0))
        reward = torch.exp(-self.alpha * (d_hw - 1).clamp(min=0))

        if not circuit_edge_weights:
            circuit_edge_weights = [1.0] * len(circuit_edge_pairs)

        total_reward = torch.tensor(0.0, device=P.device)
        total_weight = 0.0

        for (i, j), w in zip(circuit_edge_pairs, circuit_edge_weights):
            P_i = P[:, i, :]
            P_j = P[:, j, :]
            score = (P_i @ reward * P_j).sum(dim=-1)
            total_reward = total_reward + w * score.mean()
            total_weight += w

        loss = -total_reward / max(total_weight, 1e-8)
        return loss


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


@register_loss("node_placement_cost")
class NodePlacementCostLoss(nn.Module):
    """L_npc: Node-level placement cost with per-qubit circuit signal.

    cost(i, p) = n_1Q(i) * eps_1Q(p) + lambda_r * eps_readout(p)

    L_npc = sum_i sum_p P_ip * cost(i, p)

    Unlike the old node_quality loss (learnable MLP that collapsed to
    circuit-agnostic ranking), this uses precomputed constants only —
    no learnable parameters, so collapse is impossible.

    Gradient: dL/dP_ip = cost(i, p) = n_1Q(i) * eps_1Q(p) + lambda_r * eps_readout(p)
    Per-node circuit signal (n_1Q) appears directly in the gradient.

    Args:
        lambda_r: Weight for readout error term relative to 1Q gate term.
            Readout term is a global regularizer (same for all logical qubits).
    """

    def __init__(self, lambda_r: float = 1.0) -> None:
        super().__init__()
        self.lambda_r = lambda_r

    def forward(self, P: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Compute node placement cost loss.

        Required kwargs:
            grama_g_single: (l,) per-logical-qubit 1Q gate count.
            grama_s_gate: (h,) per-physical-qubit 1Q gate error (raw).
            grama_s_read: (h,) per-physical-qubit readout error (raw).
        """
        g_single = kwargs.get("grama_g_single")
        s_gate = kwargs.get("grama_s_gate")
        s_read = kwargs.get("grama_s_read")

        if g_single is None or s_gate is None or s_read is None:
            return torch.tensor(0.0, device=P.device, requires_grad=True)

        g_single = g_single.to(P.device)  # (l,)
        s_gate = s_gate.to(P.device)      # (h,)
        s_read = s_read.to(P.device)      # (h,)

        # cost(i, p) = n_1Q(i) * eps_1Q(p) + lambda_r * eps_readout(p)
        # Shape: (l, h)
        cost_matrix = g_single.unsqueeze(1) * s_gate.unsqueeze(0) \
            + self.lambda_r * s_read.unsqueeze(0)

        # L = sum_i sum_p P_ip * cost(i, p), averaged over batch
        loss = (P * cost_matrix.unsqueeze(0)).sum(dim=(-2, -1)).mean()
        return loss


@register_loss("grama")
class GraMALoss(nn.Module):
    """L_grama: trace-form QAP objective from GraMA (Piao et al. 2026, Eq. 5).

    Paper objective (with paper's X having shape (n_physical, k_logical)):

        f(X) = Tr(A X^T W X) + gamma * <S, X>

    where:
      - W (n×n): Floyd–Warshall on -log(1 - eps_2q) — multiplicative-fidelity-
        correct all-pairs distance between physical qubits.
      - A (k×k): binary (0/1) logical adjacency aggregated across all circuits
        in the (multi-programmed) logical workload.
      - S (n×k): single-qubit cost matrix S = s_read · 1_k^T + s_gate · g^T,
        where g[j] is the number of single-qubit gates on logical qubit j.
      - gamma (scalar): mean(2 W X_uniform A) / mean(S), computed once with the
        uniform-initialized continuous matrix X_c = (1/n) · 1_n 1_k^T (Eq. 9 in
        the paper). Acts as an automatic balancer between the two terms.

    Our convention: P has shape (B, l, h) = (batch, logical, physical), i.e.
    P = X^T relative to the paper. Substituting X = P^T:

        Tr(A X^T W X) = Tr(A P W P^T)
        <S, X>        = <S^T, P>

    Both terms are computed in batched einsum form. The loss is **fully
    differentiable in P** — A, W, S, and gamma are detached constants per batch.

    Args:
        binary_adjacency: If True (default, faithful to paper), the logical
            adjacency matrix entries are clipped to {0, 1}. If False, the raw
            2Q-gate interaction count is used as edge weight (frequency-weighted).
        gamma_mode: 'auto' (default) computes gamma per batch via Eq. 9;
            'fixed' uses gamma_fixed.
        gamma_fixed: Fixed gamma value when gamma_mode='fixed'.
        normalize_by_size: If True, divides the loss by (l + |E_log|) so that
            its magnitude is comparable across batches with different logical
            qubit counts. Default False (matches the paper, which sums).
    """

    def __init__(
        self,
        binary_adjacency: bool = True,
        gamma_mode: str = "auto",
        gamma_fixed: float = 1.0,
        normalize_by_size: bool = True,
    ) -> None:
        super().__init__()
        if gamma_mode not in ("auto", "fixed"):
            raise ValueError(f"gamma_mode must be 'auto' or 'fixed', got {gamma_mode}")
        self.binary_adjacency = binary_adjacency
        self.gamma_mode = gamma_mode
        self.gamma_fixed = gamma_fixed
        self.normalize_by_size = normalize_by_size

    def forward(self, P: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Compute GraMA QAP loss.

        Required kwargs:
            grama_W: (h, h) hardware weighted-distance matrix.
            grama_s_read: (h,) raw readout error per physical qubit.
            grama_s_gate: (h,) raw max single-qubit gate error per physical qubit.
            grama_g_single: (l,) per-logical 1Q gate count.
            circuit_edge_pairs: list of (i, j) logical qubit pairs with 2Q gates.
            circuit_edge_weights: optional list of 2Q gate counts per edge.
        """
        device = P.device
        dtype = P.dtype
        B, l, h = P.shape

        W = kwargs.get("grama_W")
        s_read = kwargs.get("grama_s_read")
        s_gate = kwargs.get("grama_s_gate")
        g_single = kwargs.get("grama_g_single")
        edge_pairs = kwargs.get("circuit_edge_pairs", [])
        edge_weights = kwargs.get("circuit_edge_weights", [])

        if W is None or s_read is None or s_gate is None or g_single is None:
            return torch.tensor(0.0, device=device, requires_grad=True)

        W = W.to(device=device, dtype=dtype)
        s_read = s_read.to(device=device, dtype=dtype)
        s_gate = s_gate.to(device=device, dtype=dtype)
        g_single = g_single.to(device=device, dtype=dtype)

        # ---- Build logical adjacency A (l × l) ----
        A = torch.zeros((l, l), device=device, dtype=dtype)
        if edge_pairs:
            if self.binary_adjacency or not edge_weights:
                weights = [1.0] * len(edge_pairs)
            else:
                weights = list(edge_weights)
            idx = torch.tensor(edge_pairs, device=device, dtype=torch.long)
            w = torch.tensor(weights, device=device, dtype=dtype)
            # Symmetric
            A[idx[:, 0], idx[:, 1]] = w
            A[idx[:, 1], idx[:, 0]] = w

        # ---- Build single-qubit cost S (h × l) ----
        # S[i, j] = s_read[i] + s_gate[i] * g_single[j]
        S = s_read.unsqueeze(1) + torch.outer(s_gate, g_single)  # (h, l)

        # ---- Term 1: trace QAP — Tr(A P W P^T) per batch ----
        # WP^T: einsum over physical: (h,h) @ (B,h,l) -> (B,h,l)
        WP_T = torch.einsum("hk,blk->bhl", W, P)        # (B, h, l) — actually (B,h,l)
        # A P: (l,l) @ (B,l,h) -> (B,l,h)
        AP = torch.einsum("ij,bjh->bih", A, P)          # (B, l, h)
        # trace term: sum over l,h of AP * (W P^T)^T = sum AP_{i,h} * WP_T[h, i] / per batch
        # Equivalent compact form: (AP * (WP_T transposed))
        trace_term = (AP * WP_T.transpose(1, 2)).sum(dim=(-1, -2))  # (B,)

        # ---- gamma (auto-balancing scalar, per Eq. 9) ----
        # X_c = (1/n) 1_n 1_k^T (paper's notation, n=h, k=l)
        # 2 W X_c A reduces to (2/n) * (W·1_h)[u] * (1_l^T A)[i]
        # Its mean equals (2/n) * mean_h(W·1) * mean_l(A·1)  (since all entries factor)
        with torch.no_grad():
            if self.gamma_mode == "fixed":
                gamma = torch.tensor(self.gamma_fixed, device=device, dtype=dtype)
            else:
                row_sum_W = W.sum(dim=1)                # (h,)
                col_sum_A = A.sum(dim=0)                # (l,)
                # 2WX_cA shape (h, l); mean factors:
                t_mean = (2.0 / h) * row_sum_W.mean() * col_sum_A.mean()
                s_mean = S.mean()
                if s_mean.abs() < 1e-12:
                    gamma = torch.tensor(0.0, device=device, dtype=dtype)
                else:
                    gamma = t_mean / s_mean
                # safety: clamp to non-negative finite range
                if not torch.isfinite(gamma):
                    gamma = torch.tensor(0.0, device=device, dtype=dtype)

        # ---- Term 2: <S^T, P> per batch ----
        # S has shape (h, l), so S^T (l, h) lines up with P (B, l, h).
        S_T = S.transpose(0, 1)                          # (l, h)
        single_term = (P * S_T.unsqueeze(0)).sum(dim=(-1, -2))  # (B,)

        # ---- Combine ----
        per_batch = trace_term + gamma * single_term     # (B,)

        if self.normalize_by_size:
            denom = float(l + max(len(edge_pairs), 1))
            per_batch = per_batch / denom

        return per_batch.mean()


@register_loss("qap_fidelity")
class QAPFidelityLoss(nn.Module):
    """L_qap: Unified QAP fidelity loss using C_eff.

    Implements tr(Ã_c P C_eff P^T) — the matrix form of the QAP objective
    where C_eff encodes SWAP chain costs (3×ε₂ per SWAP) and direct gate
    costs (ε₂ for adjacent pairs) in a single cost matrix.

    Advantages over separate error_distance + adjacency losses:
    - No artificial weighting between loss components needed
    - All terms in -log(fidelity) units — physically meaningful
    - SWAP 3× overhead correctly reflected (error_distance uses raw ε₂)
    - Gradient does not saturate: SWAP chain costs provide large dynamic range
    - Naturally size-aware: larger backends have larger C_eff values

    The gradient ∂L/∂P = 2·Ã_c·P·C_eff is also used for iterative score
    refinement (mirror descent feedback term).

    Args:
        normalize: If True, divide by (l + |E|) for cross-batch comparability.
    """

    def __init__(self, normalize: bool = True) -> None:
        super().__init__()
        self.normalize = normalize

    def forward(self, P: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Compute QAP fidelity loss.

        Required kwargs:
            c_eff: (h, h) effective cost matrix.
            circuit_adj: (l, l) gate-count weighted adjacency matrix Ã_c.
        """
        c_eff = kwargs.get("c_eff")
        circuit_adj = kwargs.get("circuit_adj")

        if c_eff is None or circuit_adj is None:
            return torch.tensor(0.0, device=P.device, requires_grad=True)

        c_eff = c_eff.to(device=P.device, dtype=P.dtype)
        circuit_adj = circuit_adj.to(device=P.device, dtype=P.dtype)

        # tr(Ã_c P C_eff P^T) = sum_{i,j,k,l} Ã_c[i,j] P[j,k] C_eff[k,l] P[i,l]
        # Efficient: PC = P @ C_eff, then sum (Ã_c @ PC) * P
        PC = torch.matmul(P, c_eff)          # (B, l, h)
        APC = torch.matmul(circuit_adj, PC)   # (B, l, h) — Ã_c broadcast over batch
        trace = (APC * P).sum(dim=(-2, -1))   # (B,)

        if self.normalize:
            l = P.size(1)
            num_edges = max((circuit_adj > 0).sum().item() // 2, 1)
            trace = trace / (l + num_edges)

        return trace.mean()


@register_loss("exclusion")
class ExclusionLoss(nn.Module):
    """L_excl: Pairwise collision loss for one-to-one mapping.

    Penalizes multiple logical qubits mapping to the same physical qubit
    by measuring pairwise collision probability:

    L_excl = (1/h) * sum_j sum_{i!=k} P_ij * P_kj
           = (1/h) * [sum_j c_j^2 - ||P||_F^2]

    Optimal value is 0 (each physical qubit assigned to at most one logical
    qubit). Active even during soft P (high tau), unlike threshold-based
    penalties. Gradient: dL/dP_ij = (2/h) * sum_{k!=i} P_kj, proportional
    to how much other logical qubits occupy physical qubit j.
    """

    def forward(self, P: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Compute pairwise collision loss.

        Args:
            P: (batch, l, h) row-stochastic matrix.
        """
        h = P.size(-1)
        col_sums = P.sum(dim=1)  # (batch, h)
        col_sums_sq = (col_sums ** 2).sum(dim=-1)  # (batch,)
        frobenius_sq = (P ** 2).sum(dim=(1, 2))  # (batch,)
        collision = (col_sums_sq - frobenius_sq) / h  # (batch,)
        return collision.mean()


# ---------------------------------------------------------------------------
# Combined Stage 2 Loss (built from config)
# ---------------------------------------------------------------------------

class Stage2Loss(nn.Module):
    """Combined Stage 2 loss built from config components.

    Each component is a registered loss with a weight. The total loss is
    the weighted sum of all active components.

    Args:
        components: List of dicts with 'name', 'weight', and optional 'params' keys.
            params is a dict of keyword arguments passed to the loss constructor.
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
            params = comp.get("params", {})

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
                self.losses[name] = cls(quality_score=quality_score, **params)
            else:
                self.losses[name] = cls(**params)

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
