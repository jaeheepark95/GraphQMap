# src/kmw/losses/loss.py

"""
Loss functions for the KMW project.

What lives in this file
-----------------------
This file contains both:
1. the main mapping-task loss
2. the reindexer auxiliary losses

That split follows the project decision to keep one main loss file rather than
splitting task loss and reindexer loss into many small modules.

Main ideas
----------
The combined design says:

- the mapper predicts latent-frame logits S*
- those logits must be decoded back to native frame:
      S_nat = R_L^T @ S* @ R_H
- training assignment uses log-domain Sinkhorn on S_nat
- the task loss is computed in the native frame after decode
- the task loss contains:
      PST-related proxy cost
      SWAP proxy
      depth proxy

It also says:
- locality and consistency losses belong to the reindexer side
- for the first stable run, keep:
      alpha_loc = 0.0
      beta_cons = 0.0

Important terminology note
--------------------------
The clarification patch explicitly says that the quantities historically called
"L_PST_1Q", "L_PST_2Q", and "L_PST_total" are NOT literal physical PST values.
They are normalized optimization proxies built from normalized c1 / c2 tensors.

So, in code, we intentionally use names like:
    L_pst_proxy_1q
    L_pst_proxy_2q
    L_pst_proxy_total

That keeps the formulas the same while making the meaning clearer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from kmw.models.model import decode_to_native, sinkhorn_assignment


# =============================================================================
# 1) Configuration container
# =============================================================================

@dataclass
class LossConfig:
    """
    Central place for default loss-related hyperparameters.

    These defaults match the current locked implementation decisions:
    - task loss weights are active
    - locality / consistency losses are implemented but off by default
    - mapper training assignment uses stabilized log-domain Sinkhorn

    You can override these from config files later.
    """

    # Mapper Sinkhorn defaults
    tau_m: float = 0.10
    sinkhorn_iters: int = 20

    # Task-loss weights
    lambda_p: float = 1.0
    lambda_s: float = 1.0
    lambda_d: float = 0.25

    # Depth proxy coefficient
    kappa_depth: float = 1.0

    # Reindexer auxiliary-loss weights
    alpha_loc: float = 0.0
    beta_cons: float = 0.0

    # Numerical epsilon for safe division / clamping
    eps: float = 1e-6


# =============================================================================
# 2) Small tensor helpers
# =============================================================================

def _assert_finite(x: torch.Tensor, name: str) -> None:
    """
    Fail loudly if a tensor contains NaN or Inf.
    """
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or Inf.")


def _ensure_batched_square_matrix(x: torch.Tensor, name: str) -> torch.Tensor:
    """
    Accept:
        (N, N)    -> convert to (1, N, N)
        (B, N, N) -> keep as is
    """
    if x.ndim == 2:
        return x.unsqueeze(0)
    if x.ndim == 3:
        return x
    raise ValueError(f"{name} must have shape (N,N) or (B,N,N), got {tuple(x.shape)}")


def _ensure_batched_vector(x: torch.Tensor, name: str) -> torch.Tensor:
    """
    Accept:
        (N,)   -> convert to (1, N)
        (B, N) -> keep as is
    """
    if x.ndim == 1:
        return x.unsqueeze(0)
    if x.ndim == 2:
        return x
    raise ValueError(f"{name} must have shape (N,) or (B,N), got {tuple(x.shape)}")


def _assert_same_batch_size(*tensors: torch.Tensor) -> None:
    """
    Check that all tensors have the same batch size.
    """
    batch_sizes = [t.shape[0] for t in tensors]
    if len(set(batch_sizes)) != 1:
        raise ValueError(f"Batch-size mismatch across tensors: {batch_sizes}")


def _safe_mass(x: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Safe positive denominator:
        max(sum(x), eps)

    Returns a batch-shaped tensor if x is batched.
    """
    if x.ndim == 2:
        # (B, N)
        return x.sum(dim=-1).clamp(min=eps)
    if x.ndim == 3:
        # (B, N, N)
        return x.sum(dim=(-2, -1)).clamp(min=eps)
    raise ValueError(f"_safe_mass expects rank-2 or rank-3 tensor, got {tuple(x.shape)}")


def _offdiag(A: torch.Tensor) -> torch.Tensor:
    """
    Zero the diagonal of a batched square matrix.

    Input:
        A : (B, N, N)

    Output:
        A_off : (B, N, N)
    """
    A = _ensure_batched_square_matrix(A, "A")
    _, N, _ = A.shape
    eye = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0)
    return A * (1.0 - eye)


def _batch_mean_scalar(x: torch.Tensor) -> torch.Tensor:
    """
    Reduce a per-sample tensor to a scalar suitable for backward().

    Example:
        x shape (B,) -> scalar mean
    """
    if x.ndim == 0:
        return x
    return x.mean()


def _pair_expectation(P_map: torch.Tensor, pair_cost: torch.Tensor) -> torch.Tensor:
    """
    Compute expected pairwise physical cost for each logical pair.

    Mathematical meaning
    --------------------
    For each logical pair (u, v), compute:

        E_cost(u, v) =
            sum_{i, j} P_map[u, i] * P_map[v, j] * pair_cost[i, j]

    Shapes
    ------
    P_map     : (B, U, I)
    pair_cost : (B, I, J)

    Output
    ------
    E_cost    : (B, U, V)
              where U == V == number of logical rows (here 27)

    In our problem, U = V = I = J = 27.
    """
    return torch.einsum("bui,bij,bvj->buv", P_map, pair_cost, P_map)


def _index_distance_matrix(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Build the normalized squared index-distance matrix used in locality loss:

        dist[i, j] = ((i - j) / (n - 1))^2

    For n=27, this becomes:
        ((i - j) / 26)^2
    """
    idx = torch.arange(n, device=device, dtype=dtype)
    diff = idx[:, None] - idx[None, :]
    denom = max(n - 1, 1)
    return (diff / denom) ** 2


def _zero_like_scalar(reference: torch.Tensor) -> torch.Tensor:
    """
    Create a scalar zero tensor on the same device/dtype as 'reference'.
    """
    return torch.zeros((), device=reference.device, dtype=reference.dtype)


def _validate_required_loss_scalars(loss_dict: Dict[str, torch.Tensor]) -> None:
    """
    Fail loudly if any loss/diagnostic scalar in the dictionary is non-finite.

    We intentionally only validate tensor values here.
    """
    for key, value in loss_dict.items():
        if torch.is_tensor(value):
            _assert_finite(value, key)


# =============================================================================
# 3) Task loss: formulas computed in native frame after decode
# =============================================================================

def compute_task_loss_from_assignment(
    A: torch.Tensor,
    m: torch.Tensor,
    Bmat: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    D: torch.Tensor,
    P_map: torch.Tensor,
    config: Optional[LossConfig] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute the full native-frame task loss from an already-computed soft assignment.

    This function implements the formulas from the design:

    - active logical set determined by m
    - off-diagonal circuit mass from A_off
    - PST 1Q proxy term
    - PST 2Q proxy term using C2_use = B * c2
    - PST total proxy
    - SWAP proxy using D
    - depth proxy = kappa_depth * swap
    - total task loss with lambda weights

    Inputs
    ------
    A      : (B, N, N) or (N, N)
    m      : (B, N)    or (N,)
    Bmat   : (B, N, N) or (N, N)
    c1     : (B, N)    or (N,)
    c2     : (B, N, N) or (N, N)
    D      : (B, N, N) or (N, N)
    P_map  : (B, N, N) or (N, N)

    Returns
    -------
    Dictionary with:
        per-sample tensors
        scalar means for backward/logging

    Naming note
    -----------
    We use 'pst_proxy' names intentionally, because the clarification patch
    says these are normalized proxy costs, not literal PST probabilities.
    """
    if config is None:
        config = LossConfig()

    # Make all inputs batched for a uniform implementation.
    A = _ensure_batched_square_matrix(A, "A")
    m = _ensure_batched_vector(m, "m")
    Bmat = _ensure_batched_square_matrix(Bmat, "Bmat")
    c1 = _ensure_batched_vector(c1, "c1")
    c2 = _ensure_batched_square_matrix(c2, "c2")
    D = _ensure_batched_square_matrix(D, "D")
    P_map = _ensure_batched_square_matrix(P_map, "P_map")

    _assert_same_batch_size(A, m, Bmat, c1, c2, D, P_map)

    _assert_finite(A, "A")
    _assert_finite(m, "m")
    _assert_finite(Bmat, "Bmat")
    _assert_finite(c1, "c1")
    _assert_finite(c2, "c2")
    _assert_finite(D, "D")
    _assert_finite(P_map, "P_map")

    # -------------------------------------------------------------------------
    # 12.2 Off-diagonal circuit mass
    # -------------------------------------------------------------------------
    A_off = _offdiag(A)                         # (B, N, N)
    mass_2q = _safe_mass(A_off, config.eps)    # (B,)
    mass_1q = _safe_mass(m, config.eps)        # (B,)

    # -------------------------------------------------------------------------
    # 12.3 PST 1Q proxy
    #
    # L_PST_1Q_num =
    #     sum_u m[u] * sum_j P_map[u, j] * c1[j]
    #
    # We compute:
    #   expected_c1_per_logical[u] = sum_j P_map[u, j] * c1[j]
    # then weight by m[u].
    # -------------------------------------------------------------------------
    expected_c1_per_logical = torch.einsum("buj,bj->bu", P_map, c1)  # (B, N)
    L_pst_proxy_1q_num = (m * expected_c1_per_logical).sum(dim=-1)   # (B,)
    L_pst_proxy_1q_per_sample = L_pst_proxy_1q_num / mass_1q          # (B,)

    # -------------------------------------------------------------------------
    # 12.4 PST 2Q proxy
    #
    # C2_use = B * c2
    #
    # L_PST_2Q_num =
    #   sum_{u,v} A_off[u,v] *
    #   sum_{i,j} P_map[u,i] * P_map[v,j] * C2_use[i,j]
    # -------------------------------------------------------------------------
    C2_use = Bmat * c2
    expected_c2_per_pair = _pair_expectation(P_map, C2_use)          # (B, N, N)
    L_pst_proxy_2q_num = (A_off * expected_c2_per_pair).sum(dim=(-2, -1))
    L_pst_proxy_2q_per_sample = L_pst_proxy_2q_num / mass_2q         # (B,)

    # -------------------------------------------------------------------------
    # 12.5 PST total proxy
    # -------------------------------------------------------------------------
    L_pst_proxy_total_per_sample = (
        L_pst_proxy_1q_per_sample + L_pst_proxy_2q_per_sample
    )

    # -------------------------------------------------------------------------
    # 12.6 SWAP proxy
    #
    # E_D(u,v) =
    #     sum_{i,j} P_map[u,i] * P_map[v,j] * D[i,j]
    #
    # L_swap_num =
    #     sum_{u,v} A_off[u,v] * E_D(u,v)
    # -------------------------------------------------------------------------
    expected_D_per_pair = _pair_expectation(P_map, D)                # (B, N, N)
    L_swap_num = (A_off * expected_D_per_pair).sum(dim=(-2, -1))
    L_swap_per_sample = L_swap_num / mass_2q                         # (B,)

    # -------------------------------------------------------------------------
    # 12.7 Depth proxy
    # -------------------------------------------------------------------------
    L_depth_per_sample = config.kappa_depth * L_swap_per_sample

    # -------------------------------------------------------------------------
    # 12.8 Total task loss
    # -------------------------------------------------------------------------
    L_task_per_sample = (
        config.lambda_p * L_pst_proxy_total_per_sample
        + config.lambda_s * L_swap_per_sample
        + config.lambda_d * L_depth_per_sample
    )

    # Turn per-sample values into scalar means for optimization / logging.
    L_pst_proxy_1q = _batch_mean_scalar(L_pst_proxy_1q_per_sample)
    L_pst_proxy_2q = _batch_mean_scalar(L_pst_proxy_2q_per_sample)
    L_pst_proxy_total = _batch_mean_scalar(L_pst_proxy_total_per_sample)
    L_swap = _batch_mean_scalar(L_swap_per_sample)
    L_depth = _batch_mean_scalar(L_depth_per_sample)
    L_task = _batch_mean_scalar(L_task_per_sample)

    out = {
        # Main scalar losses used by the trainer
        "L_pst_proxy_1q": L_pst_proxy_1q,
        "L_pst_proxy_2q": L_pst_proxy_2q,
        "L_pst_proxy_total": L_pst_proxy_total,
        "L_swap": L_swap,
        "L_depth": L_depth,
        "L_task": L_task,

        # Per-sample diagnostics
        "L_pst_proxy_1q_per_sample": L_pst_proxy_1q_per_sample,
        "L_pst_proxy_2q_per_sample": L_pst_proxy_2q_per_sample,
        "L_pst_proxy_total_per_sample": L_pst_proxy_total_per_sample,
        "L_swap_per_sample": L_swap_per_sample,
        "L_depth_per_sample": L_depth_per_sample,
        "L_task_per_sample": L_task_per_sample,

        # Extra numerically useful diagnostics
        "mass_1q": mass_1q,
        "mass_2q": mass_2q,
        "L_pst_proxy_1q_num": L_pst_proxy_1q_num,
        "L_pst_proxy_2q_num": L_pst_proxy_2q_num,
        "L_swap_num": L_swap_num,
        "C2_use": C2_use,
    }

    _validate_required_loss_scalars(out)
    return out


def compute_task_loss_from_logits(
    S_star: torch.Tensor,
    R_L: torch.Tensor,
    R_H: torch.Tensor,
    A: torch.Tensor,
    m: torch.Tensor,
    Bmat: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    D: torch.Tensor,
    config: Optional[LossConfig] = None,
) -> Dict[str, torch.Tensor]:
    """
    Convenience helper that does the full mapper-task path:

        S* -> decode to native frame -> Sinkhorn assignment -> task loss

    This is the most direct helper for trainer.py when it already has:
    - latent logits S*
    - reindex matrices R_L / R_H
    - native-frame tensors A, m, B, c1, c2, D
    """
    if config is None:
        config = LossConfig()

    # Decode latent logits to native frame
    S_nat = decode_to_native(S_star=S_star, R_L=R_L, R_H=R_H)

    # Compute differentiable training assignment
    P_map = sinkhorn_assignment(
        S_nat=S_nat,
        tau_m=config.tau_m,
        num_iters=config.sinkhorn_iters,
    )

    # Compute native-frame task loss
    task = compute_task_loss_from_assignment(
        A=A,
        m=m,
        Bmat=Bmat,
        c1=c1,
        c2=c2,
        D=D,
        P_map=P_map,
        config=config,
    )

    task["S_nat"] = S_nat
    task["P_map"] = P_map

    _validate_required_loss_scalars(task)
    return task


def compute_task_loss_from_model_outputs(
    model_outputs: Dict[str, torch.Tensor],
    native_batch: Dict[str, torch.Tensor],
    config: Optional[LossConfig] = None,
) -> Dict[str, torch.Tensor]:
    """
    Convenience wrapper for trainer readability.

    Expected keys in model_outputs:
        S_star, R_L, R_H

    Expected keys in native_batch:
        A, m, B, c1, c2, D
    """
    required_model_keys = ("S_star", "R_L", "R_H")
    required_batch_keys = ("A", "m", "B", "c1", "c2", "D")

    for key in required_model_keys:
        if key not in model_outputs:
            raise KeyError(f"model_outputs is missing required key: {key}")

    for key in required_batch_keys:
        if key not in native_batch:
            raise KeyError(f"native_batch is missing required key: {key}")

    return compute_task_loss_from_logits(
        S_star=model_outputs["S_star"],
        R_L=model_outputs["R_L"],
        R_H=model_outputs["R_H"],
        A=native_batch["A"],
        m=native_batch["m"],
        Bmat=native_batch["B"],
        c1=native_batch["c1"],
        c2=native_batch["c2"],
        D=native_batch["D"],
        config=config,
    )


# =============================================================================
# 4) Reindexer auxiliary loss: locality
# =============================================================================

def compute_locality_loss(
    A_star: torch.Tensor,
    B_star: torch.Tensor,
    config: Optional[LossConfig] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute the locality auxiliary loss on reordered tensors.

    Design formulas
    ---------------
    Logical locality:
        W_mass = max(sum_{u,v} A*[u,v], eps)
        L_loc_log =
            (1 / W_mass) *
            sum_{u,v} A*[u,v] * ((u - v) / 26)^2

    Hardware locality:
        B_mass = max(sum_{i,j} B*[i,j], eps)
        L_loc_hw =
            (1 / B_mass) *
            sum_{i,j} B*[i,j] * ((i - j) / 26)^2

    Total locality:
        L_loc = 0.5 * (L_loc_log + L_loc_hw)

    Important note
    --------------
    This is implemented even though the initial stable run sets alpha_loc = 0.0.
    """
    if config is None:
        config = LossConfig()

    A_star = _ensure_batched_square_matrix(A_star, "A_star")
    B_star = _ensure_batched_square_matrix(B_star, "B_star")

    _assert_same_batch_size(A_star, B_star)
    _assert_finite(A_star, "A_star")
    _assert_finite(B_star, "B_star")

    _, N, _ = A_star.shape
    dist = _index_distance_matrix(n=N, device=A_star.device, dtype=A_star.dtype)  # (N, N)
    dist = dist.unsqueeze(0)  # (1, N, N) so broadcasting works over batch

    # Logical locality
    W_mass = _safe_mass(A_star, config.eps)
    L_loc_log_num = (A_star * dist).sum(dim=(-2, -1))
    L_loc_log_per_sample = L_loc_log_num / W_mass

    # Hardware locality
    B_mass = _safe_mass(B_star, config.eps)
    L_loc_hw_num = (B_star * dist).sum(dim=(-2, -1))
    L_loc_hw_per_sample = L_loc_hw_num / B_mass

    # Total locality
    L_loc_per_sample = 0.5 * (L_loc_log_per_sample + L_loc_hw_per_sample)

    out = {
        "L_loc_log": _batch_mean_scalar(L_loc_log_per_sample),
        "L_loc_hw": _batch_mean_scalar(L_loc_hw_per_sample),
        "L_loc": _batch_mean_scalar(L_loc_per_sample),
        "L_loc_log_per_sample": L_loc_log_per_sample,
        "L_loc_hw_per_sample": L_loc_hw_per_sample,
        "L_loc_per_sample": L_loc_per_sample,
        "W_mass": W_mass,
        "B_mass": B_mass,
    }

    _validate_required_loss_scalars(out)
    return out


# =============================================================================
# 5) Reindexer auxiliary loss: consistency
# =============================================================================

def sample_permutation_matrix(
    n: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample one permutation matrix Pi of shape (n, n).

    This is used for the consistency loss.

    Convention:
        Pi @ x
    relabels the rows of x according to the sampled permutation.
    """
    perm = torch.randperm(n, generator=generator, device=device)
    Pi = torch.zeros((n, n), device=device, dtype=dtype)
    Pi[torch.arange(n, device=device), perm] = 1.0
    return Pi


def relabel_vector(Pi: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Apply a relabeling to a batch of vectors:
        x' = Pi @ x

    Inputs:
        Pi : (N, N) or (B, N, N)
        x  : (B, N) or (N,)
    """
    x = _ensure_batched_vector(x, "x")

    if Pi.ndim == 2:
        Pi = Pi.unsqueeze(0).expand(x.shape[0], -1, -1)

    return torch.matmul(Pi, x.unsqueeze(-1)).squeeze(-1)


def relabel_matrix(Pi: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """
    Apply a relabeling to a batch of square matrices:
        M' = Pi @ M @ Pi^T

    Inputs:
        Pi : (N, N) or (B, N, N)
        M  : (B, N, N) or (N, N)
    """
    M = _ensure_batched_square_matrix(M, "M")

    if Pi.ndim == 2:
        Pi = Pi.unsqueeze(0).expand(M.shape[0], -1, -1)

    return torch.matmul(torch.matmul(Pi, M), Pi.transpose(1, 2))


def build_relabeled_native_batch(
    A: torch.Tensor,
    m: torch.Tensor,
    Bmat: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    D: torch.Tensor,
    Pi_L: torch.Tensor,
    Pi_H: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Construct the relabeled native inputs used for the consistency loss.

    Formulas:
        A'   = Pi_L A Pi_L^T
        m'   = Pi_L m
        B'   = Pi_H B Pi_H^T
        c1'  = Pi_H c1
        c2'  = Pi_H c2 Pi_H^T
        D'   = Pi_H D Pi_H^T
    """
    return {
        "A": relabel_matrix(Pi_L, A),
        "m": relabel_vector(Pi_L, m),
        "B": relabel_matrix(Pi_H, Bmat),
        "c1": relabel_vector(Pi_H, c1),
        "c2": relabel_matrix(Pi_H, c2),
        "D": relabel_matrix(Pi_H, D),
    }


def compute_consistency_loss(
    reindexer,
    A: torch.Tensor,
    m: torch.Tensor,
    Bmat: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    D: torch.Tensor,
    base_reindex_outputs: Dict[str, torch.Tensor],
    tau_r: float,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute the consistency auxiliary loss.

    Design rule
    -----------
    Per Pass-B update:
    - sample one random logical relabeling Pi_L
    - sample one random hardware relabeling Pi_H
    - use those same sampled relabelings consistently across all tensors
    - rerun the reindexer on the relabeled inputs
    - compare reordered outputs, NOT the permutations themselves

    Formula:
        L_cons =
            MSE(A*,  A*')
          + MSE(m*,  m*')
          + MSE(B*,  B*')
          + MSE(c1*, c1*')
          + MSE(c2*, c2*')
          + MSE(D*,  D*')

    Important engineering note
    --------------------------
    This function samples one relabeling pair for the whole batch, which matches
    the locked design and is especially natural because the default batch size is 1.
    """
    # Base outputs must already exist from the current reindexer forward pass.
    required_keys = ("A_star", "m_star", "B_star", "c1_star", "c2_star", "D_star")
    for key in required_keys:
        if key not in base_reindex_outputs:
            raise KeyError(f"base_reindex_outputs is missing required key: {key}")

    # Make sure native tensors are batched.
    A = _ensure_batched_square_matrix(A, "A")
    m = _ensure_batched_vector(m, "m")
    Bmat = _ensure_batched_square_matrix(Bmat, "Bmat")
    c1 = _ensure_batched_vector(c1, "c1")
    c2 = _ensure_batched_square_matrix(c2, "c2")
    D = _ensure_batched_square_matrix(D, "D")

    _assert_same_batch_size(A, m, Bmat, c1, c2, D)

    Bsz, N, _ = A.shape
    device = A.device
    dtype = A.dtype

    # Sample ONE logical permutation and ONE hardware permutation for this Pass-B step.
    Pi_L_single = sample_permutation_matrix(N, device=device, dtype=dtype, generator=generator)
    Pi_H_single = sample_permutation_matrix(N, device=device, dtype=dtype, generator=generator)

    # Reuse the same sampled permutations across all batch elements.
    Pi_L = Pi_L_single.unsqueeze(0).expand(Bsz, -1, -1)
    Pi_H = Pi_H_single.unsqueeze(0).expand(Bsz, -1, -1)

    # Build relabeled inputs
    relabeled = build_relabeled_native_batch(
        A=A,
        m=m,
        Bmat=Bmat,
        c1=c1,
        c2=c2,
        D=D,
        Pi_L=Pi_L,
        Pi_H=Pi_H,
    )

    # Rerun reindexer on relabeled inputs
    relabeled_outputs = reindexer(
        A=relabeled["A"],
        m=relabeled["m"],
        Bmat=relabeled["B"],
        c1=relabeled["c1"],
        c2=relabeled["c2"],
        D=relabeled["D"],
        tau_r=tau_r,
    )

    # Compare reordered forms, not raw permutations.
    mse_A_star = F.mse_loss(base_reindex_outputs["A_star"], relabeled_outputs["A_star"])
    mse_m_star = F.mse_loss(base_reindex_outputs["m_star"], relabeled_outputs["m_star"])
    mse_B_star = F.mse_loss(base_reindex_outputs["B_star"], relabeled_outputs["B_star"])
    mse_c1_star = F.mse_loss(base_reindex_outputs["c1_star"], relabeled_outputs["c1_star"])
    mse_c2_star = F.mse_loss(base_reindex_outputs["c2_star"], relabeled_outputs["c2_star"])
    mse_D_star = F.mse_loss(base_reindex_outputs["D_star"], relabeled_outputs["D_star"])

    L_cons = (
        mse_A_star
        + mse_m_star
        + mse_B_star
        + mse_c1_star
        + mse_c2_star
        + mse_D_star
    )

    out = {
        "L_cons": L_cons,
        "mse_A_star": mse_A_star,
        "mse_m_star": mse_m_star,
        "mse_B_star": mse_B_star,
        "mse_c1_star": mse_c1_star,
        "mse_c2_star": mse_c2_star,
        "mse_D_star": mse_D_star,
        "Pi_L": Pi_L,
        "Pi_H": Pi_H,
        "relabeled_outputs": relabeled_outputs,
    }

    _validate_required_loss_scalars({
        "L_cons": L_cons,
        "mse_A_star": mse_A_star,
        "mse_m_star": mse_m_star,
        "mse_B_star": mse_B_star,
        "mse_c1_star": mse_c1_star,
        "mse_c2_star": mse_c2_star,
        "mse_D_star": mse_D_star,
    })
    return out


# =============================================================================
# 6) Reindexer objective composition
# =============================================================================

def compose_reindex_loss(
    task_losses: Dict[str, torch.Tensor],
    locality_losses: Optional[Dict[str, torch.Tensor]] = None,
    consistency_losses: Optional[Dict[str, torch.Tensor]] = None,
    config: Optional[LossConfig] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compose the full reindexer objective:

        L_reindex = L_task + alpha_loc * L_loc + beta_cons * L_cons

    This helper is intentionally simple so trainer.py stays readable.
    """
    if config is None:
        config = LossConfig()

    if "L_task" not in task_losses:
        raise KeyError("task_losses must contain 'L_task'")

    L_task = task_losses["L_task"]

    if locality_losses is None:
        L_loc = _zero_like_scalar(L_task)
    else:
        if "L_loc" not in locality_losses:
            raise KeyError("locality_losses must contain 'L_loc'")
        L_loc = locality_losses["L_loc"]

    if consistency_losses is None:
        L_cons = _zero_like_scalar(L_task)
    else:
        if "L_cons" not in consistency_losses:
            raise KeyError("consistency_losses must contain 'L_cons'")
        L_cons = consistency_losses["L_cons"]

    L_reindex = L_task + config.alpha_loc * L_loc + config.beta_cons * L_cons

    out = {
        "L_task": L_task,
        "L_loc": L_loc,
        "L_cons": L_cons,
        "L_reindex": L_reindex,
    }

    _validate_required_loss_scalars(out)
    return out


# =============================================================================
# 7) Full Pass-B convenience helper
# =============================================================================

def compute_reindex_objective(
    reindexer,
    base_reindex_outputs: Dict[str, torch.Tensor],
    S_star: torch.Tensor,
    A: torch.Tensor,
    m: torch.Tensor,
    Bmat: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    D: torch.Tensor,
    tau_r: float,
    config: Optional[LossConfig] = None,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """
    Convenience helper for Pass-B in trainer.py.

    It computes:
    1. native-frame task loss from current S*
    2. locality loss on current reordered tensors
    3. consistency loss by relabeling and rerunning the reindexer
    4. final L_reindex

    Why this helper exists
    ----------------------
    The trainer is already structurally complicated because of the two-pass design.
    This function keeps Pass-B readable while still keeping all loss formulas in loss.py.

    Important
    ---------
    This helper assumes:
    - S_star was produced by the frozen-differentiable mapper
    - base_reindex_outputs came from the current non-detached reindexer forward
    - the actual parameter-freeze policy remains in trainer.py
    """
    if config is None:
        config = LossConfig()

    required_keys = ("R_L", "R_H", "A_star", "B_star")
    for key in required_keys:
        if key not in base_reindex_outputs:
            raise KeyError(f"base_reindex_outputs is missing required key: {key}")

    # 1) Task loss in native frame
    task_losses = compute_task_loss_from_logits(
        S_star=S_star,
        R_L=base_reindex_outputs["R_L"],
        R_H=base_reindex_outputs["R_H"],
        A=A,
        m=m,
        Bmat=Bmat,
        c1=c1,
        c2=c2,
        D=D,
        config=config,
    )

    # 2) Locality on reordered tensors
    locality_losses = compute_locality_loss(
        A_star=base_reindex_outputs["A_star"],
        B_star=base_reindex_outputs["B_star"],
        config=config,
    )

    # 3) Consistency by relabel -> rerun reindexer -> compare reordered outputs
    consistency_losses = compute_consistency_loss(
        reindexer=reindexer,
        A=A,
        m=m,
        Bmat=Bmat,
        c1=c1,
        c2=c2,
        D=D,
        base_reindex_outputs=base_reindex_outputs,
        tau_r=tau_r,
        generator=generator,
    )

    # 4) Compose final reindex objective
    reindex_obj = compose_reindex_loss(
        task_losses=task_losses,
        locality_losses=locality_losses,
        consistency_losses=consistency_losses,
        config=config,
    )

    out = {
        **task_losses,
        **locality_losses,
        **consistency_losses,
        **reindex_obj,
    }

    _validate_required_loss_scalars({
        "L_pst_proxy_1q": out["L_pst_proxy_1q"],
        "L_pst_proxy_2q": out["L_pst_proxy_2q"],
        "L_pst_proxy_total": out["L_pst_proxy_total"],
        "L_swap": out["L_swap"],
        "L_depth": out["L_depth"],
        "L_task": out["L_task"],
        "L_loc": out["L_loc"],
        "L_cons": out["L_cons"],
        "L_reindex": out["L_reindex"],
    })
    return out


# =============================================================================
# 8) Public export list
# =============================================================================

__all__ = [
    "LossConfig",
    "compute_task_loss_from_assignment",
    "compute_task_loss_from_logits",
    "compute_task_loss_from_model_outputs",
    "compute_locality_loss",
    "sample_permutation_matrix",
    "relabel_vector",
    "relabel_matrix",
    "build_relabeled_native_batch",
    "compute_consistency_loss",
    "compose_reindex_loss",
    "compute_reindex_objective",
]


