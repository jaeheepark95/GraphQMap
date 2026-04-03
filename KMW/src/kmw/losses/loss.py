# =============================================================================
# UPDATE LOG (2026-03-30, v1.4.1)
# - Replaced the older normalized PST/swap/depth proxy family with the v1.4.1
#   execution-surrogate loss:
#     * L_1q, L_ro, L_2q, L_native, L_route, L_task
#     * S_proxy_1q, S_proxy_ro, S_proxy_2q, S_proxy_route, S_proxy_exec
# - The 1Q and readout terms now use logical operation counts n1q / nmeas.
# - The routing term now uses a reliability-weighted route-cost matrix C_route
#   instead of an average-hop hazard.
# - The final execution quantity is treated as a surrogate score, not literal PST.
# =============================================================================
# =============================================================================
# UPDATE LOG (2026-03-30)
# - Added logical-only consistency behavior when the reindexer is running in
#   freeze_hardware_reindex / identity-R_H ablation mode.
# - This keeps stage-2 beta_cons usable during the clean R_H=I ablation by
#   avoiding impossible hardware-side consistency targets.
# =============================================================================

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
    """Central place for task-loss and auxiliary-loss hyperparameters."""

    # Mapper Sinkhorn defaults
    tau_m: float = 0.10
    sinkhorn_iters: int = 20

    # Numerical epsilons
    eps: float = 1e-6
    eps_surv: float = 1e-12

    # Reliability-weighted routing-step coefficients
    route_step_2q_mult: float = 3.0
    route_step_1q_mult: float = 2.0

    # Reindexer auxiliary-loss weights
    alpha_loc: float = 0.0
    beta_cons: float = 0.0



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
    """Compute expected pairwise physical cost for each logical pair."""
    return torch.einsum("bui,bij,bvj->buv", P_map, pair_cost, P_map)


def _safe_neglog_survival(survival: torch.Tensor, eps_surv: float) -> torch.Tensor:
    survival = survival.clamp(min=eps_surv, max=1.0)
    return -torch.log(survival)


def _compute_route_cost_matrix(
    Bmat: torch.Tensor,
    D_raw: torch.Tensor,
    e1q: torch.Tensor,
    e2q: torch.Tensor,
    config: LossConfig,
) -> torch.Tensor:
    """Build the reliability-weighted route-cost matrix C_route."""
    Bmat = _ensure_batched_square_matrix(Bmat, "Bmat")
    D_raw = _ensure_batched_square_matrix(D_raw, "D_raw")
    e1q = _ensure_batched_vector(e1q, "e1q")
    e2q = _ensure_batched_square_matrix(e2q, "e2q")

    q1q = _safe_neglog_survival(1.0 - e1q, config.eps_surv)
    q2q = _safe_neglog_survival(1.0 - e2q, config.eps_surv)
    c_step = config.route_step_2q_mult * q2q + 0.5 * config.route_step_1q_mult * (q1q.unsqueeze(-1) + q1q.unsqueeze(-2))
    c_step = c_step * Bmat

    batch_size, n, _ = Bmat.shape
    C_route = torch.zeros_like(D_raw)

    for b in range(batch_size):
        B_bool = Bmat[b] > 0.5
        D_int = D_raw[b].round().to(torch.int64)
        C_b = torch.zeros_like(D_raw[b])

        for j in range(n):
            f = torch.full((n,), float("inf"), device=D_raw.device, dtype=D_raw.dtype)
            for a in range(n):
                if a == j or bool(B_bool[a, j].item()):
                    f[a] = 0.0

            max_dist = int(D_int[:, j].max().item())
            for dist in range(2, max_dist + 1):
                nodes = [a for a in range(n) if int(D_int[a, j].item()) == dist]
                for a in nodes:
                    candidates = []
                    for nxt in torch.where(B_bool[a])[0].tolist():
                        if int(D_int[nxt, j].item()) == dist - 1:
                            candidates.append(c_step[b, a, nxt] + f[nxt])
                    if candidates:
                        f[a] = torch.stack(candidates).min()

            C_b[:, j] = torch.where(torch.isfinite(f), f, torch.zeros_like(f))
            C_b[j, j] = 0.0

        C_route[b] = 0.5 * (C_b + C_b.transpose(0, 1))

    return C_route


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
    D_raw: torch.Tensor,
    n1q: torch.Tensor,
    nmeas: torch.Tensor,
    e1q: torch.Tensor,
    ero: torch.Tensor,
    e2q: torch.Tensor,
    P_map: torch.Tensor,
    config: Optional[LossConfig] = None,
) -> Dict[str, torch.Tensor]:
    """Compute the v1.4.1 native-frame execution-surrogate loss."""
    if config is None:
        config = LossConfig()

    A = _ensure_batched_square_matrix(A, "A")
    m = _ensure_batched_vector(m, "m")
    Bmat = _ensure_batched_square_matrix(Bmat, "Bmat")
    D_raw = _ensure_batched_square_matrix(D_raw, "D_raw")
    n1q = _ensure_batched_vector(n1q, "n1q")
    nmeas = _ensure_batched_vector(nmeas, "nmeas")
    e1q = _ensure_batched_vector(e1q, "e1q")
    ero = _ensure_batched_vector(ero, "ero")
    e2q = _ensure_batched_square_matrix(e2q, "e2q")
    P_map = _ensure_batched_square_matrix(P_map, "P_map")

    _assert_same_batch_size(A, m, Bmat, D_raw, n1q, nmeas, e1q, ero, e2q, P_map)

    A_off = _offdiag(A)
    mass_1q = _safe_mass(n1q, config.eps)
    mass_meas = _safe_mass(nmeas, config.eps)
    mass_2q = _safe_mass(A_off, config.eps)

    s_1q = (1.0 - e1q).clamp(min=config.eps_surv, max=1.0)
    s_ro = (1.0 - ero).clamp(min=config.eps_surv, max=1.0)
    s_2q = (1.0 - e2q).clamp(min=config.eps_surv, max=1.0)

    S_1q = torch.einsum("buj,bj->bu", P_map, s_1q)
    S_ro = torch.einsum("buj,bj->bu", P_map, s_ro)
    S_2q = _pair_expectation(P_map, Bmat * s_2q)

    C_route = _compute_route_cost_matrix(Bmat=Bmat, D_raw=D_raw, e1q=e1q, e2q=e2q, config=config)
    H_route = _pair_expectation(P_map, C_route)

    L_1q_per_sample = (n1q * _safe_neglog_survival(S_1q, config.eps_surv)).sum(dim=-1) / mass_1q
    L_ro_per_sample = (nmeas * _safe_neglog_survival(S_ro, config.eps_surv)).sum(dim=-1) / mass_meas
    L_2q_per_sample = (A_off * _safe_neglog_survival(S_2q, config.eps_surv)).sum(dim=(-2, -1)) / mass_2q
    L_route_per_sample = (A_off * H_route).sum(dim=(-2, -1)) / mass_2q

    L_native_per_sample = L_1q_per_sample + L_ro_per_sample + L_2q_per_sample
    L_task_per_sample = L_native_per_sample + L_route_per_sample

    L_1q = _batch_mean_scalar(L_1q_per_sample)
    L_ro = _batch_mean_scalar(L_ro_per_sample)
    L_2q = _batch_mean_scalar(L_2q_per_sample)
    L_native = _batch_mean_scalar(L_native_per_sample)
    L_route = _batch_mean_scalar(L_route_per_sample)
    L_task = _batch_mean_scalar(L_task_per_sample)

    S_proxy_1q = torch.exp(-L_1q)
    S_proxy_ro = torch.exp(-L_ro)
    S_proxy_2q = torch.exp(-L_2q)
    S_proxy_route = torch.exp(-L_route)
    S_proxy_exec = torch.exp(-L_task)

    out = {
        "L_1q": L_1q,
        "L_ro": L_ro,
        "L_2q": L_2q,
        "L_native": L_native,
        "L_route": L_route,
        "L_task": L_task,
        "S_proxy_1q": S_proxy_1q,
        "S_proxy_ro": S_proxy_ro,
        "S_proxy_2q": S_proxy_2q,
        "S_proxy_route": S_proxy_route,
        "S_proxy_exec": S_proxy_exec,
        "L_1q_per_sample": L_1q_per_sample,
        "L_ro_per_sample": L_ro_per_sample,
        "L_2q_per_sample": L_2q_per_sample,
        "L_native_per_sample": L_native_per_sample,
        "L_route_per_sample": L_route_per_sample,
        "L_task_per_sample": L_task_per_sample,
        "S_1q": S_1q,
        "S_ro": S_ro,
        "S_2q": S_2q,
        "H_route": H_route,
        "C_route": C_route,
        "mass_1q": mass_1q,
        "mass_meas": mass_meas,
        "mass_2q": mass_2q,
    }
    _validate_required_loss_scalars({k: v for k, v in out.items() if torch.is_tensor(v) and v.ndim == 0})
    return out

def compute_task_loss_from_logits(
    S_star: torch.Tensor,
    R_L: torch.Tensor,
    R_H: torch.Tensor,
    A: torch.Tensor,
    m: torch.Tensor,
    Bmat: torch.Tensor,
    D_raw: torch.Tensor,
    n1q: torch.Tensor,
    nmeas: torch.Tensor,
    e1q: torch.Tensor,
    ero: torch.Tensor,
    e2q: torch.Tensor,
    config: Optional[LossConfig] = None,
) -> Dict[str, torch.Tensor]:
    """Decode latent logits, run Sinkhorn, then compute the v1.4.1 task loss."""
    if config is None:
        config = LossConfig()

    S_nat = decode_to_native(S_star=S_star, R_L=R_L, R_H=R_H)
    P_map = sinkhorn_assignment(S_nat=S_nat, tau_m=config.tau_m, num_iters=config.sinkhorn_iters)
    task = compute_task_loss_from_assignment(
        A=A,
        m=m,
        Bmat=Bmat,
        D_raw=D_raw,
        n1q=n1q,
        nmeas=nmeas,
        e1q=e1q,
        ero=ero,
        e2q=e2q,
        P_map=P_map,
        config=config,
    )
    task["S_nat"] = S_nat
    task["P_map"] = P_map
    _validate_required_loss_scalars({k: v for k, v in task.items() if torch.is_tensor(v) and v.ndim == 0})
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
        A, m, B, c1, c2, D, D_raw, n1q, nmeas, e1q, ero, e2q
    """
    required_model_keys = ("S_star", "R_L", "R_H")
    required_batch_keys = ("A", "m", "B", "c1", "c2", "D", "D_raw", "n1q", "nmeas", "e1q", "ero", "e2q")

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
        D_raw=native_batch["D_raw"],
        n1q=native_batch["n1q"],
        nmeas=native_batch["nmeas"],
        e1q=native_batch["e1q"],
        ero=native_batch["ero"],
        e2q=native_batch["e2q"],
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

    logical_only = bool(getattr(reindexer, 'force_identity_hardware', False))
    if logical_only:
        mse_B_star = _zero_like_scalar(mse_A_star)
        mse_c1_star = _zero_like_scalar(mse_A_star)
        mse_c2_star = _zero_like_scalar(mse_A_star)
        mse_D_star = _zero_like_scalar(mse_A_star)
        L_cons = mse_A_star + mse_m_star
    else:
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
    D_raw: torch.Tensor,
    n1q: torch.Tensor,
    nmeas: torch.Tensor,
    e1q: torch.Tensor,
    ero: torch.Tensor,
    e2q: torch.Tensor,
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
        D_raw=D_raw,
        n1q=n1q,
        nmeas=nmeas,
        e1q=e1q,
        ero=ero,
        e2q=e2q,
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
        "L_1q": out["L_1q"],
        "L_ro": out["L_ro"],
        "L_2q": out["L_2q"],
        "L_native": out["L_native"],
        "L_route": out["L_route"],
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


