from __future__ import annotations

from dataclasses import dataclass

import torch

from kmw1.models.model import active_row_sinkhorn_assignment, decode_canonical_to_native_logits


@dataclass
class LossConfig:
    sinkhorn_tau: float = 0.5
    sinkhorn_iters: int = 30
    eps: float = 1e-8
    eps_surv: float = 1e-12
    route_step_2q_mult: float = 3.0
    route_step_1q_mult: float = 2.0
    lambda_route: float = 1.0


def _assert_finite(x: torch.Tensor, name: str) -> None:
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or Inf.")


def _ensure_batched_square_matrix(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(0)
    if x.ndim == 3:
        return x
    raise ValueError(f"{name} must have shape (N,N) or (B,N,N), got {tuple(x.shape)}")


def _ensure_batched_vector(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.ndim == 1:
        return x.unsqueeze(0)
    if x.ndim == 2:
        return x
    raise ValueError(f"{name} must have shape (N,) or (B,N), got {tuple(x.shape)}")


def _offdiag(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_batched_square_matrix(x, "x")
    n = x.shape[-1]
    eye = torch.eye(n, device=x.device, dtype=x.dtype).unsqueeze(0)
    return x * (1.0 - eye)


def _batch_mean_scalar(x: torch.Tensor) -> torch.Tensor:
    return x if x.ndim == 0 else x.mean()


def _safe_mass(x: torch.Tensor, eps: float) -> torch.Tensor:
    if x.ndim == 2:
        return x.sum(dim=-1).clamp(min=eps)
    if x.ndim == 3:
        return x.sum(dim=(-2, -1)).clamp(min=eps)
    raise ValueError(f"_safe_mass expects rank-2 or rank-3 tensors, got {tuple(x.shape)}")


def _safe_neglog_survival(survival: torch.Tensor, eps_surv: float) -> torch.Tensor:
    return -torch.log(survival.clamp(min=eps_surv, max=1.0))


def _pair_expectation(P_map: torch.Tensor, pair_cost: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bui,bij,bvj->buv", P_map, pair_cost, P_map)


def _compute_route_cost_matrix(*, Bmat: torch.Tensor, D_raw: torch.Tensor, e1q: torch.Tensor, e2q: torch.Tensor, config: LossConfig) -> torch.Tensor:
    Bmat = _ensure_batched_square_matrix(Bmat, "Bmat")
    D_raw = _ensure_batched_square_matrix(D_raw, "D_raw")
    e1q = _ensure_batched_vector(e1q, "e1q")
    e2q = _ensure_batched_square_matrix(e2q, "e2q")
    q1q = _safe_neglog_survival(1.0 - e1q, config.eps_surv)
    q2q = _safe_neglog_survival(1.0 - e2q, config.eps_surv)
    c_step = config.route_step_2q_mult * q2q + 0.5 * config.route_step_1q_mult * (q1q.unsqueeze(-1) + q1q.unsqueeze(-2))
    c_step = c_step * Bmat
    bsz, n, _ = Bmat.shape
    C_route = torch.zeros_like(D_raw)
    for b in range(bsz):
        B_bool = Bmat[b] > 0.5
        D_int = D_raw[b].round().to(torch.int64)
        C_b = torch.zeros_like(D_raw[b])
        for dst in range(n):
            f = torch.full((n,), float("inf"), device=D_raw.device, dtype=D_raw.dtype)
            for src in range(n):
                if src == dst or bool(B_bool[src, dst].item()):
                    f[src] = 0.0
            max_dist = int(D_int[:, dst].max().item())
            for dist in range(2, max_dist + 1):
                nodes = [src for src in range(n) if int(D_int[src, dst].item()) == dist]
                for src in nodes:
                    candidates = []
                    for nxt in torch.where(B_bool[src])[0].tolist():
                        if int(D_int[nxt, dst].item()) == dist - 1:
                            candidates.append(c_step[b, src, nxt] + f[nxt])
                    if candidates:
                        f[src] = torch.stack(candidates).min()
            C_b[:, dst] = torch.where(torch.isfinite(f), f, torch.zeros_like(f))
            C_b[dst, dst] = 0.0
        C_route[b] = 0.5 * (C_b + C_b.transpose(0, 1))
    return C_route


def compute_task_loss_from_assignment(*, A: torch.Tensor, m: torch.Tensor, Bmat: torch.Tensor, D_raw: torch.Tensor,
                                      n1q: torch.Tensor, nmeas: torch.Tensor, e1q: torch.Tensor, ero: torch.Tensor,
                                      e2q: torch.Tensor, P_map: torch.Tensor, config: LossConfig | None = None) -> dict[str, torch.Tensor]:
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
    active = m
    A_off = _offdiag(A) * active.unsqueeze(-1) * active.unsqueeze(-2)
    n1q_eff = n1q * active
    nmeas_eff = nmeas * active
    mass_1q = _safe_mass(n1q_eff, config.eps)
    mass_meas = _safe_mass(nmeas_eff, config.eps)
    mass_2q = _safe_mass(A_off, config.eps)
    s_1q = (1.0 - e1q).clamp(min=config.eps_surv, max=1.0)
    s_ro = (1.0 - ero).clamp(min=config.eps_surv, max=1.0)
    s_2q = (1.0 - e2q).clamp(min=config.eps_surv, max=1.0)
    S_1q = torch.einsum("buj,bj->bu", P_map, s_1q)
    S_ro = torch.einsum("buj,bj->bu", P_map, s_ro)
    S_2q = _pair_expectation(P_map, Bmat * s_2q)
    C_route = _compute_route_cost_matrix(Bmat=Bmat, D_raw=D_raw, e1q=e1q, e2q=e2q, config=config)
    H_route = _pair_expectation(P_map, C_route)
    L_1q_per_sample = (n1q_eff * _safe_neglog_survival(S_1q, config.eps_surv)).sum(dim=-1) / mass_1q
    L_ro_per_sample = (nmeas_eff * _safe_neglog_survival(S_ro, config.eps_surv)).sum(dim=-1) / mass_meas
    L_2q_per_sample = (A_off * _safe_neglog_survival(S_2q, config.eps_surv)).sum(dim=(-2, -1)) / mass_2q
    L_route_per_sample = (A_off * H_route).sum(dim=(-2, -1)) / mass_2q
    L_native_per_sample = L_1q_per_sample + L_ro_per_sample + L_2q_per_sample
    L_task_per_sample = L_native_per_sample + config.lambda_route * L_route_per_sample
    out = {
        "L_1Q": _batch_mean_scalar(L_1q_per_sample),
        "L_RO": _batch_mean_scalar(L_ro_per_sample),
        "L_2Q": _batch_mean_scalar(L_2q_per_sample),
        "L_native": _batch_mean_scalar(L_native_per_sample),
        "L_route": _batch_mean_scalar(L_route_per_sample),
        "L_task": _batch_mean_scalar(L_task_per_sample),
        "route_to_native_ratio": _batch_mean_scalar(L_route_per_sample) / _batch_mean_scalar(L_native_per_sample).clamp(min=config.eps),
        "S_proxy_1q": torch.exp(-_batch_mean_scalar(L_1q_per_sample)),
        "S_proxy_ro": torch.exp(-_batch_mean_scalar(L_ro_per_sample)),
        "S_proxy_2q": torch.exp(-_batch_mean_scalar(L_2q_per_sample)),
        "S_proxy_route": torch.exp(-_batch_mean_scalar(L_route_per_sample)),
        "S_proxy_exec": torch.exp(-_batch_mean_scalar(L_task_per_sample)),
        "P_map": P_map,
    }
    # legacy aliases
    out["L_1q"] = out["L_1Q"]
    out["L_ro"] = out["L_RO"]
    out["L_2q"] = out["L_2Q"]
    for key, value in out.items():
        if torch.is_tensor(value) and value.ndim == 0:
            _assert_finite(value, key)
    return out


def compute_task_loss_from_logits(*, S_can: torch.Tensor, p: torch.Tensor, A: torch.Tensor, m: torch.Tensor,
                                  B_nat: torch.Tensor, D_raw_nat: torch.Tensor, n1q: torch.Tensor,
                                  nmeas: torch.Tensor, e1q_nat: torch.Tensor, ero_nat: torch.Tensor,
                                  e2q_nat: torch.Tensor, config: LossConfig | None = None) -> dict[str, torch.Tensor]:
    if config is None:
        config = LossConfig()
    S_nat = decode_canonical_to_native_logits(S_can=S_can, p=p)
    P_map = active_row_sinkhorn_assignment(S_logits=S_nat, m=m, tau=config.sinkhorn_tau, num_iters=config.sinkhorn_iters)
    out = compute_task_loss_from_assignment(
        A=A,
        m=m,
        Bmat=B_nat,
        D_raw=D_raw_nat,
        n1q=n1q,
        nmeas=nmeas,
        e1q=e1q_nat,
        ero=ero_nat,
        e2q=e2q_nat,
        P_map=P_map,
        config=config,
    )
    out["S_nat"] = S_nat
    out["S_can"] = S_can[:, 0] if S_can.ndim == 4 else S_can
    return out
