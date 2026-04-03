from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from kmw1.utils import batched_index_select_cols, batched_index_select_rows


@dataclass(slots=True)
class CanonicalHardwareIndex:
    p: torch.Tensor       # canonical slot -> native id
    p_inv: torch.Tensor   # native id -> canonical slot
    qscore: torch.Tensor
    degree: torch.Tensor
    mean_edge_cost: torch.Tensor

    def to_serializable(self) -> dict[str, Any]:
        return {
            "p": self.p,
            "p_inv": self.p_inv,
            "qscore": self.qscore,
            "degree": self.degree,
            "mean_edge_cost": self.mean_edge_cost,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CanonicalHardwareIndex":
        return cls(**payload)


def _zscore(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = x.mean()
    std = x.std(unbiased=False)
    return (x - mean) / max(float(std.item()), eps)


def _canonicalize_matrix(mat: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return mat[p][:, p]


def _canonicalize_vector(vec: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return vec[p]


def build_canonical_hardware_index(
    B_nat: torch.Tensor,
    c1_nat: torch.Tensor,
    c2_nat: torch.Tensor,
    isolated_fallback: float | None = None,
) -> CanonicalHardwareIndex:
    """
    Deterministic canonicalizer:
    - qscore(i) = z(c1[i]) + z(meanEdgeCost[i]) - 0.3 z(deg[i])
    - repeated BFS
    - neighbors sorted by:
        1) lower edge cost
        2) lower c1
        3) higher degree
        4) lower native id
    """
    if tuple(B_nat.shape) != (27, 27):
        raise ValueError(f"B_nat must have shape (27,27), got {tuple(B_nat.shape)}")
    if tuple(c1_nat.shape) != (27,):
        raise ValueError(f"c1_nat must have shape (27,), got {tuple(c1_nat.shape)}")
    if tuple(c2_nat.shape) != (27, 27):
        raise ValueError(f"c2_nat must have shape (27,27), got {tuple(c2_nat.shape)}")

    adj = ((B_nat + B_nat.transpose(0, 1)) > 0).float()
    adj.fill_diagonal_(0.0)
    degree = adj.sum(dim=-1)

    if isolated_fallback is None:
        positive_edges = c2_nat[adj > 0.5]
        isolated_fallback = float(positive_edges.mean().item()) if positive_edges.numel() > 0 else 1.0

    mean_edge_cost = torch.full((27,), float(isolated_fallback), dtype=torch.float32)
    for i in range(27):
        nbrs = torch.where(adj[i] > 0.5)[0]
        if nbrs.numel() == 0:
            continue
        vals = torch.stack([torch.minimum(c2_nat[i, j], c2_nat[j, i]) for j in nbrs])
        mean_edge_cost[i] = vals.mean()

    qscore = _zscore(c1_nat) + _zscore(mean_edge_cost) - 0.3 * _zscore(degree)

    visited: set[int] = set()
    order: list[int] = []

    def pick_root() -> int:
        remaining = [i for i in range(27) if i not in visited]
        remaining.sort(key=lambda i: (float(qscore[i].item()), int(i)))
        return int(remaining[0])

    while len(visited) < 27:
        root = pick_root()
        queue = [root]
        if root not in visited:
            visited.add(root)
            order.append(root)

        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            nbrs = [int(v) for v in torch.where(adj[u] > 0.5)[0].tolist() if int(v) not in visited]
            nbrs.sort(
                key=lambda v: (
                    float(torch.minimum(c2_nat[u, v], c2_nat[v, u]).item()),
                    float(c1_nat[v].item()),
                    -float(degree[v].item()),
                    int(v),
                )
            )
            for v in nbrs:
                if v in visited:
                    continue
                visited.add(v)
                order.append(v)
                queue.append(v)

    if len(order) != 27 or len(set(order)) != 27:
        raise ValueError(f"Invalid canonical order produced: {order}")

    p = torch.tensor(order, dtype=torch.long)
    p_inv = torch.empty_like(p)
    p_inv[p] = torch.arange(27, dtype=torch.long)

    if not torch.equal(p_inv[p], torch.arange(27, dtype=torch.long)):
        raise ValueError("Invalid canonical inverse permutation.")

    return CanonicalHardwareIndex(
        p=p,
        p_inv=p_inv,
        qscore=qscore,
        degree=degree,
        mean_edge_cost=mean_edge_cost,
    )


def canonicalize_hardware_tensors(
    *,
    p: torch.Tensor,
    B: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    D: torch.Tensor,
    B_raw: torch.Tensor | None = None,
    c1_raw: torch.Tensor | None = None,
    c2_raw: torch.Tensor | None = None,
    D_raw: torch.Tensor | None = None,
    e1q_raw: torch.Tensor | None = None,
    ero_raw: torch.Tensor | None = None,
    e2q_raw: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    out = {
        "B_can": _canonicalize_matrix(B, p),
        "c1_can": _canonicalize_vector(c1, p),
        "c2_can": _canonicalize_matrix(c2, p),
        "D_can": _canonicalize_matrix(D, p),
    }
    if B_raw is not None:
        out["B_raw_can"] = _canonicalize_matrix(B_raw, p)
    if c1_raw is not None:
        out["c1_raw_can"] = _canonicalize_vector(c1_raw, p)
    if c2_raw is not None:
        out["c2_raw_can"] = _canonicalize_matrix(c2_raw, p)
    if D_raw is not None:
        out["D_raw_can"] = _canonicalize_matrix(D_raw, p)
    if e1q_raw is not None:
        out["e1q_can"] = _canonicalize_vector(e1q_raw, p)
    if ero_raw is not None:
        out["ero_can"] = _canonicalize_vector(ero_raw, p)
    if e2q_raw is not None:
        out["e2q_can"] = _canonicalize_matrix(e2q_raw, p)
    return out


def batched_decode_canonical_to_native_logits(S_can: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    S_can: (B,N,N) or (B,1,N,N)
    p    : (B,N), canonical slot -> native id
    Returns S_nat where column j in canonical frame is moved to native column p[j].
    """
    if S_can.ndim == 4:
        if S_can.shape[1] != 1:
            raise ValueError("S_can with rank 4 must have a singleton channel.")
        S_can = S_can[:, 0]
    if S_can.ndim != 3 or p.ndim != 2:
        raise ValueError("Expected S_can=(B,N,N) and p=(B,N).")
    Bsz, N, N2 = S_can.shape
    if N != N2:
        raise ValueError("S_can must be square.")
    out = torch.zeros_like(S_can)
    out.scatter_(dim=-1, index=p.unsqueeze(1).expand(-1, N, -1), src=S_can)
    return out


def batched_canonical_to_native_assignment(M_can: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    if M_can.ndim != 3 or p.ndim != 2:
        raise ValueError("Expected M_can=(B,N,N) and p=(B,N)")
    out = torch.zeros_like(M_can)
    out.scatter_(dim=-1, index=p.unsqueeze(1).expand(-1, M_can.shape[1], -1), src=M_can)
    return out
