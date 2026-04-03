from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


def _assert_rank(x: torch.Tensor, expected_rank: int, name: str) -> None:
    if x.ndim != expected_rank:
        raise ValueError(f"{name} must have rank {expected_rank}, got shape {tuple(x.shape)}")


def _assert_finite(x: torch.Tensor, name: str) -> None:
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or Inf.")


def build_mlp(in_dim: int, hidden_dim: int, out_dim: int | None = None, dropout: float = 0.1) -> nn.Sequential:
    out_dim = hidden_dim if out_dim is None else out_dim
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class HardwareTokenEncoder(nn.Module):
    def __init__(self, n: int = 27, token_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = build_mlp(in_dim=2 * n + 1, hidden_dim=token_dim, out_dim=token_dim, dropout=dropout)

    def forward(self, B_can: torch.Tensor, c2_can: torch.Tensor, c1_can: torch.Tensor) -> torch.Tensor:
        _assert_rank(B_can, 3, "B_can")
        _assert_rank(c2_can, 3, "c2_can")
        _assert_rank(c1_can, 2, "c1_can")
        x = torch.cat([B_can, c2_can, c1_can.unsqueeze(-1)], dim=-1)
        tokens = self.mlp(x)
        _assert_finite(tokens, "hardware_tokens")
        return tokens


class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channels: int, token_dim: int = 128, attn_dim: int = 128, num_heads: int = 4) -> None:
        super().__init__()
        if attn_dim % num_heads != 0:
            raise ValueError("attn_dim must be divisible by num_heads")
        self.in_channels = in_channels
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.q_proj = nn.Linear(in_channels, attn_dim)
        self.k_proj = nn.Linear(token_dim, attn_dim)
        self.v_proj = nn.Linear(token_dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, in_channels)
        self.alpha_attn = nn.Parameter(torch.tensor(0.0))

    def forward(self, fmap: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        _assert_rank(fmap, 4, "fmap")
        _assert_rank(tokens, 3, "tokens")
        bsz, c, h, w = fmap.shape
        seq = fmap.flatten(start_dim=2).transpose(1, 2)
        q = self.q_proj(seq)
        k = self.k_proj(tokens)
        v = self.v_proj(tokens)
        q = q.view(bsz, h * w, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, tokens.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, tokens.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(bsz, h * w, self.attn_dim)
        delta = self.out_proj(out).transpose(1, 2).reshape(bsz, c, h, w)
        result = fmap + self.alpha_attn * delta
        _assert_finite(result, "cross_attention_result")
        return result


class AttnUNet27(nn.Module):
    def __init__(self, token_dim: int = 128, num_heads: int = 4) -> None:
        super().__init__()
        self.conv_down1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv_down2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.attn_down2 = CrossAttentionBlock(128, token_dim=token_dim, attn_dim=128, num_heads=num_heads)
        self.conv_bottleneck = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.attn_bottleneck = CrossAttentionBlock(256, token_dim=token_dim, attn_dim=128, num_heads=num_heads)
        self.conv_up1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.attn_up1 = CrossAttentionBlock(128, token_dim=token_dim, attn_dim=128, num_heads=num_heads)
        self.conv_head = nn.Conv2d(128, 1, kernel_size=3, padding=1)

    @staticmethod
    def _resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, A_spatial: torch.Tensor, T_hw: torch.Tensor) -> torch.Tensor:
        _assert_rank(A_spatial, 4, "A_spatial")
        d1 = F.relu(self.conv_down1(A_spatial))
        d2 = F.relu(self.conv_down2(self._resize(d1, (14, 14))))
        d2 = self.attn_down2(d2, T_hw)
        bottleneck = F.relu(self.conv_bottleneck(self._resize(d2, (7, 7))))
        bottleneck = self.attn_bottleneck(bottleneck, T_hw)
        up = F.relu(self.conv_up1(self._resize(bottleneck, (14, 14))))
        up = self.attn_up1(up + d2, T_hw)
        S = self.conv_head(self._resize(up, (27, 27)))
        _assert_finite(S, "S_can")
        return S


def decode_canonical_to_native_logits(S_can: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    if S_can.ndim == 4:
        if S_can.shape[1] != 1:
            raise ValueError("S_can with rank 4 must have channel dimension = 1")
        S_can = S_can[:, 0]
    if S_can.ndim != 3 or p.ndim != 2:
        raise ValueError("Expected S_can=(B,N,N), p=(B,N)")
    out = torch.zeros_like(S_can)
    out.scatter_(dim=-1, index=p.unsqueeze(1).expand(-1, S_can.shape[1], -1), src=S_can)
    _assert_finite(out, "S_nat")
    return out


def _active_count_from_mask(m: torch.Tensor) -> int:
    return int((m > 0.5).sum().item())


def _rectangular_log_sinkhorn_single(logits: torch.Tensor, target_col_sum: float, num_iters: int = 30, eps: float = 1e-12) -> torch.Tensor:
    logp = logits.clone()
    for _ in range(num_iters):
        logp = logp - torch.logsumexp(logp, dim=-1, keepdim=True)
        col = torch.exp(logp).sum(dim=-2, keepdim=True).clamp(min=eps)
        logp = logp - torch.log(col / target_col_sum)
    return torch.exp(logp)


def active_row_sinkhorn_assignment(S_logits: torch.Tensor, m: torch.Tensor, tau: float = 0.5, num_iters: int = 30) -> torch.Tensor:
    if S_logits.ndim != 3 or m.ndim != 2:
        raise ValueError("Expected S_logits=(B,N,N) and m=(B,N)")
    bsz, n, _ = S_logits.shape
    out = torch.zeros_like(S_logits)
    for b in range(bsz):
        k = _active_count_from_mask(m[b])
        if k <= 0:
            continue
        active_logits = S_logits[b, :k, :] / tau
        target_col_sum = float(k) / float(n)
        P_act = _rectangular_log_sinkhorn_single(active_logits, target_col_sum=target_col_sum, num_iters=num_iters)
        out[b, :k, :] = P_act
        row_sums = P_act.sum(dim=-1)
        col_sums = P_act.sum(dim=-2)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2, rtol=1e-2):
            raise ValueError("Active-row Sinkhorn row sums are not approximately 1.")
        target_cols = torch.full_like(col_sums, target_col_sum)
        if not torch.allclose(col_sums, target_cols, atol=5e-2, rtol=5e-2):
            raise ValueError("Active-row Sinkhorn column sums are not approximately K/N.")
    _assert_finite(out, "P_nat_act")
    return out


@dataclass
class EvalAssignmentResult:
    M_nat: torch.Tensor
    active_mapping: list[int]


def active_row_hungarian(S_nat: torch.Tensor, m: torch.Tensor) -> EvalAssignmentResult:
    if linear_sum_assignment is None:
        raise ImportError("scipy is required for Hungarian assignment during evaluation.")
    if S_nat.ndim != 2 or m.ndim != 1:
        raise ValueError("Expected S_nat=(N,N), m=(N,)")
    k = _active_count_from_mask(m)
    scores = S_nat[:k].detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(-scores)
    M = torch.zeros_like(S_nat)
    M[row_ind, col_ind] = 1.0
    mapping = [int(torch.argmax(M[u]).item()) for u in range(k)]
    return EvalAssignmentResult(M_nat=M, active_mapping=mapping)


class KMWCanonicalModel(nn.Module):
    def __init__(self, n: int = 27, token_dim: int = 128, sinkhorn_iters: int = 30, dropout: float = 0.1) -> None:
        super().__init__()
        self.n = n
        self.token_encoder = HardwareTokenEncoder(n=n, token_dim=token_dim, dropout=dropout)
        self.mapper = AttnUNet27(token_dim=token_dim, num_heads=4)
        self.sinkhorn_iters = sinkhorn_iters

    def forward(self, *, A: torch.Tensor, B_can: torch.Tensor, c1_can: torch.Tensor, c2_can: torch.Tensor) -> dict[str, torch.Tensor]:
        if A.ndim != 3:
            raise ValueError(f"A must have shape (B,N,N), got {tuple(A.shape)}")
        T_hw = self.token_encoder(B_can=B_can, c2_can=c2_can, c1_can=c1_can)
        S_can = self.mapper(A.unsqueeze(1), T_hw)
        return {"T_hw_can": T_hw, "S_can": S_can}

    @staticmethod
    def decode_to_native(S_can: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return decode_canonical_to_native_logits(S_can=S_can, p=p)
