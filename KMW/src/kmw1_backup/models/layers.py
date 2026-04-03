from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class LogSinkhorn(nn.Module):
    def __init__(self, num_iters: int = 30) -> None:
        super().__init__()
        self.num_iters = num_iters

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 3:
            raise ValueError(f"logits must have shape (B,N,N), got {tuple(logits.shape)}")
        logP = logits
        for _ in range(self.num_iters):
            logP = logP - torch.logsumexp(logP, dim=-1, keepdim=True)
            logP = logP - torch.logsumexp(logP, dim=-2, keepdim=True)
        P = logP.exp()
        _assert_finite(P, "Sinkhorn output")
        return P


class HardwareTokenEncoder(nn.Module):
    """
    Token rule:
        x_hw[j] = concat(B_can[j,:], c2_can[j,:], c1_can[j])
    """
    def __init__(self, n: int = 27, token_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.n = n
        self.token_dim = token_dim
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
    def __init__(
        self,
        in_channels: int,
        token_dim: int = 128,
        attn_dim: int = 128,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if attn_dim % num_heads != 0:
            raise ValueError("attn_dim must be divisible by num_heads")
        self.in_channels = in_channels
        self.token_dim = token_dim
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

        Bsz, C, H, W = fmap.shape
        seq = fmap.flatten(start_dim=2).transpose(1, 2)  # (B, HW, C)

        Q = self.q_proj(seq)
        K = self.k_proj(tokens)
        V = self.v_proj(tokens)

        Q = Q.view(Bsz, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(Bsz, tokens.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(Bsz, tokens.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(Bsz, H * W, self.attn_dim)
        delta = self.out_proj(out).transpose(1, 2).reshape(Bsz, C, H, W)
        result = fmap + self.alpha_attn * delta
        _assert_finite(result, "cross_attention_result")
        return result


class AttnUNet27(nn.Module):
    """
    Locked shallow conditional U-Net:
    27 -> 14 -> 7 -> 14 -> 27
    """
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
    def _resize(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
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
