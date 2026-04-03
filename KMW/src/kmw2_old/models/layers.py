from __future__ import annotations

import torch
import torch.nn as nn


class SharedTokenEncoder(nn.Module):
    """Exact v1.1 shared token encoder."""

    def __init__(self, in_dim: int, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class CrossAttentionInjector(nn.Module):
    """Stores num_heads but implements the same effective single-head behavior as v1.1."""

    def __init__(self, fmap_channels: int, token_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.da = token_dim
        self.q_proj = nn.Linear(fmap_channels, self.da)
        self.k_proj = nn.Linear(token_dim, self.da)
        self.v_proj = nn.Linear(token_dim, self.da)
        self.o_proj = nn.Linear(self.da, fmap_channels)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, F_map: torch.Tensor, T_tokens: torch.Tensor) -> torch.Tensor:
        B, C, H, W = F_map.shape
        F_flat = F_map.view(B, C, -1).permute(0, 2, 1)
        Q = self.q_proj(F_flat)
        K = self.k_proj(T_tokens)
        V = self.v_proj(T_tokens)
        scaling = self.da ** -0.5
        attn_weights = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) * scaling, dim=-1)
        out = torch.matmul(attn_weights, V)
        delta_F = self.o_proj(out).permute(0, 2, 1).view(B, C, H, W)
        return F_map + self.alpha * delta_F
