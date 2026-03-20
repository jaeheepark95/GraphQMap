"""Bidirectional Cross-Attention module for GraphQMap.

Enables circuit and hardware embeddings to mutually reference each other.
Structure (repeated N times):
  C = LayerNorm(C + MultiHeadCrossAttention(Q=C, K=H, V=H))
  C = LayerNorm(C + FFN(C))
  H = LayerNorm(H + MultiHeadCrossAttention(Q=H, K=C, V=C))
  H = LayerNorm(H + FFN(H))
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Position-wise feed-forward network.

    Args:
        d_model: Input/output dimension.
        d_ff: Hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionLayer(nn.Module):
    """Single bidirectional cross-attention layer.

    Args:
        d_model: Embedding dimension.
        num_heads: Number of attention heads.
        d_ff: FFN hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Circuit attends to Hardware
        self.cross_attn_c2h = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.norm_c1 = nn.LayerNorm(d_model)
        self.ffn_c = FeedForward(d_model, d_ff, dropout)
        self.norm_c2 = nn.LayerNorm(d_model)

        # Hardware attends to Circuit
        self.cross_attn_h2c = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.norm_h1 = nn.LayerNorm(d_model)
        self.ffn_h = FeedForward(d_model, d_ff, dropout)
        self.norm_h2 = nn.LayerNorm(d_model)

    def forward(
        self,
        C: torch.Tensor,
        H: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            C: Circuit embeddings (batch, l, d) or (l, d).
            H: Hardware embeddings (batch, h, d) or (h, d).

        Returns:
            Updated (C, H) tuple.
        """
        # Circuit attends to Hardware: Q=C, K=H, V=H
        attn_out_c, _ = self.cross_attn_c2h(query=C, key=H, value=H)
        C = self.norm_c1(C + attn_out_c)
        C = self.norm_c2(C + self.ffn_c(C))

        # Hardware attends to Circuit: Q=H, K=C, V=C
        attn_out_h, _ = self.cross_attn_h2c(query=H, key=C, value=C)
        H = self.norm_h1(H + attn_out_h)
        H = self.norm_h2(H + self.ffn_h(H))

        return C, H


class CrossAttentionModule(nn.Module):
    """Stacked bidirectional cross-attention layers.

    Args:
        d_model: Embedding dimension.
        num_layers: Number of cross-attention layers.
        num_heads: Number of attention heads.
        d_ff: FFN hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        d_ff: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        C: torch.Tensor,
        H: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through all cross-attention layers.

        Args:
            C: Circuit embeddings (batch, l, d).
            H: Hardware embeddings (batch, h, d).

        Returns:
            Updated (C', H') tuple after all layers.
        """
        for layer in self.layers:
            C, H = layer(C, H)
        return C, H
