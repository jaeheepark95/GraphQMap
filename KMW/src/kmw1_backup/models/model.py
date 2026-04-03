from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from kmw1.models.layers import AttnUNet27, HardwareTokenEncoder, LogSinkhorn


def decode_canonical_to_native_logits(S_can: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    Move canonical hardware columns back to native hardware IDs.

    S_can: (B,1,N,N) or (B,N,N)
    p    : (B,N), canonical slot -> native id
    """
    if S_can.ndim == 4:
        if S_can.shape[1] != 1:
            raise ValueError("S_can with rank 4 must have channel dimension = 1")
        S_can = S_can[:, 0]
    if S_can.ndim != 3 or p.ndim != 2:
        raise ValueError("Expected S_can=(B,N,N), p=(B,N)")
    out = torch.zeros_like(S_can)
    out.scatter_(dim=-1, index=p.unsqueeze(1).expand(-1, S_can.shape[1], -1), src=S_can)
    if not torch.isfinite(out).all():
        raise ValueError("Decoded native logits contain NaN or Inf.")
    return out


def sinkhorn_assignment(S_logits: torch.Tensor, tau: float = 0.5, num_iters: int = 30) -> torch.Tensor:
    sinkhorn = LogSinkhorn(num_iters=num_iters)
    return sinkhorn(S_logits / tau)


class KMWCanonicalModel(nn.Module):
    """
    Canonical-hardware v1.4 mapper.
    - logical order: identity
    - hardware order: deterministic canonical preprocessing order
    - no learned reindexer
    """
    def __init__(
        self,
        n: int = 27,
        token_dim: int = 128,
        sinkhorn_iters: int = 30,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n = n
        self.token_encoder = HardwareTokenEncoder(n=n, token_dim=token_dim, dropout=dropout)
        self.mapper = AttnUNet27(token_dim=token_dim, num_heads=4)
        self.sinkhorn_iters = sinkhorn_iters

    def forward(
        self,
        *,
        A: torch.Tensor,
        B_can: torch.Tensor,
        c1_can: torch.Tensor,
        c2_can: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if A.ndim != 3:
            raise ValueError(f"A must have shape (B,N,N), got {tuple(A.shape)}")
        T_hw = self.token_encoder(B_can=B_can, c2_can=c2_can, c1_can=c1_can)
        S_can = self.mapper(A.unsqueeze(1), T_hw)
        return {
            "T_hw_can": T_hw,
            "S_can": S_can,
        }

    @staticmethod
    def decode_to_native(S_can: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return decode_canonical_to_native_logits(S_can=S_can, p=p)

    @staticmethod
    def training_assignment(S_nat: torch.Tensor, tau: float = 0.5, num_iters: int = 30) -> torch.Tensor:
        return sinkhorn_assignment(S_logits=S_nat, tau=tau, num_iters=num_iters)
