from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .layers import CrossAttentionInjector, SharedTokenEncoder


class UNetMapping(nn.Module):
    """Exact v1.1 shallow U-Net with three token-injection sites."""

    def __init__(self, in_channels: int = 5, token_dim: int = 128):
        super().__init__()
        self.logical_encoder = SharedTokenEncoder(in_dim=3, embed_dim=token_dim)
        self.physical_encoder = SharedTokenEncoder(in_dim=4, embed_dim=token_dim)
        self.type_embed_log = nn.Parameter(torch.randn(1, 1, token_dim))
        self.type_embed_phy = nn.Parameter(torch.randn(1, 1, token_dim))

        self.down1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.down2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.attn_down = CrossAttentionInjector(128, token_dim)

        self.bottleneck = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.attn_bottleneck = CrossAttentionInjector(256, token_dim)

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.attn_up = CrossAttentionInjector(128, token_dim)
        self.final = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, X: torch.Tensor, Tlog_raw: torch.Tensor, Tphy_raw: torch.Tensor) -> torch.Tensor:
        Tlog = self.logical_encoder(Tlog_raw) + self.type_embed_log
        Tphy = self.physical_encoder(Tphy_raw) + self.type_embed_phy
        T = torch.cat([Tlog, Tphy], dim=1)

        d1 = torch.relu(self.down1(X))
        d2 = torch.relu(self.down2(d1))
        d2 = self.attn_down(d2, T)

        bn = torch.relu(self.bottleneck(d2))
        bn = self.attn_bottleneck(bn, T)

        u1 = torch.relu(self.up1(bn))
        u1 = F.interpolate(u1, size=d2.shape[2:])
        u1 = self.attn_up(u1, T)

        S = self.final(u1 + d2)
        S = F.interpolate(S, size=(27, 27))
        return S.squeeze(1)


class AssignmentHead:
    @staticmethod
    def sinkhorn(S: torch.Tensor, tau: float = 0.5, iterations: int = 20) -> torch.Tensor:
        S_norm = (S - torch.max(S, dim=-1, keepdim=True).values) / tau
        P = torch.exp(S_norm)
        for _ in range(iterations):
            P = P / (P.sum(dim=-1, keepdim=True) + 1e-12)
            P = P / (P.sum(dim=-2, keepdim=True) + 1e-12)
        return P

    @staticmethod
    def hungarian(S_batch: torch.Tensor) -> torch.Tensor:
        squeeze_back = False
        if S_batch.ndim == 2:
            S_batch = S_batch.unsqueeze(0)
            squeeze_back = True
        S_np = S_batch.detach().cpu().numpy()
        S_np = np.nan_to_num(S_np, nan=-1e6, posinf=1e6, neginf=-1e6)
        S_np += 1e-9 * np.random.randn(*S_np.shape)
        M = torch.zeros_like(S_batch)
        for b in range(S_np.shape[0]):
            row_ind, col_ind = linear_sum_assignment(S_np[b], maximize=True)
            M[b, row_ind, col_ind] = 1.0
        return M.squeeze(0) if squeeze_back else M
