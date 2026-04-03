from __future__ import annotations

import torch
import torch.nn as nn


class MappingProxyLoss(nn.Module):
    """Exact v1.1 proxy-loss formulas."""

    def __init__(self, lambda_p: float = 1.0, lambda_s: float = 0.1, lambda_d: float = 0.1, kappa: float = 1.0):
        super().__init__()
        self.lp = lambda_p
        self.ls = lambda_s
        self.ld = lambda_d
        self.kappa = kappa

    def forward(self, P: torch.Tensor, W: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor, D: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        loss_1q = torch.sum(P * c1.unsqueeze(1), dim=(1, 2))
        loss_2q = torch.sum(torch.bmm(torch.bmm(P, c2), P.transpose(1, 2)) * W, dim=(1, 2))
        l_pst = loss_1q + loss_2q

        logical_dist_eff = torch.bmm(torch.bmm(P, D), P.transpose(1, 2))
        l_swap = torch.sum(logical_dist_eff * W, dim=(1, 2))
        l_depth = self.kappa * l_swap

        total_loss = (
            self.lp * l_pst.mean()
            + self.ls * l_swap.mean()
            + self.ld * l_depth.mean()
        )
        return total_loss
