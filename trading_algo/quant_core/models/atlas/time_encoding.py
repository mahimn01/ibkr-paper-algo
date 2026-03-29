from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class Time2Vec(nn.Module):
    def __init__(self, d_time: int = 8) -> None:
        super().__init__()
        self.d_time = d_time
        self.omega = nn.Parameter(torch.randn(d_time))
        self.phi = nn.Parameter(torch.randn(d_time))

    def forward(self, timestamps: Tensor) -> Tensor:
        # timestamps: (B, L)
        tau = (timestamps - timestamps.mean(dim=-1, keepdim=True)) / (
            timestamps.std(dim=-1, keepdim=True) + 1e-8
        )
        # tau: (B, L) -> (B, L, 1)
        tau = tau.unsqueeze(-1)
        # omega, phi: (d_time,) broadcast to (B, L, d_time)
        raw = self.omega * tau + self.phi
        # te[0] = linear, te[i>0] = sin(periodic)
        te = torch.cat([raw[..., :1], torch.sin(raw[..., 1:])], dim=-1)
        return te


class CalendarEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dow_embed = nn.Embedding(5, 4)
        self.month_embed = nn.Embedding(12, 4)

    def forward(
        self,
        dow: Tensor,
        month: Tensor,
        is_opex: Tensor,
        is_qtr_end: Tensor,
    ) -> Tensor:
        # dow: (B, L) long [0-4], month: (B, L) long [0-11]
        # is_opex, is_qtr_end: (B, L) float
        d = self.dow_embed(dow)        # (B, L, 4)
        m = self.month_embed(month)    # (B, L, 4)
        binary = torch.stack([is_opex, is_qtr_end], dim=-1)  # (B, L, 2)
        return torch.cat([d, m, binary], dim=-1)  # (B, L, 10)
