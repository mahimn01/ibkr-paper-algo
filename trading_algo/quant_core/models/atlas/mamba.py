from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SelectiveSSM(nn.Module):
    def __init__(self, d_inner: int, d_state: int, dt_rank: int) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank

        # A_log: log of negative A matrix (learned in log space for stability)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .unsqueeze(0)
            .expand(d_inner, -1)
            .clone()
        )

        self.D = nn.Parameter(torch.ones(d_inner))

        # Projections for B, C, dt from input
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner)

    def forward(self, u: Tensor) -> Tensor:
        # u: (B, L, D_inner)
        B, L, D = u.shape

        # Project input to get dt, B_input, C
        x_dbl = self.x_proj(u)  # (B, L, dt_rank + 2*d_state)
        dt_raw, B_input, C = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # dt: (B, L, dt_rank) -> (B, L, D_inner)
        dt = F.softplus(self.dt_proj(dt_raw))  # (B, L, D_inner)

        # A: (D_inner, d_state) — always negative
        A = -torch.exp(self.A_log)

        # Sequential scan (MPS-compatible, no parallel scan)
        h = torch.zeros(B, D, self.d_state, device=u.device, dtype=u.dtype)
        ys = []

        for t in range(L):
            dt_t = dt[:, t]                # (B, D_inner)
            B_t = B_input[:, t]            # (B, d_state)
            C_t = C[:, t]                  # (B, d_state)
            u_t = u[:, t]                  # (B, D_inner)

            # Discretize
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A)    # (B, D_inner, d_state)
            B_bar = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # (B, D_inner, d_state)

            # State update
            h = A_bar * h + B_bar * u_t.unsqueeze(-1)

            # Output
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)  # (B, D_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, L, D_inner)

        # D skip connection
        y = y + u * self.D
        return y


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        d_inner = d_model * expand_factor
        dt_rank = math.ceil(d_model / 16)

        # Input projection: splits into x and z branches
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # Depthwise conv on x branch
        self.conv1d = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
        )

        self.ssm = SelectiveSSM(d_inner, d_state, dt_rank)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, d_model)
        residual = x
        x = self.norm(x)

        # Split into two branches
        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        x_branch, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Conv1d on x branch: (B, L, D) -> (B, D, L) -> conv -> (B, D, L) -> (B, L, D)
        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[..., :residual.shape[1]]  # trim to original length
        x_branch = x_branch.transpose(1, 2)
        x_branch = F.silu(x_branch)

        # SSM
        y = self.ssm(x_branch)  # (B, L, d_inner)

        # Gate with z branch
        y = y * F.silu(z)

        # Project back and residual
        out = self.out_proj(y)  # (B, L, d_model)
        return residual + out


class MambaBackbone(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand_factor)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, d_model)
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)
