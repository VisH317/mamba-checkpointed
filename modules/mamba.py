import torch
from torch import nn, Tensor
import torch.nn.functional as F
import triton
from modules.mamba_utils import generate_a, RMSNorm
    

class S4(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, dt_rank: int):
        super().__init__()

        self.d_in = d_in
        self.d_hidden = d_hidden
        self.dt_rank = dt_rank

        grid = lambda meta: (triton.cdiv(d_hidden * d_hidden, meta["BLOCK_SIZE"]))
        self.A = torch.empty(d_in, d_hidden)
        generate_a[grid](self.A, d_hidden)

        self.D = nn.Parameter(torch.ones(d_in, d_in))

        self.x_proj = nn.Linear(d_in, 2 * d_hidden + dt_rank, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_in, bias=True)

    def forward(self, x: Tensor) -> Tensor:

        (b, l, d_in) = x.size()

        bcd = self.x_proj(x)
        (B, C, dt) = bcd.split(split_size=[self.dt_rank, self.d_hidden, self.d_hidden], dim=-1)
        delta = F.softplus(self.dt_proj(dt))

        # discretization
        Ad = torch.exp(torch.einsum(self.A, delta, "b l d_in, d_in, n -> b l d_in n"))
        Bx = torch.einsum(delta, B, x, 'b l d_in, b l n, b l d_in -> b l d_in n')

        h = torch.zeros(b, self.d_in, self.d_hidden)
        ys = []
        for i in range(l):
            h = Ad[:, i] * h + Bx[:, i]
            y = torch.einsum(x, C[:, i, :], "b d_in n, b n -> b d_in")
            ys.append(y)

        y = y + x * self.D

        return y


class MambaBlock(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_ssm_hidden: int, dt_rank: int, n_conv: int):
        super().__init__()

        self.conv = nn.Conv1d(d_in, d_in, kernel_size=n_conv, padding=n_conv-1)
        self.x_proj = nn.Linear(d_in, d_hidden)
        self.norm = RMSNorm()

        self.x_res_proj = nn.Linear(d_hidden, d_hidden, bias=False)
        self.out_proj = nn.Linear(d_hidden, d_in)

        self.ssm = S4(d_hidden, d_ssm_hidden, dt_rank)

    def forward(self, x: Tensor) -> Tensor:
        x_norm = self.norm(x)

        main_x = F.silu(self.conv(self.x_proj(x_norm)))
        res_x = F.silu(self.x_res_proj(x_norm))

        o_raw = self.ssm(main_x)
        o_gated = o_raw * res_x
        o = self.out_proj(o_gated)

        return o + x
