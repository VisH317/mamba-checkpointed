import torch
import math
from torch import nn, Tensor
import torch.nn.functional as F
import triton
import triton.language as tl
# from flashfftconv import FlashFFTConv
from modules.mamba_utils import RMSNorm
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange


device = torch.device("cuda")

# @triton.jit
# def generate_a(A, d_in, d_hidden, BLOCK_SIZE: tl.constexpr):
#     pid = tl.program_id(axis=0)
#     tl.device_print("pid", pid)
#     block_start = pid * BLOCK_SIZE
#     offsets = block_start + tl.arange(0, BLOCK_SIZE)
#     mask = offsets < d_in * d_hidden

#     # loopy doopy because there is no parallel filter function / indexing in triton bruh :(
#     for offset in offsets: 
#         x = offset // d_in
#         y = offset - (x * d_in)
#         value = tl.sqrt(2 * x + 1) * tl.sqrt(2 * y + 1) if x > y else tl.zeros(1, dtype=tl.float32) if x < y else x + 1
#         tl.store(A + offset, value, mask=mask)

def generate_a(d_in: int, d_hidden: int):
    t = torch.empty(d_in, d_hidden).to(device=device)
    for i in range(d_in):
        for j in range(d_hidden):
            if i == j: t[i][j] = i + 1
            elif i > j: t[i][j] = math.sqrt(2 * i + 1) * math.sqrt(2 * j + 1)
            else: t[i][j] = 0
    
    return t


class S4Checkpointed(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, dt_rank: int, d_memory: int, checkpoint_step: int = 16):
        super().__init__()

        self.d_in = d_in
        self.d_hidden = d_hidden
        self.dt_rank = dt_rank
        self.d_memory = d_memory
        self.step = checkpoint_step

        # BLOCK_SIZE = 32

        # grid = lambda meta: (triton.cdiv(d_hidden * d_in, meta["BLOCK_SIZE"]), )
        # self.A = torch.empty(d_in, d_hidden).to(device=device)
        # generate_a[grid](self.A, d_in, d_hidden, BLOCK_SIZE=BLOCK_SIZE)
        self.A = generate_a(d_in, d_hidden)

        self.D = nn.Parameter(torch.ones(d_in)).cuda()

        self.x_proj = nn.Linear(d_in, 3 * d_hidden + 2 * d_in + 2 * dt_rank, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_in, bias=True)
        self.dt_local_proj = nn.Linear(dt_rank, d_in, bias=True)

        self.w_K = nn.Linear(d_in, d_memory)
        self.w_Q = nn.Linear(d_in, d_memory)
        self.w_V = nn.Linear(d_in, d_memory)
        self.w_O = nn.Linear(d_memory, d_in)


    # refactor all these rearranges later haha
    def forward(self, x: Tensor) -> Tensor:
        bcd = rearrange(self.x_proj(x), "b l d -> b d l")
        (B, B_local, C, E, F, dt, dt_local) = bcd.split(split_size=[self.d_hidden, self.d_hidden, self.d_hidden, self.d_in, self.d_in, self.dt_rank, self.dt_rank], dim=-2)
        delta = rearrange(self.dt_proj(rearrange(dt, "b d l -> b l d")), "b l d -> b d l")
        delta_local = rearrange(self.dt_local_proj(rearrange(dt_local, "b d l -> b l d")), "b l d -> b d l")

        y = selective_scan_fn(rearrange(x, "b l d -> b d l"), delta, self.A, B, C, self.D, z=None, delta_softplus=True)
        y_local = selective_scan_fn(rearrange(x, "b l d -> b d l"), delta_local, self.A, B_local, C=torch.ones_like(B), D=None, z=None, delta_softplus=True) # TODO: gating where
        memory = y_local[:, :, ::self.step]
        att = self.w_O(self.global_attention(x, rearrange(memory, "b d m -> b m d")))
        att_gated = att * rearrange(E, "b d l -> b l d")

        print("y: ", y)
        return rearrange(y, "b d l -> b l d") + att_gated + x * rearrange(F, "b d l -> b l d")


    def global_attention(self, x: Tensor, memory: Tensor):
        Q: Tensor = self.w_Q(x)
        K: Tensor = self.w_K(memory)
        O_w = F.softmax((Q @ K.transpose(-2, -1)) / torch.sqrt(torch.tensor([self.d_hidden]).cuda()), dim=-1) # multiply by the values and weighted the sum!!!!!! poggers in the chat
        # print(O_w.size(), self.w_V(memory).size())
        return torch.einsum("b o l, b l n -> b o n", O_w, self.w_V(memory))



class MambaBlock(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_ssm_hidden: int, dt_rank: int, n_conv: int = 5):
        super().__init__()

        self.conv = nn.Conv1d(d_hidden, d_hidden, bias=True, kernel_size=n_conv, groups=d_in, padding=n_conv-3) # convolutions must be much longer to understand gene level relationships
        # TODO: ADD GATING TO the CONVOLUTION
        self.x_proj = nn.Linear(d_in, d_hidden)
        self.norm = RMSNorm()

        self.x_res_proj = nn.Linear(d_in, d_hidden, bias=False)
        self.out_proj = nn.Linear(d_hidden, d_in)

        self.ssm = S4Checkpointed(d_hidden, d_ssm_hidden, dt_rank, d_ssm_hidden)

    def forward(self, x: Tensor) -> Tensor:
        x_norm = self.norm(x)

        main_x = rearrange(F.silu(self.conv(rearrange(self.x_proj(x_norm), "b l d -> b d l"))), "b d l -> b l d")
        res_x = F.silu(self.x_res_proj(x_norm))

        o_raw = self.ssm(main_x)
        o_gated = o_raw * res_x
        o = self.out_proj(o_gated)

        return o + x



# recurrent implementation (will use later for inference optimization)
        # for i in range(l):
        #     h = Ad[:, i] * h + Bx[:, i]
        #     h_local = A_localx[:, i] * h_local + Bx[:, i]
        #     if i % self.step == 0: memory[:, i // self.step] = h_local.flatten(start_dim=-2, end_dim=-1)
        #     att = self.global_attention(x[:, i].unsqueeze(0), memory, (i // self.step) + 1)
        #     att = self.w_O(att)
        #     att = self.w_E(att).reshape(b, self.d_in)
        #     y = torch.bmm(h, C[:, i, :].unsqueeze(-1)).reshape(b, self.d_in) + att # @ E[:, i, :] TODO: ADD IN THIS E MATRIX LATER
        #     ys.append(y)
        # y = torch.stack(ys, dim=1) + x * self.D
