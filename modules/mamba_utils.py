import torch
from torch import nn, Tensor
import triton
import triton.language as tl


class RMSNorm(nn.Module):
    def __init__(self, affine_transform: bool = True, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.affine_transform = affine_transform
        
        # kevin was here :D
        if affine_transform:
            self.beta = nn.Parameter(torch.zeros(1)).cuda()
            self.gamma = nn.Parameter(torch.ones(1)).cuda()
    
    def forward(self, input: Tensor) -> Tensor:
        size = input.size()[-1]
        var = (torch.rsqrt(torch.sum(input ** 2, dim=-1) / size) + self.eps).cuda()
        # print(input.size(), var.size())
        out = input * var.unsqueeze(-1)
        if self.affine_transform: out = out * self.gamma + self.beta
        return out


# class FFTConv(nn.Module):
#     def __init__(self   )

@triton.jit
def generate_a(A, d_in: int, d_hidden: int, BLOCK_SIZE: tl.constexpr):
    pass
    # pid = tl.program_id(axis=0)
    # tl.device_print("pid", pid)
    # block_start = pid * BLOCK_SIZE
    # offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # mask = offsets < d_in * d_hidden

    # for offset in offsets: # loopy doopy because there is no parallel filter function / indexing in triton bruh :(
    #     x = offset // d_in
    #     y = offset - (x * d_in)
    #     value = tl.sqrt(2 * x + 1) * tl.sqrt(2 * y + 1) if x > y else tl.zeros(1, dtype=tl.float32) if x < y else x + 1
    #     tl.store(A + offset, value, mask=mask)