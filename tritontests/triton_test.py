import triton
import triton.language as tl
import torch

torch.set_default_device("cuda")

@triton.jit
def triton_hello_world(X, Y, Z, K: tl.constexpr, L: tl.constexpr):
    Ks = tl.arange(0, K)
    Ls = tl.arange(0, L)[:, None]

    # load from memory
    x = tl.load(X + Ks)
    y = tl.load(Y + Ls*K + Ks)
    z = x + y

    tl.store(Z + Ls*K + Ks, z)


@triton.jit
def triton_hw_block(X, Y, Z, K: tl.constexpr, L: tl.constexpr):
    pid = tl.program_id(0)
    lid = pid * L
    
    Ks = tl.arange(0, K)
    Ls = tl.arange(0, L)[:, None]

    x = tl.load(X + Ks)
    y = tl.load(Y + (Ls + lid) * K + Ks)
    z = x + y

    tl.store(Z + (Ls + lid) * K + Ks, z)

L = 2**10
x, y = torch.arange(4),torch.ones(L, 4)
z = torch.zeros(L, 4)
num_blocks = 8
triton_hw_block[(L // num_blocks,)](x, y, z, 4, num_blocks)
print(z.shape, z)
