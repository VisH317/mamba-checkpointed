import torch
import triton
import triton.language as tl

import triton.language

print(torch.version.cuda)
torch.set_default_device("cuda")
# constants
K = 16
BLOCKS = 8
SEQLEN = K * BLOCKS

x = torch.arange(SEQLEN)
y = torch.zeros(SEQLEN)

def cumsum(x):
    y = []
    h = 0
    for k in range(len(x)):
        h += x[k]
        y.append(h)
    
    return h, y

# h_, y_ = cumsum(x.tolist())
# print(h_, y_)


@triton.jit
def plus_fn(a, b):
    # This is a helper function where a and b are tensors.
    return a + b

@triton.jit
def cumsum_tt(X, Y, H, K: tl.constexpr):
    # This is the base triton function. Capital letters are pointers to memory.

    # Create a tensor from 0 to K - 1
    Ks = tl.arange(0, K)

    # Load in a sequence of K x's (blue)
    x = tl.load(X + Ks)

    # Compute h (green) and y (yellow) on axis 0.
    hs = tl.associative_scan(x, 0, plus_fn)
    y = hs

    # Write out K y's
    tl.store(Y + Ks, y)

    # Write out only the last h to memory.
    tl.store(H + Ks * 0, hs, mask=Ks == (K-1))


h = torch.zeros(1)
cumsum_tt[(1,)](x, y, h, K=SEQLEN)

@triton.jit
def cumsum_tt_block(X, H_0, Y, H, K: tl.constexpr):
    pid = tl.program_id(0)
    kid = pid * K
    Ks = tl.arange(0, K)

    x = tl.load(X + Ks + kid)

    h_0 = tl.load(H_0 + Ks + pid, Ks == 0, 0) # need to see what this does

    x = plus_fn(h_0, x)

    hs = tl.associative_scan(x, 0, plus_fn)
    y = hs

    tl.store(Y + Ks + kid, y)
    tl.store(H + Ks * 0 + pid, hs, mask=Ks == (K-1))

# h = torch.zeros(BLOCKS)
# cumsum_tt_block[(BLOCKS,)](x, torch.arange(0, BLOCKS), y, h, K=K)

# print(y)
# print(h)

# def cumsum_block(x, y, K):
#     seqlen = y.shape[0]
#     BLOCKS = seqlen // K
#     h = torch.zeros(2, BLOCKS)
#     cumsum_tt_block[(BLOCKS,)](x, h[0], y, h[0], K=K)
#     h[1, 1:] = h[0].cumsum(0)[:-1]
#     cumsum_tt_block[(BLOCKS,)](x, h[1], y, h[1], K=K)
#     print(y)

# cumsum_block(x, y, K)
