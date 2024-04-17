import torch
import triton
import triton.language as tl

torch.set_default_device("cuda")

alpha = 0.9

SEQLEN = 128

def ema(x, alpha):
    y = []
    h = 0
    for i in range(len(x)):
        h = alpha * h + (1 - alpha) * x[i]
        y.append(h)
    
    return y, h

def ssm_scan(x, a, b, c):
    y = []
    h = 0
    for i in range(len(x)):
        h = h * a + b * x[i]
        y.append(c * h)
    
    return y, h

def op(x, y):
    return (x[0] * y[0], x[1]*y[0] + y[1])

def ssm_associative(x, a, b, c):
    y = []
    h = 0
    for i in range(len(x)):
        h_new = (a, b * x[i])
        h = op(h, h_new)
        y.append(c * h)
    
    return y, h


@triton.jit
def op_t(fa, fb, sa, sb):
    a = fa * sa
    b = sa * fb + sb
    return a, b

@triton.jit
def ssm_load(Ks, A, B, C):
    a = tl.load(A + Ks)
    b = tl.load(B + Ks)
    c = tl.load(C + Ks)
    return a, b, c

@triton.jit
def ssm_tt_associative(X, Y, A, B, C, K: tl.constexpr):
    Ks = tl.arange(0, K)

    bid = tl.program_id(0)
    kid = bid * K
    x = tl.load(X + Ks + kid)
    a, b, c = ssm_load(Ks + kid, A, B, C)

    h1, h2 = tl.associative_scan((a, b*x), 0, op_t)
    y = c * h2

    tl.store(Y + Ks + kid, y)


@triton.jit
def ssm_scan_tt_f(h1, h2, h2_0, reversed: tl.constexpr = 0, dim: tl.constexpr = 0):
    Ks = tl.arange(0, h2.shape[dim])

    n1, n2 = op_t(tl.zeros_like(h1) + 1.0, h2_0, h1, h2) # calculate initial ssm state by running through first pass

    h1, h2 = tl.associative_scan((n1, n2), dim, op_t, reverse=reversed)
    return h1, h2

@triton.jit
def ssm_scan_tt(X, A, B, C, H_0, Y, H, K: tl.constexpr):
    pid = tl.program_id(0)
    n = tl.num_programs(0)
    kid = pid * K
    Ks = tl.arange(0, K)

    a, b, c = ssm_load(Ks + kid, A, B, C)
    h_span = Ks*0 + kid
    x = tl.load(X + Ks + kid)

    h2_0 = tl.load(H_0 + n + h_span, Ks==0, 0)

    # h1, h2 = ssm_scan_tt_f(a, b*x, h2_0, reversed=False, dim=0)
    n1, n2 = op_t(tl.zeros_like(a) + 1.0, h2_0, a, b*x) # calculate initial ssm state by running through first pass

    h1, h2 = tl.associative_scan((n1, n2), 0, op_t)

    tl.store(Y + Ks + kid, h2)

    tl.store(H + 0 * n + h_span, h1, Ks == (K-1))
    tl.store(H + 1 * n + h_span, h2, Ks == (K-1))

BLOCKSIZE = 16
BLOCKS = SEQLEN // BLOCKSIZE

x = torch.arange(0, SEQLEN)
y = torch.arange(0, SEQLEN)
a = torch.ones(SEQLEN) * alpha
b = torch.ones(SEQLEN) - alpha
c = torch.ones(SEQLEN)
h = torch.zeros(2, 2, BLOCKS).float().cuda()

ssm_scan_tt[(BLOCKS,)](x, a, b, c, h[0], y, h[0], K=BLOCKSIZE)
ssm_tt_associative[(1,)](h[0, 1], h[0,0], torch.ones(BLOCKS), torch.ones(BLOCKS), h[1, 1], K=BLOCKS)
ssm_scan_tt[(BLOCKS,)](x, a, b, c, torch.roll(h[1], 1), y, h[1], K=BLOCKSIZE)

print("worked")


@triton.jit
def rol(a1, b1_last, b1_cur, a2, b2_last, b2_cur):
    return a1 + a2, tl.where(a2 == 1, b1_cur, 0) + b2_last, b2_cur # recomputes previous hidden state of each item for A derivative

@triton.jit
def roll(y, dim, rev=0):
    _, rh2, _ = tl.associative_scan((1 + 0 * y, 0.0*y, y), dim, rol, reverse=rev)
    return rh2

@triton.jit
def ssm_store(Ks, dA, da, dB, db, dC, dc):
    tl.store(dA + Ks, da)
    tl.store(dB + Ks, db)
    tl.store(dC + Ks, dc)

@triton.jit
def ssm1_tt(X, dX, A, dA, B, dB, C, dC, Y, dY, K: tl.constexpr):
    Ks = tl.arange(0, K)
    a, b, c = ssm_load(Ks, A, B, C)
    x = tl.load(X + Ks)
    dy = tl.load(dY + Ks)
    id2 = tl.zeros_like(a) # 0 a-size matrix, the original hidden state matrix

    h1, h2 = ssm_scan_tt_f(a, b*x, id2)
    y = c * h2
    tl.store(Y + Ks, y)
    a_shift = tl.load(A + Ks + 1, Ks + 1 < K, 0) # get a shifted version of the a matrix

    h1, dh = ssm_scan(a_shift, c * dy, id2, reversed=1) # reverse computation using the shifted A matrix
    rh2 = roll(h2, 0)

    tl.store(dX + Ks, b * dh)
    ssm_store(Ks, dA, dh*rh2, dB, dh*x, dC, h2 * dy)
