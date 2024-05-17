import torch
from torch import nn, Tensor
import triton
import triton.language as tl


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


class Pooler(nn.Module):
    def __init__(self, d_hidden: int) -> None:
        super().__init__()
        self.lin = nn.Linear(d_hidden, d_hidden)
        self.act = nn.Tanh()
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        # pooling is taking first token apparently :skull:
        out = hidden_states[:, 0]
        return self.act(self.lin(out))
    
    
class Classifier(nn.Module):
    def __init__(self, d_hidden: int, num_classes: int, dropout_p = 0.25) -> None:
        super().__init__()
        self.lin = nn.Linear(d_hidden, num_classes)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout_p)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.softmax(self.dropout(self.lin(x)))

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