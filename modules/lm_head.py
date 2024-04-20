import torch
from torch import nn, Tensor
from modules.mamba_utils import RMSNorm


class LMHead(nn.Module):
    def __init__(self, d_model: int, out_size: int):
        super().__init__()
        self.lin = nn.Linear(d_model, out_size)
        self.norm = RMSNorm()
        # self.sm = nn.Softmax(-1)

    def forward(self, input: Tensor) -> Tensor:
        return self.lin(self.norm(input))


