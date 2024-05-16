import torch
from torch import nn, Tensor
import torch.nn.functional as F
from modules.mamba_utils import RMSNorm


class LMHead(nn.Module):
    def __init__(self, d_model: int, out_size: int):
        super().__init__()
        # self.lin1 = nn.Linear(d_model, d_model)
        self.lin2 = nn.Linear(d_model, out_size)
        # self.norm = nn.LayerNorm(d_model)
        # self.dropout = nn.Dropout(p=0.25)
        # self.sm = nn.Softmax(-1)

    def forward(self, input: Tensor) -> Tensor:
        return self.lin2(input)

