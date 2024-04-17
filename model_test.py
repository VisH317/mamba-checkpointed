import torch
from torch import nn, Tensor
from modules.mamba_checkpointed import MambaBlock

model = MambaBlock(8, 16, 64, 16, 5).cuda()

model(torch.rand(1, 32, 8).cuda())


