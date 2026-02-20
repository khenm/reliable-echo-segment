import torch.nn as nn
from src.utils.logging import get_logger
from src.models.mamba2.block import Mamba2Block

logger = get_logger()

class MambaBlock(nn.Module):
    """
    Wrapper for Mamba-2 Block.
    """
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.inner = Mamba2Block(dim=d_model, **kwargs)

    def forward(self, x):
        return self.inner(x)
    
    def step(self, x, state=None):
        return self.inner.step(x, state)
