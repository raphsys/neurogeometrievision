import torch
import torch.nn as nn
from ..areas import V1Area, V2Area, MTArea

class DorsalStream(nn.Module):
    """
    The 'Where'/'How' pathway: V1 -> V2 -> MT.
    Receives input potentially from M-pathway (lower spatial res).
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.v1 = V1Area(in_channels, out_channels=32)
        # Dorsal stream often bypasses or has quicker transit. 
        # Here we model V1->V2->MT
        self.v2 = V2Area(32, out_channels=32) 
        self.mt = MTArea(32, out_channels=64)

    def forward(self, x):
        v1_out = self.v1(x)['combined_response']
        v2_out = self.v2(v1_out)['combined_response']
        mt_out = self.mt(v2_out)['combined_response']
        return mt_out