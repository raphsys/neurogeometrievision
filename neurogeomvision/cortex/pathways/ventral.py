import torch
import torch.nn as nn
from ..areas import V1Area, V2Area, V4Area

class VentralStream(nn.Module):
    """
    The 'What' pathway: V1 -> V2 -> V4.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.v1 = V1Area(in_channels, out_channels=32)
        self.v2 = V2Area(32, out_channels=64)
        self.v4 = V4Area(64, out_channels=128)

    def forward(self, x):
        v1_out = self.v1(x)['combined_response']
        v2_out = self.v2(v1_out)['combined_response']
        v4_out = self.v4(v2_out)['combined_response']
        return v4_out