import torch
import torch.nn as nn
from typing import Dict

class V2Area(nn.Module):
    """
    Secondary Visual Cortex (V2).
    Feature combinations, slightly larger RF.
    """
    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x) -> Dict[str, torch.Tensor]:
        out = self.layers(x)
        return {"combined_response": out}