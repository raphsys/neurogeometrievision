import torch
import torch.nn as nn
from typing import Dict

class V4Area(nn.Module):
    """
    V4 Area (Ventral Stream).
    Shape and Color integration.
    """
    def __init__(self, in_channels: int, out_channels: int = 128):
        super().__init__()
        self.process = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2) # Downsample to increase effective RF
        )

    def forward(self, x) -> Dict[str, torch.Tensor]:
        out = self.process(x)
        return {"combined_response": out}