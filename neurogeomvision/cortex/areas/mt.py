import torch
import torch.nn as nn
from typing import Dict

class MTArea(nn.Module):
    """
    Middle Temporal (MT/V5) Area (Dorsal Stream).
    Motion detection (simplified here as low-res spatial features/Magno integration).
    """
    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        # Large kernel to capture broader changes (proxy for motion in static images)
        self.process = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)) # Forced low spatial resolution
        )

    def forward(self, x) -> Dict[str, torch.Tensor]:
        out = self.process(x)
        return {"combined_response": out}