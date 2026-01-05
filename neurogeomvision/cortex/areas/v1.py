import torch
import torch.nn as nn
import numpy as np
from typing import Dict

class V1Area(nn.Module):
    """
    Primary Visual Cortex (V1).
    Specializes in edge detection (Orientation).
    """
    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        # Simulating simple cells with Gabor-like filters via Conv2d
        self.simple_cells = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        # Complex cells (max pooling over local region to achieve phase invariance)
        self.complex_cells = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        simple = self.simple_cells(x)
        complex_out = self.complex_cells(simple)
        return {
            "combined_response": complex_out,
            "simple_cells": simple
        }