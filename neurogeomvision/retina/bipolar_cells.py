import torch
import torch.nn as nn
from typing import Dict, List

class BipolarCellLayer(nn.Module):
    """
    Simulates Bipolar Cells with Center-Surround receptive fields (DoG).
    Splits signal into ON and OFF channels.
    """
    
    def __init__(self, in_channels: int, kernel_size: int = 5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 # ON and OFF for each input channel
        
        # We implement Difference of Gaussians (DoG) via depthwise convolutions
        # Center (Excitatory for ON)
        self.center_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                     padding=1, groups=in_channels, bias=False)
        # Surround (Inhibitory for ON)
        self.surround_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                       padding=kernel_size//2, groups=in_channels, bias=False)
        
        self._init_dog_weights()

    def _init_dog_weights(self):
        # Initialize with fixed Gaussian approximations for stability
        nn.init.dirac_(self.center_conv.weight) # Sharp center
        nn.init.constant_(self.surround_conv.weight, 1.0 / (self.surround_conv.kernel_size[0]**2)) # Broad surround

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        center = self.center_conv(x)
        surround = self.surround_conv(x)
        
        # ON Cells: Center - Surround
        on_response = torch.relu(center - surround)
        
        # OFF Cells: Surround - Center
        off_response = torch.relu(surround - center)
        
        # Concatenate ON and OFF
        # [B, C*2, H, W]
        combined = torch.cat([on_response, off_response], dim=1)
        
        return {
            "combined_response": combined,
            "on_response": on_response,
            "off_response": off_response
        }