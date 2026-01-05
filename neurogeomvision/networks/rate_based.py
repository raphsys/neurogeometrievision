import torch.nn as nn

class RateBasedBlock(nn.Module):
    """
    Standard Conv-BatchNorm-ReLU block used as a rate-based proxy.
    """
    def __init__(self, in_c, out_c, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.block(x)