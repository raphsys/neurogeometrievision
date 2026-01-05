import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .photoreceptors import PhotoreceptorLayer
from .bipolar_cells import BipolarCellLayer
from .ganglion_cells import GanglionCellLayer

class RetinotopicMapping(nn.Module):
    """
    Maps Retinal output to Cortical input space.
    Currently a simplified spatial interpolation.
    TODO: Implement Log-Polar mapping for foveal magnification.
    """
    def __init__(self, target_shape: Tuple[int, int]):
        super().__init__()
        self.target_shape = target_shape
        self.mode = 'bilinear'
        self.simplified = True # Mark as simplified implementation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=self.target_shape, mode=self.mode, align_corners=False)

class BioInspiredRetina(nn.Module):
    def __init__(self, 
                 input_shape: Tuple[int, int, int], # C, H, W
                 cortical_shape: Tuple[int, int] = (64, 64)):
        super().__init__()
        c, h, w = input_shape
        
        self.photoreceptors = PhotoreceptorLayer(input_channels=c)
        # Bipolar input channels = photoreceptor types (3)
        self.bipolar = BipolarCellLayer(in_channels=3)
        # Ganglion input = Bipolar output (ON+OFF = 6)
        self.ganglion = GanglionCellLayer(in_channels=6)
        
        self.mapping_p = RetinotopicMapping(cortical_shape)
        # M-pathway usually feeds into different areas, we keep it distinct
        # M is naturally lower res, but we might map it to cortical shape for consistency
        self.mapping_m = RetinotopicMapping((cortical_shape[0]//2, cortical_shape[1]//2))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        photo_out = self.photoreceptors(x)
        bipolar_out = self.bipolar(photo_out['combined_response'])
        ganglion_out = self.ganglion(bipolar_out['combined_response'])
        
        # Apply Retinotopy
        p_cortical = self.mapping_p(ganglion_out['p_pathway'])
        m_cortical = self.mapping_m(ganglion_out['m_pathway'])
        
        return {
            "retina_p_out": p_cortical,
            "retina_m_out": m_cortical,
            "raw_spikes": ganglion_out
        }