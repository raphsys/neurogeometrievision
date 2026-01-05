import torch
import torch.nn as nn
from typing import Dict, Tuple

from .retina import BioInspiredRetina
from .cortex.pathways import VentralStream, DorsalStream

class IntegratedVisionSystem(nn.Module):
    """
    Complete pipeline: Retina -> Cortex (Ventral/Dorsal) -> Classifier.
    """
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (3, 32, 32),
                 n_classes: int = 10,
                 use_retina: bool = True,
                 device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.use_retina = use_retina
        
        # 1. Retina
        if use_retina:
            self.retina = BioInspiredRetina(input_shape, cortical_shape=(32, 32))
            # Retina outputs P (6 channels) and M (6 channels) channels after Bipolar(ON/OFF) and Ganglion processing
            # Bipolar has 6 chans (3*2), Ganglion maps them.
            # Let's check GanglionCellLayer: p_conv outputs 'in_channels' amount. 
            # In Retina.py, Ganglion takes 6 chans, outputs 6 chans for P.
            retina_out_c = 6 
        else:
            retina_out_c = input_shape[0]

        # 2. Pathways
        # Ventral receives P-pathway (High detail)
        self.ventral = VentralStream(in_channels=retina_out_c)
        
        # Dorsal receives M-pathway (Low detail/Motion)
        # If use_retina is False, both get same input
        self.dorsal = DorsalStream(in_channels=retina_out_c)
        
        # 3. Fusion & Classification
        # Ventral output: 128 channels, halved resolution twice (V1->V2(stride1)->V4(maxpool))
        # Dorsal output: 64 channels, adaptive pool 4x4
        
        # We need to calculate flatten size dynamically or assume standard
        # Let's use a dummy forward in init to check sizes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(256), # LazyLinear infers input shape
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        # Retina Pass
        if self.use_retina:
            ret_out = self.retina(x)
            p_in = ret_out['retina_p_out']
            m_in = ret_out['retina_m_out']
        else:
            p_in = x
            m_in = x
            ret_out = {}
            
        # Cortex Pass
        ventral_feat = self.ventral(p_in)
        dorsal_feat = self.dorsal(m_in)
        
        # Feature Fusion
        # Global pooling for ventral to match dorsal's compactness?
        v_pool = torch.nn.functional.adaptive_avg_pool2d(ventral_feat, (4, 4))
        d_pool = torch.nn.functional.adaptive_avg_pool2d(dorsal_feat, (4, 4))
        
        # Concatenate [B, C, 4, 4]
        fusion = torch.cat([v_pool, d_pool], dim=1)
        
        logits = self.classifier(fusion)
        
        return {
            "final_output": logits,
            "retina_outputs": ret_out,
            "ventral_features": ventral_feat,
            "dorsal_features": dorsal_feat
        }