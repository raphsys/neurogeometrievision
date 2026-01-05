"""
Module visual_processing.py - Traitement visuel avec SNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional

# Import nécessaire
from .neurons import LIFLayer


class RetinaEncoder(nn.Module):
    """
    Encodeur rétinien SNN.
    """
    
    def __init__(self,
                 image_size: Tuple[int, int],
                 n_channels: int = 2,  # ON et OFF
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.height, self.width = image_size
        self.n_channels = n_channels
        self.device = device
        
        # Filtres DoG simplifiés
        self.filters = self._create_filters()
        
        # Couche LIF
        n_neurons = n_channels * self.height * self.width  # Utiliser self.height et self.width
        self.lif_layer = LIFLayer(n_neurons, device=device)
    
    def _create_filters(self) -> torch.Tensor:
        """Crée des filtres rétiniens."""
        filters = []
        kernel_size = 5
        
        for i in range(self.n_channels):
            if i % 2 == 0:
                # Filtre ON
                filt = torch.ones(kernel_size, kernel_size, device=self.device) * -0.1
                center = kernel_size // 2
                filt[center, center] = 1.0
            else:
                # Filtre OFF
                filt = torch.ones(kernel_size, kernel_size, device=self.device) * 0.1
                center = kernel_size // 2
                filt[center, center] = -1.0
            
            # Normalisation
            filt = filt / (filt.abs().sum() + 1e-8)
            filters.append(filt.unsqueeze(0).unsqueeze(0))
        
        return torch.cat(filters, dim=0)
    
    def forward(self, image: torch.Tensor) -> Dict:
        """
        Encode une image.
        
        Args:
            image: Image (height, width) ou (batch, 1, height, width)
            
        Returns:
            Résultats
        """
        if len(image.shape) == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif len(image.shape) == 3:
            image = image.unsqueeze(1)
        
        batch_size = image.shape[0]
        
        # Filtrage
        filtered = []
        for i in range(self.n_channels):
            filt = self.filters[i:i+1]
            conv = F.conv2d(image, filt, padding=self.filters.shape[-1]//2)
            filtered.append(conv)
        
        filtered_tensor = torch.cat(filtered, dim=1)
        
        # Flatten
        filtered_flat = filtered_tensor.flatten(1)
        
        # Génération de spikes
        spikes, voltages = self.lif_layer(filtered_flat)
        
        # Remise en forme
        spikes_reshaped = spikes.view(batch_size, self.n_channels, self.height, self.width)
        
        return {
            'spikes': spikes_reshaped,
            'voltages': voltages,
            'filtered': filtered_tensor
        }
