"""
Module networks.py - Architectures de réseaux SNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional
import math


class SNNClassifier(nn.Module):
    """
    Classificateur SNN.
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_sizes: List[int],
                 num_classes: int,
                 n_timesteps: int = 5,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.n_timesteps = n_timesteps
        self.device = device
        
        # Couches cachées
        self.layers = nn.ModuleList()
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            from .layers import SNNLinear
            self.layers.append(
                SNNLinear(prev_size, hidden_size, device=device)
            )
            prev_size = hidden_size
        
        # Couche de sortie
        self.output_layer = nn.Linear(prev_size, num_classes, device=device)
        
        # Taille de la dernière couche cachée
        self.last_hidden_size = prev_size
    
    def reset_state(self):
        """Réinitialise tous les états."""
        for layer in self.layers:
            layer.reset_state()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass.
        
        Args:
            x: Entrée (batch_size, input_size) ou (input_size,)
            
        Returns:
            logits: Prédictions
            info: Informations
        """
        # Gestion de la forme
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            batch_size = 1
            was_1d = True
        else:
            batch_size = x.shape[0]
            was_1d = False
        
        if x.dtype != torch.float32:
            x = x.float()
        
        # Accumulateur pour les spikes
        spike_accumulator = []
        
        # Simulation sur plusieurs pas de temps
        for t in range(self.n_timesteps):
            # Forward pass à travers les couches SNN
            spikes = x
            for layer in self.layers:
                spikes, _ = layer(spikes)
            
            # S'assurer de la forme correcte
            if len(spikes.shape) == 1:
                spikes = spikes.unsqueeze(0)
            
            spike_accumulator.append(spikes)
        
        # Moyenne sur les pas de temps
        if spike_accumulator:
            spikes_stacked = torch.stack(spike_accumulator, dim=1)  # (batch, timesteps, features)
            pooled = spikes_stacked.mean(dim=1)  # (batch, features)
        else:
            pooled = x
        
        # Couche de sortie
        logits = self.output_layer(pooled)
        
        # Retourner à la forme originale si nécessaire
        if was_1d:
            logits = logits.squeeze(0)
        
        # Informations
        info = {
            'n_timesteps': self.n_timesteps,
            'pooled_output_shape': pooled.shape,
            'logits_shape': logits.shape,
            'batch_size': batch_size,
            'last_hidden_size': self.last_hidden_size
        }
        
        return logits, info


class SNNVisualEncoder(nn.Module):
    """
    Encodeur visuel SNN.
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int, int],  # (channels, height, width)
                 encoding_size: int = 128,
                 n_timesteps: int = 3,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.channels, self.height, self.width = input_shape
        self.encoding_size = encoding_size
        self.n_timesteps = n_timesteps
        self.device = device
        
        # Couches convolutionnelles
        from .layers import SNNConv2d
        self.conv_layers = nn.ModuleList([
            SNNConv2d(self.channels, 16, kernel_size=3, padding=1, device=device),
            SNNConv2d(16, 32, kernel_size=3, padding=1, device=device),
        ])
        
        # Pooling spatial
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Taille après pooling
        self.flattened_size = 32 * 4 * 4
        
        # Couche linéaire
        from .layers import SNNLinear
        self.encoder = SNNLinear(self.flattened_size, encoding_size, device=device)
    
    def reset_state(self):
        """Réinitialise tous les états."""
        for layer in self.conv_layers:
            layer.reset_state()
        self.encoder.reset_state()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Encode une image.
        
        Args:
            x: Image (batch, channels, height, width) ou (channels, height, width)
            
        Returns:
            encoding: Encodage
            info: Informations
        """
        # Gestion de la forme
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            batch_size = 1
            was_3d = True
        else:
            batch_size = x.shape[0]
            was_3d = False
        
        if x.dtype != torch.float32:
            x = x.float()
        
        # Accumulateur pour les encodages
        encoding_accumulator = []
        
        # Simulation sur plusieurs pas de temps
        for t in range(self.n_timesteps):
            # Forward pass convolutionnel
            spikes = x
            for conv_layer in self.conv_layers:
                spikes, _ = conv_layer(spikes)
            
            # Pooling spatial
            pooled = self.spatial_pool(spikes)
            
            # Flatten
            flattened = pooled.reshape(batch_size, -1)
            
            # Encodage
            encoding_spikes, _ = self.encoder(flattened)
            
            if len(encoding_spikes.shape) == 1:
                encoding_spikes = encoding_spikes.unsqueeze(0)
            
            encoding_accumulator.append(encoding_spikes)
        
        # Moyenne sur les pas de temps
        if encoding_accumulator:
            encoding_stacked = torch.stack(encoding_accumulator, dim=1)  # (batch, timesteps, encoding_size)
            encoding = encoding_stacked.mean(dim=1)  # (batch, encoding_size)
        else:
            encoding = torch.zeros(batch_size, self.encoding_size, device=self.device)
        
        # Retourner à la forme originale si nécessaire
        if was_3d:
            encoding = encoding.squeeze(0)
        
        # Informations
        info = {
            'n_timesteps': self.n_timesteps,
            'encoding_shape': encoding.shape,
            'batch_size': batch_size,
            'flattened_size': self.flattened_size
        }
        
        return encoding, info
