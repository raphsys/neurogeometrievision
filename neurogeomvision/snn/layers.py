"""
Module layers.py - Couches SNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional
import math

# Import nécessaire pour utiliser LIFLayer dans SNNLinear
from .neurons import LIFLayer


class SNNLinear(nn.Module):
    """
    Couche linéaire pour SNN.
    """
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 tau_m: float = 20.0,
                 v_thresh: float = 1.0,
                 bias: bool = True,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.tau_m = tau_m
        self.v_thresh = v_thresh
        self.device = device
        
        # Poids synaptiques
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, device=device) * 0.1
        )
        
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, device=device)
            )
        else:
            self.register_parameter('bias', None)
        
        # Couche LIF interne
        self.lif_layer = LIFLayer(out_features, tau_m=tau_m, v_thresh=v_thresh, device=device)
    
    def reset_state(self):
        """Réinitialise l'état."""
        self.lif_layer.reset_state()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Entrée (batch_size, in_features) ou (in_features,)
            
        Returns:
            spikes: Sorties binaires
            voltages: Potentiels
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        if x.dtype != torch.float32:
            x = x.float()
        
        # Transformation linéaire
        currents = F.linear(x, self.weight, self.bias)
        
        # Couche LIF
        spikes, voltages = self.lif_layer(currents)
        
        return spikes, voltages


class SNNConv2d(nn.Module):
    """
    Couche convolutionnelle pour SNN.
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 tau_m: float = 20.0,
                 v_thresh: float = 1.0,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.tau_m = tau_m
        self.v_thresh = v_thresh
        self.device = device
        
        # Convolution
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        
        # Initialisation
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        
        # Calcul du nombre de neurones
        self.n_neurons = None
        self.lif_layer = None
    
    def reset_state(self):
        """Réinitialise l'état."""
        if self.lif_layer is not None:
            self.lif_layer.reset_state()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Entrée (batch, channels, height, width)
            
        Returns:
            spikes: Sorties binaires
            voltages: Potentiels
        """
        if x.dtype != torch.float32:
            x = x.float()
        
        # Convolution
        currents = self.conv(x)
        
        # Forme de sortie
        batch_size, out_channels, height, width = currents.shape
        
        # Calcul du nombre de neurones
        n_neurons = out_channels * height * width
        
        # Redimensionner pour LIFLayer
        currents_flat = currents.reshape(batch_size, n_neurons)
        
        # Créer ou réutiliser la couche LIF
        if self.lif_layer is None or self.n_neurons != n_neurons:
            self.lif_layer = LIFLayer(n_neurons, tau_m=self.tau_m, v_thresh=self.v_thresh, device=self.device)
            self.n_neurons = n_neurons
        
        # Appliquer LIF
        spikes_flat, voltages_flat = self.lif_layer(currents_flat)
        
        # Remettre en forme
        spikes = spikes_flat.reshape(batch_size, out_channels, height, width)
        voltages = voltages_flat.reshape(batch_size, out_channels, height, width)
        
        return spikes, voltages


class TemporalPooling(nn.Module):
    """
    Pooling temporel pour SNN.
    """
    
    def __init__(self,
                 window_size: int = 10,
                 mode: str = 'mean',
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.window_size = window_size
        self.mode = mode  # 'mean', 'max', 'sum'
        self.device = device
        
        # Buffer
        self.buffer = []
    
    def reset_state(self):
        """Réinitialise l'état."""
        self.buffer = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pooling temporel.
        
        Args:
            x: Entrée (n_neurons,) ou (batch_size, n_neurons)
            
        Returns:
            Sortie poolée
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # Ajouter au buffer
        self.buffer.append(x.detach().clone())
        
        # Garder seulement window_size éléments
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        
        # Pooling
        if len(self.buffer) > 0:
            stacked = torch.stack(self.buffer, dim=0)
            
            if self.mode == 'mean':
                pooled = stacked.mean(dim=0)
            elif self.mode == 'max':
                pooled = stacked.max(dim=0)[0]
            elif self.mode == 'sum':
                pooled = stacked.sum(dim=0)
            else:
                raise ValueError(f"Mode {self.mode} non supporté")
            
            return pooled
        else:
            return x
