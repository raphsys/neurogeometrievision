"""
Module wilson_cowan.py - Modèle de Wilson-Cowan pour la dynamique corticale
VERSION OPTIMISÉE
"""

import torch
import numpy as np
from typing import Tuple, Optional, List
import math
import matplotlib.pyplot as plt


class WilsonCowanModel:
    """
    Modèle de Wilson-Cowan pour les dynamiques d'activité corticale.
    VERSION OPTIMISÉE avec convolution rapide.
    """
    
    def __init__(self, 
                 spatial_shape: Tuple[int, int],
                 device: str = 'cpu'):
        """
        Args:
            spatial_shape: (height, width) du cortex
            device: 'cpu' ou 'cuda'
        """
        self.spatial_shape = spatial_shape
        self.height, self.width = spatial_shape
        self.device = device
        
        # Paramètres
        self.tau_e = 10.0
        self.tau_i = 20.0
        
        # Poids synaptiques
        self.w_ee = 10.0
        self.w_ei = 12.0  
        self.w_ie = 10.0
        self.w_ii = 2.0
        
        # Fonction de transfert
        self.beta = 1.5
        self.theta = 4.0
        
        # Entrées externes
        self.I_ext_e = 0.0
        self.I_ext_i = 0.0
        
        # Connectivité spatiale (pré-calculée)
        self.exc_kernel = self._create_kernel(3.0)
        self.inh_kernel = self._create_kernel(6.0)
        
        # État
        self.E = None
        self.I = None
        
        # Initialise
        self.initialize_state()
    
    def _create_kernel(self, sigma: float) -> torch.Tensor:
        """Crée un noyau gaussien - OPTIMISÉ."""
        kernel_size = int(2 * sigma * 2) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        half = kernel_size // 2
        
        # Broadcasting pour éviter meshgrid
        coords = torch.arange(-half, half + 1, device=self.device).float()
        x = coords.view(1, -1)
        y = coords.view(-1, 1)
        
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        return kernel / kernel.sum()
    
    def initialize_state(self, 
                        noise_level: float = 0.1,
                        pattern: str = 'random'):
        """Initialise les états E et I."""
        if pattern == 'random':
            self.E = torch.rand(self.height, self.width, device=self.device) * noise_level
            self.I = torch.rand(self.height, self.width, device=self.device) * noise_level
            
        elif pattern == 'bump':
            self.E = torch.zeros(self.height, self.width, device=self.device)
            self.I = torch.zeros(self.height, self.width, device=self.device)
            
            center_y, center_x = self.height // 2, self.width // 2
            radius = min(self.height, self.width) // 4
            
            # Crée un masque circulaire vectorisé
            y_coords, x_coords = torch.meshgrid(
                torch.arange(self.height, device=self.device),
                torch.arange(self.width, device=self.device),
                indexing='ij'
            )
            
            dist_sq = (y_coords - center_y)**2 + (x_coords - center_x)**2
            mask = dist_sq < radius**2
            
            self.E[mask] = 0.5
            self.I[mask] = 0.3
    
    def sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """Fonction sigmoïde de transfert."""
        return 1.0 / (1.0 + torch.exp(-self.beta * (x - self.theta)))
    
    def spatial_convolution(self, 
                          activity: torch.Tensor,
                          kernel: torch.Tensor) -> torch.Tensor:
        """
        Convolution spatiale rapide.
        """
        activity_4d = activity.unsqueeze(0).unsqueeze(0)
        kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
        
        ks = kernel.shape[0]
        pad = ks // 2
        
        convolved = torch.nn.functional.conv2d(
            torch.nn.functional.pad(activity_4d, (pad, pad, pad, pad), mode='reflect'),
            kernel_4d,
            padding=0
        ).squeeze()
        
        return convolved
    
    def step(self, dt: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Un pas d'intégration - OPTIMISÉ.
        """
        # Convolutions spatiales
        E_conv_exc = self.spatial_convolution(self.E, self.exc_kernel)
        E_conv_inh = self.spatial_convolution(self.E, self.inh_kernel)
        I_conv = self.spatial_convolution(self.I, self.inh_kernel)
        
        # Entrées
        input_E = self.w_ee * E_conv_exc - self.w_ie * I_conv + self.I_ext_e
        input_I = self.w_ei * E_conv_inh - self.w_ii * I_conv + self.I_ext_i
        
        # Équations différentielles
        dE_dt = (-self.E + self.sigmoid(input_E)) / self.tau_e
        dI_dt = (-self.I + self.sigmoid(input_I)) / self.tau_i
        
        # Intégration
        E_new = torch.clamp(self.E + dE_dt * dt, 0, 1)
        I_new = torch.clamp(self.I + dI_dt * dt, 0, 1)
        
        self.E = E_new
        self.I = I_new
        
        return E_new, I_new
    
    def simulate(self, 
                n_steps: int = 100,
                dt: float = 1.0) -> torch.Tensor:
        """
        Simulation complète.
        """
        for _ in range(n_steps):
            self.step(dt)
        
        return self.E
    
    def generate_pattern(self, pattern_type: str = 'stripes') -> torch.Tensor:
        """
        Génère un pattern spécifique.
        """
        if pattern_type == 'stripes':
            self.w_ee = 12.0
            self.w_ei = 10.0
            self.I_ext_e = 2.0
            
            # Initialise avec des rayures
            self.E = torch.zeros(self.height, self.width, device=self.device)
            period = 10
            for y in range(self.height):
                if (y // period) % 2 == 0:
                    self.E[y, :] = 0.6
            self.I = self.E * 0.7
        
        # Simulation
        self.simulate(n_steps=50, dt=0.5)
        
        return self.E
