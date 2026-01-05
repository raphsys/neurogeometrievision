"""
Module retinal_cells.py - Cellules rétiniennes intermédiaires
Cellules horizontales, bipolaires, amacrines
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional
import math


class HorizontalCell(nn.Module):
    """
    Cellule horizontale - Inhibition latérale.
    """
    
    def __init__(self,
                 receptive_field_size: int = 15,
                 spatial_constant: float = 3.0,
                 temporal_constant: float = 50.0,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.receptive_field_size = receptive_field_size
        self.spatial_constant = spatial_constant
        self.temporal_constant = temporal_constant
        self.device = device
        
        # Filtre spatial DoG (Difference of Gaussians)
        self.spatial_filter = self._create_spatial_filter()
        
        # Filtre temporel
        self.register_buffer('temporal_state', torch.tensor(0.0, device=device))
        
        # Gain d'inhibition
        self.inhibition_gain = 0.5
    
    def _create_spatial_filter(self) -> torch.Tensor:
        """Crée un filtre DoG pour l'inhibition latérale."""
        size = self.receptive_field_size
        center = size // 2
        
        # Grille de coordonnées
        y, x = torch.meshgrid(
            torch.arange(size, device=self.device) - center,
            torch.arange(size, device=self.device) - center,
            indexing='ij'
        )
        
        r = torch.sqrt(x**2 + y**2)
        
        # Centre positif, périphérie négative
        sigma_center = self.spatial_constant
        sigma_surround = sigma_center * 2.5
        
        center_gauss = torch.exp(-r**2 / (2 * sigma_center**2))
        surround_gauss = torch.exp(-r**2 / (2 * sigma_surround**2))
        
        # Normaliser
        center_gauss = center_gauss / center_gauss.sum()
        surround_gauss = surround_gauss / surround_gauss.sum()
        
        # Filtre DoG
        dog_filter = center_gauss - 0.5 * surround_gauss
        
        return dog_filter.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    def reset_state(self):
        """Réinitialise l'état."""
        self.temporal_state = torch.tensor(0.0, device=self.device)
    
    def forward(self, photoreceptor_input: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """
        Inhibition latérale.
        
        Args:
            photoreceptor_input: Entrée des photorécepteurs (H, W) ou (B, H, W)
            dt: Pas de temps
            
        Returns:
            Signal d'inhibition
        """
        if len(photoreceptor_input.shape) == 2:
            photoreceptor_input = photoreceptor_input.unsqueeze(0)  # (1, H, W)
        
        batch_size, height, width = photoreceptor_input.shape
        
        # Convolution spatiale
        if height < self.receptive_field_size or width < self.receptive_field_size:
            # Si trop petit, pas de convolution
            spatial_response = photoreceptor_input
        else:
            spatial_response = F.conv2d(
                photoreceptor_input.unsqueeze(1),  # (B, 1, H, W)
                self.spatial_filter,
                padding=self.receptive_field_size // 2
            ).squeeze(1)  # (B, H, W)
        
        # Filtre temporel
        alpha = math.exp(-dt / self.temporal_constant)
        self.temporal_state = alpha * self.temporal_state + (1 - alpha) * spatial_response
        
        # Normalisation
        inhibition = torch.tanh(self.temporal_state * self.inhibition_gain)
        
        return inhibition


class BipolarCell(nn.Module):
    """
    Cellule bipolaire - Transmission centre-surround.
    Types : ON-center, OFF-center
    """
    
    def __init__(self,
                 cell_type: str = 'ON',  # 'ON' ou 'OFF'
                 center_size: int = 5,
                 surround_size: int = 15,
                 center_gain: float = 1.0,
                 surround_gain: float = 0.7,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.cell_type = cell_type
        self.center_size = center_size
        self.surround_size = surround_size
        self.center_gain = center_gain
        self.surround_gain = surround_gain
        self.device = device
        
        # Filtres centre et surround
        self.center_filter = self._create_gaussian_filter(center_size, sigma=center_size/3)
        self.surround_filter = self._create_gaussian_filter(surround_size, sigma=surround_size/4)
        
        # Non-linéarité
        self.nonlinearity = nn.Sigmoid()
    
    def _create_gaussian_filter(self, size: int, sigma: float) -> torch.Tensor:
        """Crée un filtre gaussien."""
        center = size // 2
        
        y, x = torch.meshgrid(
            torch.arange(size, device=self.device) - center,
            torch.arange(size, device=self.device) - center,
            indexing='ij'
        )
        
        r = torch.sqrt(x**2 + y**2)
        gaussian = torch.exp(-r**2 / (2 * sigma**2))
        gaussian = gaussian / gaussian.sum()
        
        return gaussian.unsqueeze(0).unsqueeze(0)
    
    def reset_state(self):
        """Réinitialise l'état."""
        # Cette classe n'a pas d'état à réinitialiser, mais on doit avoir la méthode
        pass
    
    def forward(self,
                photoreceptor_input: torch.Tensor,
                horizontal_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Réponse centre-surround.
        
        Args:
            photoreceptor_input: Entrée directe
            horizontal_input: Inhibition latérale (optionnel)
            
        Returns:
            Réponse de la cellule bipolaire
        """
        if len(photoreceptor_input.shape) == 2:
            photoreceptor_input = photoreceptor_input.unsqueeze(0)  # (1, H, W)
        
        batch_size, height, width = photoreceptor_input.shape
        
        # Réponse du centre
        if height >= self.center_size and width >= self.center_size:
            center_response = F.conv2d(
                photoreceptor_input.unsqueeze(1),
                self.center_filter,
                padding=self.center_size // 2
            ).squeeze(1)
        else:
            center_response = photoreceptor_input
        
        # Réponse du surround
        if height >= self.surround_size and width >= self.surround_size:
            surround_response = F.conv2d(
                photoreceptor_input.unsqueeze(1),
                self.surround_filter,
                padding=self.surround_size // 2
            ).squeeze(1)
        else:
            surround_response = photoreceptor_input.mean() * torch.ones_like(photoreceptor_input)
        
        # Combinaison centre-surround
        if self.cell_type == 'ON':
            # ON-center: centre+, surround-
            response = self.center_gain * center_response - self.surround_gain * surround_response
        else:  # OFF-center
            # OFF-center: centre-, surround+
            response = -self.center_gain * center_response + self.surround_gain * surround_response
        
        # Ajouter l'inhibition latérale si disponible
        if horizontal_input is not None:
            if horizontal_input.shape != response.shape:
                horizontal_input = F.interpolate(
                    horizontal_input.unsqueeze(1),
                    size=response.shape[-2:],
                    mode='bilinear'
                ).squeeze(1)
            response = response - 0.3 * horizontal_input
        
        # Non-linéarité
        response = self.nonlinearity(response)
        
        return response

class AmacrineCell(nn.Module):
    """
    Cellule amacrine - Modulation temporelle et traitement non-linéaire.
    """
    
    def __init__(self,
                 temporal_filter_tau: float = 30.0,
                 nonlinear_gain: float = 2.0,
                 feedback_strength: float = 0.3,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.temporal_filter_tau = temporal_filter_tau
        self.nonlinear_gain = nonlinear_gain
        self.feedback_strength = feedback_strength
        self.device = device
        
        # État temporel
        self.register_buffer('temporal_state', torch.tensor(0.0, device=device))
        self.register_buffer('feedback_state', torch.tensor(0.0, device=device))
        
        # Non-linéarité
        self.nonlinearity = lambda x: torch.tanh(self.nonlinear_gain * x)
    
    def reset_state(self):
        """Réinitialise l'état."""
        self.temporal_state = torch.tensor(0.0, device=self.device)
        self.feedback_state = torch.tensor(0.0, device=self.device)
    
    def forward(self, bipolar_input: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """
        Traitement amacrine.
        
        Args:
            bipolar_input: Entrée des cellules bipolaires
            dt: Pas de temps
            
        Returns:
            Signal modulé
        """
        if len(bipolar_input.shape) == 2:
            bipolar_input = bipolar_input.unsqueeze(0)
        
        # Filtre temporel
        alpha = math.exp(-dt / self.temporal_filter_tau)
        self.temporal_state = alpha * self.temporal_state + (1 - alpha) * bipolar_input
        
        # Feedback
        feedback = self.feedback_strength * self.feedback_state
        self.feedback_state = self.temporal_state.detach()
        
        # Non-linéarité
        modulated = self.nonlinearity(self.temporal_state - feedback)
        
        return modulated


class RetinalNetwork(nn.Module):
    """
    Réseau rétinien complet : photorécepteurs -> horizontales -> bipolaires -> amacrines.
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int],
                 n_on_cells: int = 1,
                 n_off_cells: int = 1,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.input_shape = input_shape
        self.device = device
        
        # Couches
        self.horizontal_cells = HorizontalCell(device=device)
        
        # Cellules bipolaires ON et OFF
        self.on_bipolar_cells = nn.ModuleList([
            BipolarCell(cell_type='ON', device=device)
            for _ in range(n_on_cells)
        ])
        
        self.off_bipolar_cells = nn.ModuleList([
            BipolarCell(cell_type='OFF', device=device)
            for _ in range(n_off_cells)
        ])
        
        # Cellules amacrines
        self.amacrine_cells = nn.ModuleList([
            AmacrineCell(device=device)
            for _ in range(max(n_on_cells, n_off_cells))
        ])
    
    def reset_state(self):
        """Réinitialise tous les états."""
        self.horizontal_cells.reset_state()
        for cell in self.on_bipolar_cells:
            cell.reset_state()
        for cell in self.off_bipolar_cells:
            cell.reset_state()
        for cell in self.amacrine_cells:
            cell.reset_state()
        # Pas de photoreceptors dans RetinalNetwork - c'est dans SimpleRetinaModel
    
    def forward(self,
                photoreceptor_input: torch.Tensor,
                dt: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Traitement complet.
        
        Args:
            photoreceptor_input: Sortie des photorécepteurs
            dt: Pas de temps
            
        Returns:
            Dictionnaire avec les réponses de chaque couche
        """
        # Cellules horizontales
        horizontal_response = self.horizontal_cells(photoreceptor_input, dt)
        
        # Cellules bipolaires ON
        on_responses = []
        for bipolar_cell in self.on_bipolar_cells:
            response = bipolar_cell(photoreceptor_input, horizontal_response)
            on_responses.append(response)
        
        # Cellules bipolaires OFF
        off_responses = []
        for bipolar_cell in self.off_bipolar_cells:
            response = bipolar_cell(photoreceptor_input, horizontal_response)
            off_responses.append(response)
        
        # Cellules amacrines (appliquées aux réponses bipolaires)
        amacrine_responses = []
        for i, amacrine_cell in enumerate(self.amacrine_cells):
            # Prendre la réponse bipolaire correspondante
            if i < len(on_responses):
                bipolar_response = on_responses[i]
            elif i < len(off_responses):
                bipolar_response = off_responses[i]
            else:
                bipolar_response = on_responses[0] if on_responses else off_responses[0]
            
            amacrine_response = amacrine_cell(bipolar_response, dt)
            amacrine_responses.append(amacrine_response)
        
        return {
            'horizontal': horizontal_response,
            'on_bipolar': torch.stack(on_responses) if on_responses else torch.tensor([]),
            'off_bipolar': torch.stack(off_responses) if off_responses else torch.tensor([]),
            'amacrine': torch.stack(amacrine_responses) if amacrine_responses else torch.tensor([])
        }        


def create_retinal_circuit(input_shape: Tuple[int, int],
                          n_channels: int = 3,
                          device: str = 'cpu') -> RetinalNetwork:
    """
    Crée un circuit rétinien standard.
    
    Args:
        input_shape: Forme d'entrée (H, W)
        n_channels: Nombre de canaux de sortie
        device: Device
        
    Returns:
        Réseau rétinien
    """
    # Pour la vision couleur, avoir plusieurs canaux
    n_on_cells = n_channels
    n_off_cells = n_channels
    
    return RetinalNetwork(
        input_shape=input_shape,
        n_on_cells=n_on_cells,
        n_off_cells=n_off_cells,
        device=device
    )
