"""
Module retinal_maps.py - Cartes rétinotopiques et magnifications corticales
Projections rétine -> cortex visuel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import math
import matplotlib.pyplot as plt


class RetinotopicMap(nn.Module):
    """
    Carte rétinotopique : projection log-polaires de la rétine au cortex.
    """
    
    def __init__(self,
                 retinal_shape: Tuple[int, int],
                 cortical_shape: Tuple[int, int],
                 magnification_factor: float = 10.0,
                 foveal_scale: float = 1.0,
                 peripheral_scale: float = 0.1,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.retinal_shape = retinal_shape  # (H, W)
        self.cortical_shape = cortical_shape  # (H, W)
        self.magnification_factor = magnification_factor
        self.foveal_scale = foveal_scale
        self.peripheral_scale = peripheral_scale
        self.device = device
        
        # Créer la carte de transformation
        self.retina_to_cortex_map, self.cortex_to_retina_map = self._create_maps()
        
        # Facteur de magnification en fonction de l'excenticité
        self.magnification_map = self._create_magnification_map()
    
    def _create_maps(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Crée les cartes de transformation rétine<->cortex."""
        retinal_h, retinal_w = self.retinal_shape
        cortical_h, cortical_w = self.cortical_shape
        
        # Coordonnées corticales normalisées
        cortex_y, cortex_x = torch.meshgrid(
            torch.linspace(-1, 1, cortical_h, device=self.device),
            torch.linspace(-1, 1, cortical_w, device=self.device),
            indexing='ij'
        )
        
        # Transformation log-polaires (simplifiée)
        # r = exp(ρ * cos(θ)), θ = angle
        r_cortex = torch.sqrt(cortex_x**2 + cortex_y**2)
        theta_cortex = torch.atan2(cortex_y, cortex_x)
        
        # Application de la magnification (log compression)
        # r_retina = log(1 + r_cortex * self.magnification_factor)
        r_retina = r_cortex * self.magnification_factor
        
        # Coordonnées polaires -> cartésiennes pour la rétine
        retina_x = r_retina * torch.cos(theta_cortex)
        retina_y = r_retina * torch.sin(theta_cortex)
        
        # Normaliser pour l'espace rétinien [-1, 1]
        max_r = torch.max(torch.sqrt(retina_x**2 + retina_y**2))
        if max_r > 0:
            retina_x = retina_x / max_r
            retina_y = retina_y / max_r
        
        # Map rétine -> cortex (pour l'interpolation)
        retina_to_cortex = torch.stack([cortex_y, cortex_x], dim=-1)
        
        # Map cortex -> rétine (pour la transformation inverse)
        cortex_to_retina = torch.stack([retina_y, retina_x], dim=-1)
        
        return retina_to_cortex, cortex_to_retina
    
    def _create_magnification_map(self) -> torch.Tensor:
        """Crée une carte de magnification corticale."""
        cortical_h, cortical_w = self.cortical_shape
        
        # Coordonnées corticales
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, cortical_h, device=self.device),
            torch.linspace(-1, 1, cortical_w, device=self.device),
            indexing='ij'
        )
        
        # Distance du centre (fovéa)
        r = torch.sqrt(x**2 + y**2)
        
        # Magnification décroissante avec l'excenticité
        # M(r) = M0 / (1 + α * r)
        alpha = 2.0  # Taux de décroissance
        magnification = self.magnification_factor / (1.0 + alpha * r)
        
        # Normaliser
        magnification = magnification / magnification.max()
        
        return magnification
    
    def forward(self, retinal_image: torch.Tensor, mode: str = 'retina_to_cortex') -> torch.Tensor:
        """
        Transforme une image entre espaces rétinien et cortical.
    
        Args:
            retinal_image: Image d'entrée
            mode: 'retina_to_cortex' ou 'cortex_to_retina'
        
        Returns:
            Image transformée
        """
        if len(retinal_image.shape) == 2:
            retinal_image = retinal_image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif len(retinal_image.shape) == 3:
            retinal_image = retinal_image.unsqueeze(1)  # (B, 1, H, W)
    
        batch_size, channels, retinal_h, retinal_w = retinal_image.shape
    
        if mode == 'retina_to_cortex':
            # Rétine -> Cortex
            # Préparer la grille pour grid_sample
            cortical_h, cortical_w = self.cortical_shape
        
            # Normaliser les coordonnées de la carte
            # grid_sample attend des coordonnées dans [-1, 1]
            grid = self.cortex_to_retina_map.unsqueeze(0)  # (1, H, W, 2)  # CORRECT - garder tel quel
            grid = grid.repeat(batch_size, 1, 1, 1)  # (B, H, W, 2)
        
            # Application de la transformation
            cortical_image = F.grid_sample(
                retinal_image,
                grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )
        
            # Appliquer la magnification
            magnification = self.magnification_map.unsqueeze(0).unsqueeze(0)
            cortical_image = cortical_image * magnification
        
            return cortical_image
    
        else:  # cortex_to_retina
            # Cortex -> Rétine
            cortical_h, cortical_w = self.cortical_shape
        
            # Inverser la magnification
            magnification = self.magnification_map.unsqueeze(0).unsqueeze(0)
            de_magnified = retinal_image / (magnification + 1e-8)
        
            # Transformation inverse
            grid = self.retina_to_cortex_map.unsqueeze(0)  # (1, H, W, 2)  # CORRECT - garder tel quel
            grid = grid.repeat(batch_size, 1, 1, 1)  # (B, H, W, 2)
        
            retinal_reconstructed = F.grid_sample(
                de_magnified,
                grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )
        
            return retinal_reconstructed
            
            
    def get_cortical_coordinates(self, retinal_coords: torch.Tensor) -> torch.Tensor:
        """
        Convertit des coordonnées rétiniennes en coordonnées corticales.
        
        Args:
            retinal_coords: Coordonnées (N, 2) normalisées [-1, 1]
            
        Returns:
            Coordonnées corticales (N, 2)
        """
        # Pour l'instant, transformation simplifiée
        # En réalité, c'est une transformation complexe
        retinal_x, retinal_y = retinal_coords[:, 0], retinal_coords[:, 1]
        
        # Distance rétinienne du centre
        r_retina = torch.sqrt(retinal_x**2 + retinal_y**2)
        theta_retina = torch.atan2(retinal_y, retinal_x)
        
        # Transformation inverse de log-polaires
        # r_cortex = (exp(r_retina) - 1) / self.magnification_factor
        r_cortex = r_retina / self.magnification_factor
        
        cortical_x = r_cortex * torch.cos(theta_retina)
        cortical_y = r_cortex * torch.sin(theta_retina)
        
        return torch.stack([cortical_x, cortical_y], dim=-1)


class CorticalMagnification(nn.Module):
    """
    Modèle de magnification corticale.
    Représente l'expansion disproportionnée de la fovéa dans le cortex.
    """
    
    def __init__(self,
                 max_magnification: float = 50.0,
                 magnification_slope: float = 0.8,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.max_magnification = max_magnification
        self.magnification_slope = magnification_slope
        self.device = device
    
    def magnification_at_eccentricity(self, eccentricity: torch.Tensor) -> torch.Tensor:
        """
        Calcule la magnification à une excentricité donnée.
        
        Args:
            eccentricity: Excentricité en degrés visuels
            
        Returns:
            Facteur de magnification
        """
        # Formule classique: M(e) = M0 / (1 + e/E)
        # où M0 est la magnification fovéale, E est une constante
        E = 2.5  # degrés (valeur typique)
        
        magnification = self.max_magnification / (1.0 + eccentricity / E)
        
        return magnification
    
    def cortical_distance(self, retinal_distance: torch.Tensor,
                         eccentricity: torch.Tensor) -> torch.Tensor:
        """
        Convertit une distance rétinienne en distance corticale.
        
        Args:
            retinal_distance: Distance sur la rétine (degrés)
            eccentricity: Excentricité du point de référence
            
        Returns:
            Distance sur le cortex (mm)
        """
        # Approximation: d_cortex = M(e) * d_retina
        magnification = self.magnification_at_eccentricity(eccentricity)
        cortical_distance = magnification * retinal_distance
        
        return cortical_distance
    
    def forward(self, retinal_positions: torch.Tensor) -> torch.Tensor:
        """
        Transforme des positions rétiniennes en positions corticales.
        
        Args:
            retinal_positions: Positions (N, 2) en degrés visuels
            
        Returns:
            Positions corticales (N, 2) en mm
        """
        # Coordonnées polaires
        x, y = retinal_positions[:, 0], retinal_positions[:, 1]
        eccentricity = torch.sqrt(x**2 + y**2)
        angle = torch.atan2(y, x)
        
        # Magnification locale
        magnification = self.magnification_at_eccentricity(eccentricity)
        
        # Transformation (simplifiée)
        # Dans le cortex, les angles sont préservés, les rayons sont compressés
        cortical_r = torch.log(1.0 + eccentricity) * self.magnification_slope
        
        cortical_x = cortical_r * torch.cos(angle)
        cortical_y = cortical_r * torch.sin(angle)
        
        return torch.stack([cortical_x, cortical_y], dim=-1)


def create_retinotopic_mapping(retinal_resolution: Tuple[int, int] = (100, 100),
                              cortical_resolution: Tuple[int, int] = (200, 200),
                              magnification: float = 15.0,
                              device: str = 'cpu') -> RetinotopicMap:
    """
    Crée une carte rétinotopique standard.
    
    Args:
        retinal_resolution: Résolution rétinienne (H, W)
        cortical_resolution: Résolution corticale (H, W)
        magnification: Facteur de magnification
        device: Device
        
    Returns:
        Carte rétinotopique
    """
    return RetinotopicMap(
        retinal_shape=retinal_resolution,
        cortical_shape=cortical_resolution,
        magnification_factor=magnification,
        device=device
    )


def visualize_retinal_map(retinal_map: RetinotopicMap,
                         retinal_image: Optional[torch.Tensor] = None,
                         save_path: Optional[str] = None):
    """
    Visualise une carte rétinotopique.
    
    Args:
        retinal_map: Carte à visualiser
        retinal_image: Image optionnelle à transformer
        save_path: Chemin de sauvegarde
    """
    fig, axes = plt.subplots(1, 3 if retinal_image is not None else 2, figsize=(15, 5))
    
    # 1. Carte de transformation
    retina_to_cortex = retinal_map.retina_to_cortex_map.detach().cpu()
    
    axes[0].imshow(retina_to_cortex[..., 0], cmap='viridis', aspect='auto')
    axes[0].set_title('Transformation Rétine -> Cortex (Y)')
    axes[0].set_xlabel('Cortex X')
    axes[0].set_ylabel('Cortex Y')
    plt.colorbar(axes[0].imshow(retina_to_cortex[..., 0], cmap='viridis'), ax=axes[0])
    
    # 2. Carte de magnification
    magnification = retinal_map.magnification_map.detach().cpu()
    
    im = axes[1].imshow(magnification, cmap='hot', aspect='auto')
    axes[1].set_title('Magnification Corticale')
    axes[1].set_xlabel('Cortex X')
    axes[1].set_ylabel('Cortex Y')
    plt.colorbar(im, ax=axes[1])
    
    # 3. Transformation d'image (si fournie)
    if retinal_image is not None:
        cortical_image = retinal_map(retinal_image, mode='retina_to_cortex')
        cortical_image = cortical_image.detach().cpu()
        
        if len(cortical_image.shape) == 2:
            axes[2].imshow(cortical_image, cmap='gray', aspect='auto')
        else:
            axes[2].imshow(cortical_image.permute(1, 2, 0))
        
        axes[2].set_title('Image Transformée (Cortex)')
        axes[2].set_xlabel('Cortex X')
        axes[2].set_ylabel('Cortex Y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
