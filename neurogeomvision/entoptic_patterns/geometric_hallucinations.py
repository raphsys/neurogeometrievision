"""
Module geometric_hallucinations.py - Hallucinations liées à la géométrie de V1
VERSION COMPLÈTEMENT OPTIMISÉE
"""

import torch
import numpy as np
from typing import Tuple, List, Dict
import math
import matplotlib.pyplot as plt


class GeometricHallucinations:
    """
    Hallucinations géométriques basées sur les symétries de V1.
    VERSION OPTIMISÉE : Remplacement des boucles par des opérations vectorisées
    """
    
    def __init__(self,
                 spatial_shape: Tuple[int, int],
                 orientation_bins: int = 12,  # Réduit pour performance
                 device: str = 'cpu'):
        """
        Args:
            spatial_shape: (height, width) du champ visuel
            orientation_bins: Nombre d'orientations discrétisées
            device: 'cpu' ou 'cuda'
        """
        self.spatial_shape = spatial_shape
        self.height, self.width = spatial_shape
        self.orientation_bins = orientation_bins
        self.device = device
        
        # Espace de contact V = R² × S¹
        self.theta_values = torch.linspace(0, 2*math.pi, orientation_bins, device=device)
        
        # Paramètres du modèle
        self.alpha = -0.5    # Taux de décroissance linéaire
        self.beta = 1.0      # Coefficient non-linéaire
        self.mu = 0.1        # Paramètre de bifurcation
        
        # Pré-calcule les noyaux de connectivité
        self.spatial_kernel = self._create_spatial_kernel_fast()
        self.orientation_kernel = self._create_orientation_kernel_fast()
    
    def _create_spatial_kernel_fast(self) -> torch.Tensor:
        """Crée un noyau spatial gaussien - VERSION OPTIMISÉE."""
        kernel_size = 7  # Réduit pour performance
        sigma = 2.0
        half = kernel_size // 2
        
        # Crée les coordonnées avec broadcasting
        coords = torch.arange(-half, half + 1, device=self.device).float()
        x = coords.view(1, -1)  # (1, kernel_size)
        y = coords.view(-1, 1)  # (kernel_size, 1)
        
        # Noyau gaussien 2D en une opération
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        return kernel / kernel.sum()
    
    def _create_orientation_kernel_fast(self) -> torch.Tensor:
        """Crée un noyau de similarité orientationnelle - OPTIMISÉ."""
        theta_i = self.theta_values.view(-1, 1)  # (orientation_bins, 1)
        theta_j = self.theta_values.view(1, -1)  # (1, orientation_bins)
        
        # Différence circulaire
        diff = (theta_j - theta_i + math.pi) % (2 * math.pi) - math.pi
        
        # Similarité cosinus (plus rapide que exponentielle)
        kernel = torch.cos(diff) * 0.5 + 0.5  # Échelle 0-1
        
        return kernel
    
    def apply_connectivity(self, activity: torch.Tensor) -> torch.Tensor:
        """
        Applique la connectivité invariante sous E(2) - VERSION OPTIMISÉE.
        Utilise la convolution séparable pour réduire la complexité.
        """
        h, w, nb = activity.shape
        
        # 1. Convolution spatiale par canal
        ks = self.spatial_kernel.shape[0]
        pad = ks // 2
        
        # Réorganise: (h, w, nb) -> (nb, 1, h, w)
        activity_4d = activity.permute(2, 0, 1).unsqueeze(1)
        kernel_4d = self.spatial_kernel.unsqueeze(0).unsqueeze(0)
        
        # Convolution spatiale (rapide)
        spatial_result = torch.nn.functional.conv2d(
            torch.nn.functional.pad(activity_4d, (pad, pad, pad, pad), mode='reflect'),
            kernel_4d,
            padding=0
        ).squeeze(1)  # (nb, h, w)
        
        # 2. Mixage orientationnel (matrice de poids)
        result = torch.einsum('oi,ihw->ohw', self.orientation_kernel, spatial_result)
        
        # 3. Réorganise: (nb, h, w) -> (h, w, nb)
        result = result.permute(1, 2, 0)
        
        return result
    
    def e2_evolution_step(self, 
                         activity: torch.Tensor,
                         dt: float = 0.1) -> torch.Tensor:
        """
        Un pas d'évolution selon l'équation sur E(2) ⋉ S¹ - OPTIMISÉ.
        """
        # Terme linéaire
        linear = self.alpha * activity
        
        # Terme cubique
        cubic = self.beta * activity**3
        
        # Connectivité
        connectivity = self.mu * self.apply_connectivity(activity)
        
        # Bruit
        noise = torch.randn_like(activity) * 0.01
        
        # Évolution
        new_activity = activity + dt * (linear + cubic + connectivity + noise)
        
        # Normalisation
        new_activity = torch.tanh(new_activity)
        
        return new_activity
    
    def generate_hallucination(self,
                              pattern_type: str = 'pinwheels',
                              n_steps: int = 30) -> torch.Tensor:
        """
        Génère une hallucination géométrique - VERSION OPTIMISÉE.
        """
        # Initialisation
        activity = torch.randn(self.height, self.width, self.orientation_bins, 
                             device=self.device) * 0.1
        
        # Simulation
        for step in range(n_steps):
            activity = self.e2_evolution_step(activity, dt=0.2)
        
        return activity
    
    def project_to_visual_field(self, 
                               activity: torch.Tensor,
                               method: str = 'max') -> torch.Tensor:
        """
        Projette l'activité de l'espace de contact vers le champ visuel.
        """
        if method == 'max':
            visual_field, _ = torch.max(activity, dim=2)
        elif method == 'sum':
            visual_field = torch.sum(activity, dim=2)
        elif method == 'mean':
            visual_field = torch.mean(activity, dim=2)
        else:
            visual_field = torch.mean(activity, dim=2)
        
        # Normalise
        vmin, vmax = visual_field.min(), visual_field.max()
        if vmax - vmin > 1e-6:
            visual_field = (visual_field - vmin) / (vmax - vmin)
        
        return visual_field
        

    def classify_pattern(self, activity: torch.Tensor) -> Dict[str, any]:
        """
        Classe le type de pattern généré.
        """
        visual_mean = self.project_to_visual_field(activity, 'mean')
        variance = visual_mean.var().item()
        mean_intensity = activity.mean().item()
    
        # Classification simple basée sur la variance
        if variance < 0.01:
            pattern_type = 'uniform'
        elif variance < 0.05:
            pattern_type = 'stripes'
        elif variance < 0.1:
            pattern_type = 'hexagons'
        else:
            pattern_type = 'pinwheels'
    
        return {
            'type': pattern_type,
            'variance': variance,
            'intensity': mean_intensity
        }
    
    def visualize_hallucination(self,
                               activity: torch.Tensor,
                               save_path: str = None) -> dict:
        """
        Visualise une hallucination géométrique.
        """
        # Projections
        visual_mean = self.project_to_visual_field(activity, 'mean')
        visual_max = self.project_to_visual_field(activity, 'max')
        
        # Classification - AJOUTEZ CETTE LIGNE
        classification = self.classify_pattern(activity)
        
        # Crée la figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # 1. Moyenne
        im1 = axes[0].imshow(visual_mean.cpu().numpy(), cmap='hot')
        axes[0].set_title("Projection moyenne")
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)
        
        # 2. Maximum
        im2 = axes[1].imshow(visual_max.cpu().numpy(), cmap='hot')
        axes[1].set_title("Projection max")
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046)
        
        # 3. Histogramme
        axes[2].hist(visual_mean.cpu().numpy().flatten(), bins=30, alpha=0.7)
        axes[2].set_title(f"Distribution - {classification['type']}")  # Amélioré
        axes[2].set_xlabel("Activité")
        axes[2].set_ylabel("Fréquence")
        
        plt.suptitle("Hallucinations Géométriques", fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
        
        return {
            'activity': activity,
            'visual_fields': {
                'mean': visual_mean,
                'max': visual_max
            },
            'classification': classification,  # MAINTENANT défini
            'figure': fig
        }
