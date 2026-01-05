"""
Module developmental.py - Apprentissage développemental
Formation des cartes corticales et dominance oculaire
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
import math
import matplotlib.pyplot as plt


class DevelopmentalLearning:
    """
    Apprentissage développemental des cartes corticales.
    """
    
    def __init__(self,
                 cortical_size: Tuple[int, int],
                 input_size: int = 2,  # Par exemple: 2 yeux
                 learning_rate: float = 0.01,
                 device: str = 'cpu'):
        
        self.cortical_size = cortical_size
        self.height, self.width = cortical_size
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.device = device
        
        # Cartes de poids
        self.weights = torch.randn(self.height, self.width, input_size, device=device) * 0.1
        
        # Paramètres développementaux
        self.critical_period = 1000
        self.current_step = 0
    
    def competitive_learning(self,
                           input_pattern: torch.Tensor,
                           neighborhood: float = 2.0) -> torch.Tensor:
        """
        Apprentissage compétitif (Kohonen-like).
        """
        # Trouve le neurone gagnant (plus proche de l'entrée)
        distances = torch.norm(self.weights - input_pattern, dim=2)  # (height, width)
        winner_idx = torch.argmin(distances)
        winner_y = winner_idx // self.width
        winner_x = winner_idx % self.width
        
        # Fonction de voisinage gaussienne
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.height, device=self.device),
            torch.arange(self.width, device=self.device),
            indexing='ij'
        )
        
        dist_sq = (x_coords - winner_x)**2 + (y_coords - winner_y)**2
        neighborhood_mask = torch.exp(-dist_sq / (2 * neighborhood**2))
        
        # Mise à jour des poids
        for i in range(self.input_size):
            delta = self.learning_rate * neighborhood_mask * (input_pattern[i] - self.weights[:, :, i])
            self.weights[:, :, i] += delta
        
        self.current_step += 1
        
        return self.weights
    
    def develop_ocular_dominance(self,
                                n_steps: int = 1000,
                                noise_level: float = 0.1) -> torch.Tensor:
        """
        Développe des colonnes de dominance oculaire.
        """
        for step in range(n_steps):
            # Génère un stimulus d'entrée (biaisé vers un œil)
            if torch.rand(1).item() < 0.5:
                # Stimulus œil gauche dominant
                input_pattern = torch.tensor([1.0, 0.0], device=self.device) + \
                               torch.randn(2, device=self.device) * noise_level
            else:
                # Stimulus œil droit dominant
                input_pattern = torch.tensor([0.0, 1.0], device=self.device) + \
                               torch.randn(2, device=self.device) * noise_level
            
            # Apprentissage compétitif
            neighborhood = 3.0 * math.exp(-step / 500)  # Voisinage décroissant
            self.competitive_learning(input_pattern, neighborhood)
            
            if (step + 1) % 100 == 0:
                print(f"Step {step + 1}/{n_steps}, "
                      f"Voisinage: {neighborhood:.3f}")
        
        return self.weights
    
    def compute_ocular_dominance_index(self) -> torch.Tensor:
        """
        Calcule l'index de dominance oculaire.
        OD = (R - L) / (R + L)
        """
        R = self.weights[:, :, 0]  # Œil droit
        L = self.weights[:, :, 1]  # Œil gauche
        
        od_index = (R - L) / (R + L + 1e-8)
        return od_index
    
    def visualize_development(self, save_path: str = None):
        """Visualise le développement."""
        od_index = self.compute_ocular_dominance_index()
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Œil droit
        im1 = axes[0].imshow(self.weights[:, :, 0].cpu().numpy(), cmap='hot')
        axes[0].set_title("Poids œil droit")
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])
        
        # Œil gauche
        im2 = axes[1].imshow(self.weights[:, :, 1].cpu().numpy(), cmap='hot')
        axes[1].set_title("Poids œil gauche")
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])
        
        # Index de dominance
        im3 = axes[2].imshow(od_index.cpu().numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
        axes[2].set_title("Index de dominance oculaire")
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2])
        
        plt.suptitle(f"Développement de dominance oculaire (step {self.current_step})", fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
        
        return fig


class OcularDominance(DevelopmentalLearning):
    """
    Modèle spécifique de dominance oculaire.
    """
    
    def __init__(self, cortical_size: Tuple[int, int], **kwargs):
        super().__init__(cortical_size, input_size=2, **kwargs)
    
    def correlation_based_learning(self,
                                  left_input: torch.Tensor,
                                  right_input: torch.Tensor,
                                  correlation_strength: float = 0.5):
        """
        Apprentissage basé sur la corrélation entre les yeux.
        """
        # Calcule la corrélation locale
        correlation = torch.dot(left_input.flatten(), right_input.flatten()) / \
                     (torch.norm(left_input) * torch.norm(right_input) + 1e-8)
        
        # Stimulus combiné pondéré par la corrélation
        combined = correlation_strength * correlation * left_input + \
                  (1 - correlation_strength) * right_input
        
        # Normalise
        combined = combined / (torch.norm(combined) + 1e-8)
        
        # Apprentissage
        self.competitive_learning(combined)
        
        return correlation.item()
    
    def monocular_deprivation(self,
                             deprived_eye: str = 'left',
                             deprivation_strength: float = 0.8,
                             n_steps: int = 500):
        """
        Simule une privation monoculaire.
        """
        original_weights = self.weights.clone()
        
        for step in range(n_steps):
            if deprived_eye == 'left':
                # Stimulus seulement œil droit
                input_pattern = torch.tensor([1.0, 0.0], device=self.device)
            else:
                # Stimulus seulement œil gauche
                input_pattern = torch.tensor([0.0, 1.0], device=self.device)
            
            # Apprentissage avec voisinage réduit
            neighborhood = 1.0 * math.exp(-step / 200)
            self.competitive_learning(input_pattern, neighborhood)
            
            # Affiche la progression
            if (step + 1) % 50 == 0:
                od_change = torch.mean(torch.abs(self.weights - original_weights)).item()
                print(f"Privation step {step + 1}/{n_steps}, "
                      f"Changement: {od_change:.4f}")
        
        return self.weights
