"""
Module integration.py - Intégration avec les modules existants
Apprentissage des filtres Gabor et champs d'association
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
import math
import matplotlib.pyplot as plt

# Import des modules existants
from ..v1_simple_cells.gabor_filters import GaborFilterBank
from ..association_field.field_models import AssociationField
from .hebbian import HebbianLearning
from .oja import OjaLearning
from .natural_statistics import NaturalStatistics, ICA_Learning


class PlasticityIntegrator:
    """
    Intégrateur de plasticité pour tous les modules.
    """
    
    def __init__(self,
                 image_size: Tuple[int, int],
                 device: str = 'cpu'):
        
        self.image_size = image_size
        self.height, self.width = image_size
        self.device = device
        
        # Modules
        self.gabor_bank = None
        self.association_field = None
        self.hebbian = None
        self.ica = None
        
    def learn_gabor_filters_from_natural_images(self,
                                               images: List[torch.Tensor],
                                               n_orientations: int = 8,
                                               n_epochs: int = 100) -> GaborFilterBank:
        """
        Apprend les filtres de Gabor à partir d'images naturelles.
        """
        # Extrait les patches
        patch_size = 16
        natural_stats = NaturalStatistics(patch_size, self.device)
        
        all_patches = []
        for image in images[:10]:  # Limite à 10 images pour la vitesse
            patches = natural_stats.extract_patches(image, n_patches=500)
            all_patches.append(patches)
        
        patches_tensor = torch.cat(all_patches, dim=0)
        
        # ICA pour apprendre les filtres
        ica = ICA_Learning(input_dim=patch_size*patch_size,
                          n_components=n_orientations * 3,  # orientations × échelles
                          device=self.device)
        
        filters_flat = ica.learn_gabor_filters(patches_tensor, n_epochs=n_epochs)
        
        # Crée le banc de filtres Gabor
        self.gabor_bank = GaborFilterBank(
            img_size=self.image_size,
            n_orientations=n_orientations,
            device=self.device
        )
        
        print(f"✓ {len(filters_flat)} filtres appris à partir de {len(images)} images")
        
        return self.gabor_bank
    
    def learn_association_field_hebbian(self,
                                       n_iterations: int = 1000) -> AssociationField:
        """
        Apprend le champ d'association par plasticité hebbienne.
        """
        # Crée le champ d'association
        self.association_field = AssociationField(
            spatial_shape=self.image_size,
            orientation_bins=12,
            device=self.device
        )
        
        # Apprentissage hebbien des connexions
        connection_strengths = torch.zeros(12, 12, device=self.device)  # orientations
        
        for _ in range(n_iterations):
            # Stimulus: deux orientations corrélées
            theta1 = torch.rand(1).item() * math.pi
            theta2 = theta1 + torch.randn(1).item() * 0.2  # Légère variation
            
            # Activation hebbienne
            activity1 = torch.cos(torch.arange(12, device=self.device) * math.pi/12 - theta1)
            activity2 = torch.cos(torch.arange(12, device=self.device) * math.pi/12 - theta2)
            
            # Mise à jour hebbienne
            delta = torch.outer(activity1, activity2)
            connection_strengths += 0.01 * delta
            
            # Normalisation
            connection_strengths = connection_strengths / torch.norm(connection_strengths)
        
        print(f"✓ Champ d'association appris après {n_iterations} itérations")
        
        return self.association_field
    
    def develop_orientation_columns(self,
                                   input_size: Tuple[int, int] = (50, 50),
                                   n_steps: int = 5000) -> torch.Tensor:
        """
        Développe des colonnes d'orientation par apprentissage compétitif.
        """
        height, width = input_size
        
        # Cartes de poids pour chaque orientation
        n_orientations = 8
        weights = torch.randn(height, width, n_orientations, device=self.device) * 0.1
        
        for step in range(n_steps):
            # Stimulus d'orientation aléatoire
            theta = torch.rand(1).item() * math.pi
            orientation_profile = torch.cos(torch.arange(n_orientations, device=self.device) * 
                                          math.pi/n_orientations - theta)
            
            # Position aléatoire
            y = torch.randint(0, height, (1,)).item()
            x = torch.randint(0, width, (1,)).item()
            
            # Voisinage gaussien
            neighborhood = 3.0 * math.exp(-step / 2000)
            y_coords, x_coords = torch.meshgrid(
                torch.arange(height, device=self.device),
                torch.arange(width, device=self.device),
                indexing='ij'
            )
            
            dist_sq = (x_coords - x)**2 + (y_coords - y)**2
            neighborhood_mask = torch.exp(-dist_sq / (2 * neighborhood**2))
            
            # Mise à jour compétitive
            for o in range(n_orientations):
                activation = orientation_profile[o]
                weights[:, :, o] += 0.01 * neighborhood_mask * activation
            
            # Normalisation
            norms = torch.norm(weights, dim=2, keepdim=True)
            weights = weights / torch.clamp(norms, min=1e-8)
            
            if (step + 1) % 1000 == 0:
                print(f"Step {step + 1}/{n_steps}, Voisinage: {neighborhood:.3f}")
        
        # Carte d'orientation dominante
        dominant_orientation = torch.argmax(weights, dim=2)
        
        return dominant_orientation
    
    def visualize_learned_features(self,
                                 save_path: str = None):
        """Visualise les caractéristiques apprises."""
        if self.gabor_bank is None or self.association_field is None:
            print("❌ Modules non initialisés")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        # 1. Filtres Gabor appris
        if hasattr(self.gabor_bank, 'filters'):
            n_to_show = min(6, len(self.gabor_bank.filters))
            for i in range(n_to_show):
                ax = axes[i // 3, i % 3]
                filter_img = self.gabor_bank.filters[i].cpu().numpy()
                im = ax.imshow(filter_img, cmap='RdBu_r')
                ax.set_title(f"Filtre {i+1}")
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.suptitle("Caractéristiques apprises par plasticité", fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
        
        return fig


# Fonctions utilitaires
def learn_gabor_filters(images: List[torch.Tensor],
                       image_size: Tuple[int, int] = (64, 64),
                       n_orientations: int = 8,
                       n_epochs: int = 100,
                       device: str = 'cpu') -> GaborFilterBank:
    """
    Fonction utilitaire pour apprendre des filtres Gabor.
    """
    integrator = PlasticityIntegrator(image_size, device)
    return integrator.learn_gabor_filters_from_natural_images(
        images, n_orientations, n_epochs
    )


def learn_association_field(image_size: Tuple[int, int] = (64, 64),
                           n_iterations: int = 1000,
                           device: str = 'cpu') -> AssociationField:
    """
    Fonction utilitaire pour apprendre un champ d'association.
    """
    integrator = PlasticityIntegrator(image_size, device)
    return integrator.learn_association_field_hebbian(n_iterations)
