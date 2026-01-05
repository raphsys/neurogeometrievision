"""
Module orientation.py - Sélectivité à l'orientation dans V1
Cartes d'orientation, filtres de Gabor, hypercolonnes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional
import math


class OrientationMap(nn.Module):
    """
    Carte d'orientation - Code les orientations préférées sur une surface corticale.
    """
    
    def __init__(self,
                 map_shape: Tuple[int, int],
                 pinwheel_centers: int = 4,
                 orientation_range: Tuple[float, float] = (0, math.pi),
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.map_shape = map_shape
        self.pinwheel_centers = pinwheel_centers
        self.orientation_range = orientation_range
        self.device = device
        
        # Créer la carte d'orientation (champ vectoriel complexe)
        self.orientation_field = self._create_orientation_field()
        
        # Créer les filtres Gabor pour toutes les positions/orientations
        self.gabor_filters = self._create_gabor_filters()
    
    def _create_orientation_field(self) -> torch.Tensor:
        """Crée un champ d'orientation avec des tourbillons (pinwheels)."""
        height, width = self.map_shape
        
        # Grille de coordonnées
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=self.device),
            torch.linspace(-1, 1, width, device=self.device),
            indexing='ij'
        )
        
        # Initialiser le champ complexe
        z_field = torch.zeros(height, width, dtype=torch.complex64, device=self.device)
        
        # Ajouter des tourbillons (vortex)
        for _ in range(self.pinwheel_centers):
            # Position aléatoire du tourbillon
            cx = torch.rand(1, device=self.device) * 2 - 1
            cy = torch.rand(1, device=self.device) * 2 - 1
            
            # Vecteurs depuis le centre
            dx = x - cx
            dy = y - cy
            
            # Angle polaire (argument complexe)
            angle = torch.atan2(dy, dx)
            
            # Tourbillon: phase = ±angle
            vortex = torch.exp(1j * angle * torch.sign(torch.randn(1, device=self.device)))
            
            # Poids décroissant avec la distance
            distance = torch.sqrt(dx**2 + dy**2 + 1e-8)
            weight = torch.exp(-distance**2 / 0.3**2)
            
            z_field += weight * vortex
        
        # Normaliser et obtenir l'orientation
        orientation = 0.5 * torch.angle(z_field)  # Orientation en radians
        orientation = torch.fmod(orientation + math.pi, math.pi)  # [0, π]
        
        return orientation
    
    def _create_gabor_filters(self, filter_size: int = 15) -> torch.Tensor:
        """Crée une banque de filtres Gabor pour toutes les positions/orientations."""
        height, width = self.map_shape
        n_filters = height * width
        
        filters = torch.zeros(n_filters, 1, filter_size, filter_size, device=self.device)
        
        # Paramètres Gabor
        sigma_x = 2.0
        sigma_y = 4.0
        spatial_freq = 0.1
        
        center = filter_size // 2
        
        # Grille pour le filtre
        y_filt, x_filt = torch.meshgrid(
            torch.arange(filter_size, device=self.device) - center,
            torch.arange(filter_size, device=self.device) - center,
            indexing='ij'
        )
        
        for i in range(height):
            for j in range(width):
                idx = i * width + j
                orientation = self.orientation_field[i, j]
                
                # Rotation des coordonnées
                x_rot = x_filt * math.cos(orientation) + y_filt * math.sin(orientation)
                y_rot = -x_filt * math.sin(orientation) + y_filt * math.cos(orientation)
                
                # Enveloppe gaussienne
                gaussian = torch.exp(-0.5 * (x_rot**2 / sigma_x**2 + y_rot**2 / sigma_y**2))
                
                # Porteuse sinusoïdale (phase 0 pour cellule simple)
                carrier = torch.cos(2 * math.pi * spatial_freq * x_rot)
                
                # Filtre Gabor
                gabor = gaussian * carrier
                
                # Normalisation à somme nulle
                gabor = gabor - gabor.mean()
                gabor = gabor / (gabor.abs().sum() + 1e-8)
                
                filters[idx, 0, :, :] = gabor
        
        return filters
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Applique les filtres d'orientation à l'image.
        
        Args:
            x: Image d'entrée (H, W) ou (B, H, W)
            
        Returns:
            Réponses, carte d'orientation, carte de magnitude
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif len(x.shape) == 3:
            x = x.unsqueeze(1)  # (B, 1, H, W)
        
        batch_size, _, height, width = x.shape
        
        # Convolution avec tous les filtres Gabor
        # Note: Ceci est coûteux en calcul, en pratique on utilise des convolutions locales
        map_h, map_w = self.map_shape
        
        # Pour l'efficacité, on pourrait n'appliquer que les filtres locaux
        # Ici version simplifiée: ré-échantillonner l'image à la taille de la carte
        if height != map_h or width != map_w:
            x_resized = F.interpolate(x, size=(map_h, map_w), mode='bilinear')
        else:
            x_resized = x
        
        # Convolution avec un sous-ensemble de filtres (pour l'efficacité)
        # En pratique, on utiliserait des convolutions groupées ou locales
        responses = torch.zeros(batch_size, map_h * map_w, map_h, map_w, device=self.device)
        
        # Version simplifiée: convolution point par point
        for b in range(batch_size):
            for i in range(map_h):
                for j in range(map_w):
                    idx = i * map_w + j
                    filter_idx = min(idx, self.gabor_filters.shape[0] - 1)
                    
                    # Extraire une petite région autour du point
                    patch_size = 15
                    h_start = max(0, i - patch_size // 2)
                    h_end = min(map_h, i + patch_size // 2 + 1)
                    w_start = max(0, j - patch_size // 2)
                    w_end = min(map_w, j + patch_size // 2 + 1)
                    
                    if h_end > h_start and w_end > w_start:
                        patch = x_resized[b:b+1, :, h_start:h_end, w_start:w_end]
                        filter_patch = self.gabor_filters[filter_idx:filter_idx+1, :, :, :]
                        
                        # Ajuster la taille du filtre si nécessaire
                        if patch.shape[-2:] != filter_patch.shape[-2:]:
                            filter_patch = F.interpolate(filter_patch, size=patch.shape[-2:], mode='bilinear')
                        
                        # Produit scalaire (corrélation)
                        response = (patch * filter_patch).sum()
                        responses[b, idx, i, j] = response
        
        # Reformer en carte 2D
        responses_2d = responses.view(batch_size, map_h, map_w, map_h, map_w)
        
        # Prendre la réponse maximale à chaque position (sélectivité)
        max_response, max_idx = responses_2d.max(dim=2)  # (B, map_h, map_w, map_h, map_w) -> (B, map_h, map_w)
        
        # Obtenir l'orientation correspondante
        orientation_map = torch.zeros(batch_size, map_h, map_w, device=self.device)
        for b in range(batch_size):
            for i in range(map_h):
                for j in range(map_w):
                    idx_flat = max_idx[b, i, j]
                    i_orient = idx_flat // map_w
                    j_orient = idx_flat % map_w
                    orientation_map[b, i, j] = self.orientation_field[i_orient, j_orient]
        
        return {
            'responses': responses,  # Toutes les réponses
            'max_responses': max_response,  # Réponses maximales
            'orientation_map': orientation_map,  # Carte d'orientation
            'orientation_field': self.orientation_field.unsqueeze(0).expand(batch_size, -1, -1)
        }


class OrientationSelectivity(nn.Module):
    """
    Modèle de sélectivité à l'orientation basé sur des filtres Gabor.
    """
    
    def __init__(self,
                 n_orientations: int = 8,
                 spatial_freqs: List[float] = None,
                 phases: List[float] = None,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.n_orientations = n_orientations
        self.device = device
        
        # Paramètres des filtres
        self.spatial_freqs = spatial_freqs or [0.05, 0.1, 0.15]
        self.phases = phases or [0, math.pi/2]
        
        # Créer les filtres Gabor
        self.filters = self._create_gabor_bank()
        
        # Pooling pour les cellules complexes
        self.complex_pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    
    def _create_gabor_bank(self, filter_size: int = 15) -> nn.Parameter:
        """Crée une banque de filtres Gabor."""
        filters = []
        
        for orientation in torch.linspace(0, math.pi, self.n_orientations + 1)[:self.n_orientations]:
            for spatial_freq in self.spatial_freqs:
                for phase in self.phases:
                    gabor = self._create_gabor_filter(
                        orientation=orientation.item(),
                        spatial_freq=spatial_freq,
                        phase=phase,
                        size=filter_size
                    )
                    filters.append(gabor)
        
        # Convertir en Parameter
        filters_tensor = torch.stack(filters, dim=0)  # (n_filters, 1, H, W)
        return nn.Parameter(filters_tensor, requires_grad=False)
    
    def _create_gabor_filter(self,
                            orientation: float,
                            spatial_freq: float,
                            phase: float,
                            size: int) -> torch.Tensor:
        """Crée un filtre Gabor individuel."""
        center = size // 2
        
        # Grille
        y, x = torch.meshgrid(
            torch.arange(size, device=self.device) - center,
            torch.arange(size, device=self.device) - center,
            indexing='ij'
        )
        
        # Rotation
        x_rot = x * math.cos(orientation) + y * math.sin(orientation)
        y_rot = -x * math.sin(orientation) + y * math.cos(orientation)
        
        # Enveloppe gaussienne
        sigma_x = size / 4.0
        sigma_y = size / 2.0
        gaussian = torch.exp(-0.5 * (x_rot**2 / sigma_x**2 + y_rot**2 / sigma_y**2))
        
        # Porteuse sinusoïdale
        carrier = torch.cos(2 * math.pi * spatial_freq * x_rot + phase)
        
        # Filtre Gabor
        gabor = gaussian * carrier
        
        # Normalisation à somme nulle
        gabor = gabor - gabor.mean()
        gabor = gabor / (gabor.abs().sum() + 1e-8)
        
        return gabor.unsqueeze(0)  # (1, H, W)
    
    def forward(self, x: torch.Tensor, cell_type: str = 'simple') -> Dict[str, torch.Tensor]:
        """
        Calcule les réponses d'orientation.
        
        Args:
            x: Image d'entrée
            cell_type: 'simple' ou 'complex'
            
        Returns:
            Réponses d'orientation
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif len(x.shape) == 3:
            if x.shape[0] <= 3:
                x = x.unsqueeze(0)  # (1, C, H, W)
            else:
                x = x.unsqueeze(1)  # (B, 1, H, W)
        
        batch_size, channels, height, width = x.shape
        
        # Si multi-canal, prendre la moyenne
        if channels > 1:
            x = x.mean(dim=1, keepdim=True)
        
        # Convolution avec tous les filtres - VERSION CORRIGÉE
        n_filters = self.filters.shape[0]
        padding = self.filters.shape[-1] // 2
        
        # Préparer l'entrée pour la convolution groupée
        # Au lieu d'expandre maladroitement, utilisons une convolution normale
        responses = []
        
        # Version optimisée: convolution batch avec tous les filtres
        if batch_size == 1:
            # Cas simple: une seule image
            x_repeated = x.repeat(1, n_filters, 1, 1)  # (1, n_filters, H, W)
            filters_all = self.filters  # (n_filters, 1, Hf, Wf)
            
            # Convolution depthwise
            response = F.conv2d(x_repeated, filters_all, padding=padding, groups=n_filters)
            responses = response.unsqueeze(0)  # (1, n_filters, H, W)
        else:
            # Batch multiple: traiter chaque élément séparément
            batch_responses = []
            for b in range(batch_size):
                x_batch = x[b:b+1]  # (1, 1, H, W)
                x_repeated = x_batch.repeat(1, n_filters, 1, 1)  # (1, n_filters, H, W)
                response = F.conv2d(x_repeated, self.filters, padding=padding, groups=n_filters)
                batch_responses.append(response)
            responses = torch.cat(batch_responses, dim=0)  # (B, n_filters, H, W)
        
        if cell_type == 'simple':
            # Rectification pour cellules simples
            responses = F.relu(responses)
            
            # Regrouper par orientation
            n_filters_per_orientation = len(self.spatial_freqs) * len(self.phases)
            responses_by_orientation = responses.view(
                batch_size, self.n_orientations, n_filters_per_orientation, height, width
            )
            
            # Prendre le maximum sur les fréquences spatiales et phases
            orientation_responses, _ = responses_by_orientation.max(dim=2)  # (B, n_orientations, H, W)
            
            # Orientation préférée par pixel
            preferred_orientation, orientation_idx = orientation_responses.max(dim=1)  # (B, H, W)
            
            # Convertir l'index en angle
            orientation_angles = torch.linspace(0, math.pi, self.n_orientations, device=self.device)
            orientation_map = orientation_angles[orientation_idx]
            
            return {
                'responses': orientation_responses,
                'orientation_map': orientation_map,
                'strength_map': preferred_orientation,
                'cell_type': 'simple'
            }
        
        else:  # cell_type == 'complex'
            # Modèle énergie pour cellules complexes
            # Regrouper les paires de phase (cos/sin)
            responses_reshaped = responses.view(
                batch_size, self.n_orientations, len(self.spatial_freqs), len(self.phases), height, width
            )
            
            # Combiner les phases pour obtenir l'énergie (cos^2 + sin^2)
            if len(self.phases) == 2:
                energy = responses_reshaped[:, :, :, 0, :, :]**2 + responses_reshaped[:, :, :, 1, :, :]**2
            else:
                energy = responses_reshaped.sum(dim=3)**2
            
            # Pooling sur les fréquences spatiales
            energy_pooled = energy.mean(dim=2)  # (B, n_orientations, H, W)
            
            # Orientation préférée
            preferred_energy, orientation_idx = energy_pooled.max(dim=1)
            
            # Convertir l'index en angle
            orientation_angles = torch.linspace(0, math.pi, self.n_orientations, device=self.device)
            orientation_map = orientation_angles[orientation_idx]
            
            return {
                'energy': energy_pooled,
                'orientation_map': orientation_map,
                'energy_map': preferred_energy,
                'cell_type': 'complex'
            }

def create_orientation_filters(n_orientations: int = 8,
                              filter_size: int = 15,
                              device: str = 'cpu') -> torch.Tensor:
    """
    Crée une banque de filtres d'orientation.
    
    Args:
        n_orientations: Nombre d'orientations
        filter_size: Taille des filtres
        device: Device
        
    Returns:
        Tensor des filtres (n_orientations, 1, H, W)
    """
    filters = []
    center = filter_size // 2
    
    # Grille
    y, x = torch.meshgrid(
        torch.arange(filter_size, device=device) - center,
        torch.arange(filter_size, device=device) - center,
        indexing='ij'
    )
    
    for orientation in torch.linspace(0, math.pi, n_orientations + 1)[:n_orientations]:
        # Paramètres Gabor
        sigma_x = filter_size / 4.0
        sigma_y = filter_size / 2.0
        spatial_freq = 0.1
        
        # Rotation
        x_rot = x * math.cos(orientation) + y * math.sin(orientation)
        y_rot = -x * math.sin(orientation) + y * math.cos(orientation)
        
        # Enveloppe gaussienne
        gaussian = torch.exp(-0.5 * (x_rot**2 / sigma_x**2 + y_rot**2 / sigma_y**2))
        
        # Porteuse sinusoïdale
        carrier = torch.cos(2 * math.pi * spatial_freq * x_rot)
        
        # Filtre Gabor
        gabor = gaussian * carrier
        
        # Normalisation
        gabor = gabor - gabor.mean()
        gabor = gabor / (gabor.abs().sum() + 1e-8)
        
        filters.append(gabor.unsqueeze(0))
    
    return torch.stack(filters, dim=0)  # (n_orientations, 1, H, W)


def extract_orientation_features(image: torch.Tensor,
                                orientation_filters: torch.Tensor,
                                pooling: str = 'max') -> Dict[str, torch.Tensor]:
    """
    Extrait les caractéristiques d'orientation d'une image.
    
    Args:
        image: Image d'entrée (H, W) ou (C, H, W)
        orientation_filters: Filtres d'orientation
        pooling: Type de pooling ('max', 'mean', 'energy')
        
    Returns:
        Caractéristiques d'orientation
    """
    if len(image.shape) == 2:
        image = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif len(image.shape) == 3:
        image = image.unsqueeze(0)  # (1, C, H, W)
    
    batch_size, channels, height, width = image.shape
    
    # Si multi-canal, prendre la moyenne
    if channels > 1:
        image = image.mean(dim=1, keepdim=True)
    
    n_filters = orientation_filters.shape[0]
    padding = orientation_filters.shape[-1] // 2
    
    # Convolution avec tous les filtres
    responses = F.conv2d(image, orientation_filters, padding=padding)
    
    # Rectification
    responses = F.relu(responses)
    
    if pooling == 'max':
        # Orientation préférée (maximum)
        max_response, orientation_idx = responses.max(dim=1)
        orientation_map = orientation_idx.float() / (n_filters - 1) * math.pi
        
        return {
            'responses': responses,
            'orientation_map': orientation_map,
            'strength_map': max_response,
            'orientation_idx': orientation_idx
        }
    
    elif pooling == 'energy':
        # Modèle énergie (somme des carrés)
        energy = responses**2
        energy_sum = energy.sum(dim=1)
        
        # Vecteur résultant pour l'orientation
        angles = torch.linspace(0, math.pi, n_filters, device=image.device)
        cos_angles = torch.cos(2 * angles).view(1, n_filters, 1, 1)
        sin_angles = torch.sin(2 * angles).view(1, n_filters, 1, 1)
        
        x_component = (responses * cos_angles).sum(dim=1)
        y_component = (responses * sin_angles).sum(dim=1)
        
        orientation_map = 0.5 * torch.atan2(y_component, x_component)
        coherence_map = torch.sqrt(x_component**2 + y_component**2) / (energy_sum + 1e-8)
        
        return {
            'responses': responses,
            'orientation_map': orientation_map,
            'coherence_map': coherence_map,
            'energy': energy_sum
        }
    
    else:  # 'mean'
        # Moyenne pondérée
        orientation_angles = torch.linspace(0, math.pi, n_filters, device=image.device)
        
        # Poids par orientation
        weights = F.softmax(responses, dim=1)
        
        # Orientation moyenne pondérée
        orientation_map = (weights * orientation_angles.view(1, n_filters, 1, 1)).sum(dim=1)
        
        # Écart-type (mesure de sélectivité)
        angle_diff = orientation_angles.view(1, n_filters, 1, 1) - orientation_map.unsqueeze(1)
        angle_diff = torch.remainder(angle_diff + math.pi/2, math.pi) - math.pi/2
        selectivity_map = torch.sqrt((weights * angle_diff**2).sum(dim=1))
        
        return {
            'responses': responses,
            'orientation_map': orientation_map,
            'selectivity_map': selectivity_map,
            'weights': weights
        }
