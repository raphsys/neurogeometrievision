"""
Module cortical_areas.py - Aires corticales visuelles
V1, V2, V4, MT et leurs propriétés fonctionnelles
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import math


class V1SimpleCell(nn.Module):
    """
    Cellule simple de V1 - Sélective à l'orientation, position et phase.
    Modèle de Hubel & Wiesel.
    """
    
    def __init__(self,
                 orientation: float = 0.0,  # en radians
                 spatial_freq: float = 0.1,  # cycles/pixel
                 phase: float = 0.0,  # phase du filtre
                 sigma_x: float = 1.0,
                 sigma_y: float = 2.0,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.orientation = orientation
        self.spatial_freq = spatial_freq
        self.phase = phase
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.device = device
        
        # Créer le filtre Gabor
        self.gabor_filter = self._create_gabor_filter()
        
        # Seuil d'activation
        self.threshold = nn.Parameter(torch.tensor(0.1, device=device))
        
    def _create_gabor_filter(self, size: int = 15) -> torch.Tensor:
        """Crée un filtre Gabor 2D."""
        center = size // 2
        
        # Grille de coordonnées
        y, x = torch.meshgrid(
            torch.arange(size, device=self.device) - center,
            torch.arange(size, device=self.device) - center,
            indexing='ij'
        )
        
        # Rotation selon l'orientation
        x_rot = x * math.cos(self.orientation) + y * math.sin(self.orientation)
        y_rot = -x * math.sin(self.orientation) + y * math.cos(self.orientation)
        
        # Enveloppe gaussienne
        gaussian = torch.exp(-0.5 * (x_rot**2 / self.sigma_x**2 + y_rot**2 / self.sigma_y**2))
        
        # Porteuse sinusoïdale
        carrier = torch.cos(2 * math.pi * self.spatial_freq * x_rot + self.phase)
        
        # Filtre Gabor complet
        gabor = gaussian * carrier
        
        # Normalisation à somme nulle
        gabor = gabor - gabor.mean()
        gabor = gabor / (gabor.abs().sum() + 1e-8)
        
        return gabor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Réponse de la cellule simple.
        
        Args:
            x: Entrée visuelle (B, C, H, W) ou (H, W)
            
        Returns:
            Réponse de la cellule
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif len(x.shape) == 3:
            if x.shape[0] <= 3:  # Canaux
                x = x.unsqueeze(0)  # (1, C, H, W)
            else:  # Batch
                x = x.unsqueeze(1)  # (B, 1, H, W)
        
        batch_size, channels, height, width = x.shape
        
        # Si multi-canal, prendre la moyenne
        if channels > 1:
            x = x.mean(dim=1, keepdim=True)
        
        # Convolution avec le filtre Gabor
        if height >= 15 and width >= 15:
            response = F.conv2d(x, self.gabor_filter, padding=7)
        else:
            # Si trop petit, interpolation
            x_resized = F.interpolate(x, size=(max(15, height), max(15, width)), mode='bilinear')
            response = F.conv2d(x_resized, self.gabor_filter, padding=7)
            response = F.interpolate(response, size=(height, width), mode='bilinear')
        
        # Rectification et seuil
        response = F.relu(response - self.threshold)
        
        return response.squeeze()


class V1ComplexCell(nn.Module):
    """
    Cellule complexe de V1 - Sélective à l'orientation mais invariante à la phase.
    Pooling sur plusieurs cellules simples.
    """
    
    def __init__(self,
                 orientation: float = 0.0,
                 n_simple_cells: int = 4,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.orientation = orientation
        self.device = device
        
        # Créer une population de cellules simples avec différentes phases
        self.simple_cells = nn.ModuleList([
            V1SimpleCell(
                orientation=orientation,
                spatial_freq=0.1,
                phase=2 * math.pi * i / n_simple_cells,
                device=device
            )
            for i in range(n_simple_cells)
        ])
        
        # Pooling sur les phases (modèle énergie)
        self.pooling = nn.LPPool2d(norm_type=2, kernel_size=1, stride=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Réponse de la cellule complexe (modèle énergie).
        
        Args:
            x: Entrée visuelle
            
        Returns:
            Réponse invariante à la phase
        """
        # Réponses des cellules simples
        simple_responses = []
        for cell in self.simple_cells:
            response = cell(x)
            if len(response.shape) == 2:
                response = response.unsqueeze(0).unsqueeze(0)
            simple_responses.append(response)
        
        # Empiler les réponses
        stacked = torch.stack(simple_responses, dim=1)  # (B, n_cells, 1, H, W)
        
        # Modèle énergie: sqrt(sum(square))
        energy = torch.sqrt(torch.sum(stacked**2, dim=1) + 1e-8)
        
        return energy.squeeze()


class CorticalColumn(nn.Module):
    """
    Colonne corticale - Unité fonctionnelle de base du cortex.
    Contient des cellules avec toutes les orientations préférées.
    """
    
    def __init__(self,
                 input_size: int = 32,
                 n_orientations: int = 8,
                 n_simple_per_orientation: int = 4,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.input_size = input_size
        self.n_orientations = n_orientations
        self.device = device
        
        # Créer des cellules pour chaque orientation
        self.orientations = torch.linspace(0, math.pi, n_orientations + 1)[:n_orientations]
        
        # Cellules simples
        self.simple_cells = nn.ModuleList([
            V1SimpleCell(orientation=angle, device=device)
            for angle in self.orientations
        ])
        
        # Cellules complexes
        self.complex_cells = nn.ModuleList([
            V1ComplexCell(orientation=angle, device=device)
            for angle in self.orientations
        ])
        
        # Carte d'orientation (codage populationnel)
        self.orientation_vectors = torch.stack([
            torch.tensor([math.cos(2 * angle), math.sin(2 * angle)], device=device)
            for angle in self.orientations
        ], dim=0)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Traitement par la colonne corticale.
        
        Args:
            x: Entrée visuelle
            
        Returns:
            Dictionnaire avec réponses et carte d'orientation
        """
        # Réponses des cellules simples
        simple_responses = []
        for cell in self.simple_cells:
            response = cell(x)
            if len(response.shape) == 2:
                response = response.unsqueeze(0)
            simple_responses.append(response)
        
        # Réponses des cellules complexes
        complex_responses = []
        for cell in self.complex_cells:
            response = cell(x)
            if len(response.shape) == 2:
                response = response.unsqueeze(0)
            complex_responses.append(response)
        
        # Stacker les réponses
        simple_stacked = torch.stack(simple_responses, dim=1)  # (B, n_orientations, H, W)
        complex_stacked = torch.stack(complex_responses, dim=1)  # (B, n_orientations, H, W)
        
        # Calculer l'orientation dominante par pixel (vecteur résultant)
        batch_size, _, height, width = simple_stacked.shape
        
        # Normaliser les réponses
        simple_normalized = simple_stacked / (simple_stacked.sum(dim=1, keepdim=True) + 1e-8)
        
        # Vecteurs d'orientation pondérés
        orientation_vectors_expanded = self.orientation_vectors.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        orientation_vectors_expanded = orientation_vectors_expanded.expand(batch_size, -1, 2, height, width)
        
        weighted_vectors = simple_normalized.unsqueeze(2) * orientation_vectors_expanded
        result_vector = weighted_vectors.sum(dim=1)  # (B, 2, H, W)
        
        # Orientation et cohérence
        orientation_map = 0.5 * torch.atan2(result_vector[:, 1:], result_vector[:, :1])
        coherence_map = torch.sqrt(result_vector[:, 0:1]**2 + result_vector[:, 1:2]**2)
        
        return {
            'simple_responses': simple_stacked,
            'complex_responses': complex_stacked,
            'orientation_map': orientation_map,
            'coherence_map': coherence_map,
            'result_vector': result_vector
        }


class Hypercolumn(nn.Module):
    """
    Hypercolonne - Ensemble de colonnes corticales couvrant différentes positions.
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int],
                 column_size: int = 32,
                 stride: int = 16,
                 n_orientations: int = 8,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.input_shape = input_shape
        self.column_size = column_size
        self.stride = stride
        self.n_orientations = n_orientations
        self.device = device
        
        # Calculer le nombre de colonnes
        self.n_columns_h = (input_shape[0] - column_size) // stride + 1
        self.n_columns_w = (input_shape[1] - column_size) // stride + 1
        self.n_columns_total = self.n_columns_h * self.n_columns_w
        
        # Créer les colonnes corticales
        self.columns = nn.ModuleList([
            CorticalColumn(
                input_size=column_size,
                n_orientations=n_orientations,
                device=device
            )
            for _ in range(self.n_columns_total)
        ])
        
        # Positions des colonnes
        self.column_positions = []
        for i in range(self.n_columns_h):
            for j in range(self.n_columns_w):
                self.column_positions.append((
                    i * stride + column_size // 2,
                    j * stride + column_size // 2
                ))
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Traitement par l'hypercolonne.
        
        Args:
            x: Entrée visuelle
            
        Returns:
            Cartes d'orientation et cohérence complètes
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif len(x.shape) == 3:
            if x.shape[0] <= 3:
                x = x.unsqueeze(0)  # (1, C, H, W)
            else:
                x = x.unsqueeze(1)  # (B, 1, H, W)
        
        batch_size, channels, height, width = x.shape
        
        # Initialiser les cartes de sortie
        orientation_map = torch.zeros(batch_size, 1, height, width, device=self.device)
        coherence_map = torch.zeros(batch_size, 1, height, width, device=self.device)
        response_map = torch.zeros(batch_size, self.n_orientations, height, width, device=self.device)
        
        # Traiter chaque colonne
        for idx, (pos_h, pos_w) in enumerate(self.column_positions):
            column = self.columns[idx]
            
            # Extraire la région
            h_start = max(0, pos_h - self.column_size // 2)
            h_end = min(height, pos_h + self.column_size // 2)
            w_start = max(0, pos_w - self.column_size // 2)
            w_end = min(width, pos_w + self.column_size // 2)
            
            if h_end > h_start and w_end > w_start:
                region = x[:, :, h_start:h_end, w_start:w_end]
                
                # Traiter la région
                column_results = column(region)
                
                # Mettre à jour les cartes
                region_orientation = column_results['orientation_map']
                region_coherence = column_results['coherence_map']
                region_responses = column_results['complex_responses']
                
                # Interpolation pour correspondre à la région
                if region_orientation.shape[-2:] != (h_end - h_start, w_end - w_start):
                    region_orientation = F.interpolate(region_orientation, 
                                                      size=(h_end - h_start, w_end - w_start),
                                                      mode='bilinear')
                    region_coherence = F.interpolate(region_coherence,
                                                    size=(h_end - h_start, w_end - w_start),
                                                    mode='bilinear')
                    region_responses = F.interpolate(region_responses,
                                                    size=(h_end - h_start, w_end - w_start),
                                                    mode='bilinear')
                
                # Accumuler (moyenne pondérée par la cohérence)
                orientation_map[:, :, h_start:h_end, w_start:w_end] += region_orientation * region_coherence
                coherence_map[:, :, h_start:h_end, w_start:w_end] += region_coherence
                response_map[:, :, h_start:h_end, w_start:w_end] += region_responses.squeeze(1)
        
        # Normaliser par la cohérence
        orientation_map = orientation_map / (coherence_map + 1e-8)
        
        return {
            'orientation_map': orientation_map,
            'coherence_map': coherence_map,
            'response_map': response_map,
            'column_positions': self.column_positions,
            'n_columns': self.n_columns_total
        }


# Aires corticales supérieures
class V2Cell(nn.Module):
    """Cellule de V2 - Détection de contours, angles, jonctions."""
    
    def __init__(self,
                 feature_type: str = 'contour',  # 'contour', 'angle', 'junction'
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.feature_type = feature_type
        self.device = device
        
        # Filtres pour différentes caractéristiques
        if feature_type == 'contour':
            self.filter = self._create_contour_filter()
        elif feature_type == 'angle':
            self.filter = self._create_angle_filter()
        elif feature_type == 'junction':
            self.filter = self._create_junction_filter()
    
    def _create_contour_filter(self) -> torch.Tensor:
        """Filtre pour la détection de contours."""
        filter_size = 11
        gabor1 = self._create_gabor(0, filter_size)
        gabor2 = self._create_gabor(math.pi/2, filter_size)
        return (gabor1 + gabor2) / 2
    
    def _create_angle_filter(self) -> torch.Tensor:
        """Filtre pour la détection d'angles."""
        filter_size = 15
        filter_kernel = torch.zeros(1, 1, filter_size, filter_size, device=self.device)
        center = filter_size // 2
        
        # Créer un angle de 90 degrés
        for i in range(filter_size):
            for j in range(filter_size):
                if i >= center and j >= center:
                    filter_kernel[0, 0, i, j] = 1.0
        
        return filter_kernel / filter_kernel.sum()
    
    def _create_junction_filter(self) -> torch.Tensor:
        """Filtre pour la détection de jonctions."""
        filter_size = 15
        filter_kernel = torch.zeros(1, 1, filter_size, filter_size, device=self.device)
        center = filter_size // 2
        
        # Créer une jonction en T
        for i in range(filter_size):
            for j in range(filter_size):
                if (i == center and j >= center) or (i >= center and j == center):
                    filter_kernel[0, 0, i, j] = 1.0
        
        return filter_kernel / filter_kernel.sum()
    
    def _create_gabor(self, orientation: float, size: int) -> torch.Tensor:
        """Crée un filtre Gabor simple."""
        center = size // 2
        
        y, x = torch.meshgrid(
            torch.arange(size, device=self.device) - center,
            torch.arange(size, device=self.device) - center,
            indexing='ij'
        )
        
        x_rot = x * math.cos(orientation) + y * math.sin(orientation)
        envelope = torch.exp(-0.5 * (x_rot**2 / 4.0**2))
        carrier = torch.cos(2 * math.pi * 0.1 * x_rot)
        
        gabor = envelope * carrier
        return gabor.unsqueeze(0).unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Réponse de la cellule V2."""
        if len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        
        batch_size, channels, height, width = x.shape
        
        if channels > 1:
            x = x.mean(dim=1, keepdim=True)
        
        if height >= 15 and width >= 15:
            response = F.conv2d(x, self.filter, padding=7)
        else:
            x_resized = F.interpolate(x, size=(max(15, height), max(15, width)), mode='bilinear')
            response = F.conv2d(x_resized, self.filter, padding=7)
            response = F.interpolate(response, size=(height, width), mode='bilinear')
        
        return F.relu(response).squeeze()


class V4Cell(nn.Module):
    """Cellule de V4 - Sélective aux formes complexes, courbures."""
    
    def __init__(self,
                 shape_type: str = 'curve',  # 'curve', 'spiral', 'star'
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.shape_type = shape_type
        self.device = device
        
        # Réseau pour la reconnaissance de formes
        self.shape_net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Réponse de la cellule V4."""
        # S'assurer que x a 4 dimensions
        if len(x.shape) == 2:
            # (height, width) -> (1, 1, height, width)
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 3:
            # Déterminer si c'est (batch, height, width) ou (channels, height, width)
            if x.shape[0] <= 3:  # Probablement des canaux (C, H, W)
                x = x.unsqueeze(0)  # (1, C, H, W)
            else:  # Probablement batch sans canal (B, H, W)
                x = x.unsqueeze(1)  # (B, 1, H, W)
        
        # Maintenant x a 4 dimensions
        batch_size, channels, height, width = x.shape
        
        if channels > 1:
            x = x.mean(dim=1, keepdim=True)
        
        # Redimensionner si nécessaire pour le réseau
        if height < 32 or width < 32:
            x_resized = F.interpolate(x, size=(32, 32), mode='bilinear')
            response = self.shape_net(x_resized)
        else:
            response = self.shape_net(x)
        
        return response.view(-1)
        
        
class V4CurveDetector(nn.Module):
    """Détecteur de courbures pour V4."""
    
    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.device = device
        
        # Filtres pour détection de courbure à différentes orientations
        self.n_orientations = 8
        self.curvature_filters = nn.ModuleList()
        
        # Créer des filtres de courbure pour différentes orientations
        for orientation in torch.linspace(0, math.pi, self.n_orientations + 1)[:self.n_orientations]:
            filter_layer = nn.Conv2d(1, 1, kernel_size=15, padding=7, bias=False)
            
            # Créer un filtre de courbure (arc de cercle)
            kernel = self._create_curvature_kernel(orientation.item(), 15)
            filter_layer.weight.data = kernel
            filter_layer.weight.requires_grad = False
            
            self.curvature_filters.append(filter_layer)
        
        # Pooling et intégration
        self.integration = nn.Sequential(
            nn.Conv2d(self.n_orientations, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def _create_curvature_kernel(self, orientation: float, size: int) -> torch.Tensor:
        """Crée un noyau de détection de courbure."""
        center = size // 2
        kernel = torch.zeros(1, 1, size, size, device=self.device)
        
        # Paramètres de la courbure
        radius = size // 3
        angle_range = math.pi / 3  # 60 degrés
        
        # Créer un arc de cercle
        for i in range(size):
            for j in range(size):
                x = j - center
                y = i - center
                
                # Rotation selon l'orientation
                x_rot = x * math.cos(orientation) + y * math.sin(orientation)
                y_rot = -x * math.sin(orientation) + y * math.cos(orientation)
                
                # Distance au centre de l'arc
                dist_to_center = math.sqrt(x_rot**2 + (y_rot - radius)**2)
                
                # Angle dans le système de coordonnées de l'arc
                angle = math.atan2(y_rot - radius, x_rot)
                
                # Vérifier si le point est sur l'arc
                if abs(dist_to_center - radius) < 1.5 and -angle_range/2 <= angle <= angle_range/2:
                    kernel[0, 0, i, j] = math.cos(angle * 2)  # Poids variant selon la position sur l'arc
        
        # Normaliser le noyau
        kernel = kernel - kernel.mean()
        if kernel.abs().sum() > 0:
            kernel = kernel / (kernel.abs().sum() + 1e-8)
        
        return kernel
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Détecte les courbures dans l'image.
        
        Args:
            x: Tensor d'entrée (B, C, H, W) ou (H, W) ou (B, H, W)
            
        Returns:
            Réponse de courbure normalisée
        """
        # S'assurer que x a 4 dimensions
        if len(x.shape) == 2:
            # (H, W) -> (1, 1, H, W)
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 3:
            # Déterminer la dimension manquante
            if x.shape[0] <= 3:  # Canaux (C, H, W) où C est petit
                # (C, H, W) -> (1, C, H, W)
                x = x.unsqueeze(0)
            else:  # Batch sans canal (B, H, W)
                # (B, H, W) -> (B, 1, H, W)
                x = x.unsqueeze(1)
        
        # Maintenant x a 4 dimensions: (B, C, H, W)
        batch_size, channels, height, width = x.shape
        
        # Si multi-canal, prendre la moyenne
        if channels > 1:
            x = x.mean(dim=1, keepdim=True)
        
        # Appliquer tous les filtres de courbure
        curvature_responses = []
        for filter_layer in self.curvature_filters:
            response = filter_layer(x)
            curvature_responses.append(response)
        
        # Combiner toutes les réponses d'orientation
        if curvature_responses:
            # Stack: (B, n_orientations, H, W)
            all_responses = torch.cat(curvature_responses, dim=1)
        else:
            all_responses = torch.zeros(batch_size, 1, height, width, device=self.device)
        
        # Intégration des caractéristiques de courbure
        features = self.integration(all_responses)
        
        return features


class MTCell(nn.Module):
    """Cellule de MT (V5) - Sélective au mouvement directionnel."""
    
    def __init__(self,
                 preferred_direction: float = 0.0,  # en radians
                 speed_tuning: float = 2.0,  # vitesse préférée (pixels/frame)
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.preferred_direction = preferred_direction
        self.speed_tuning = speed_tuning
        self.device = device
        
        # Filtres spatio-temporels
        self.spatial_filter = self._create_spatial_filter()
        self.temporal_filter = self._create_temporal_filter()
        
    def _create_spatial_filter(self) -> torch.Tensor:
        """Filtre spatial orienté."""
        size = 9
        orientation = self.preferred_direction
        center = size // 2
        
        y, x = torch.meshgrid(
            torch.arange(size, device=self.device) - center,
            torch.arange(size, device=self.device) - center,
            indexing='ij'
        )
        
        x_rot = x * math.cos(orientation) + y * math.sin(orientation)
        y_rot = -x * math.sin(orientation) + y * math.cos(orientation)
        
        # Filtre Gabor spatial
        spatial = torch.exp(-0.5 * (x_rot**2 / 2.0**2 + y_rot**2 / 4.0**2))
        spatial = spatial * torch.cos(2 * math.pi * 0.15 * x_rot)
        
        return spatial.unsqueeze(0).unsqueeze(0)
    
    def _create_temporal_filter(self) -> torch.Tensor:
        """Filtre temporel (différence de Gaussiennes)."""
        time_steps = 5
        t = torch.arange(time_steps, device=self.device).float()
        
        # Gaussienne rapide (ON)
        tau_fast = 1.0
        fast = torch.exp(-t / tau_fast)
        
        # Gaussienne lente (OFF)
        tau_slow = 2.0
        slow = torch.exp(-t / tau_slow)
        
        # DoT (Difference of Temporal filters)
        temporal = fast - 0.7 * slow
        
        return temporal.view(1, 1, time_steps, 1, 1)
    
    def forward(self, video_sequence: torch.Tensor) -> torch.Tensor:
        """
        Réponse au mouvement.
        
        Args:
            video_sequence: Séquence vidéo (T, H, W) ou (B, T, H, W)
            
        Returns:
            Réponse directionnelle
        """
        if len(video_sequence.shape) == 3:
            video_sequence = video_sequence.unsqueeze(0)  # (1, T, H, W)
        
        batch_size, time_steps, height, width = video_sequence.shape
        
        # Ajouter dimension canal
        video_sequence = video_sequence.unsqueeze(2)  # (B, T, 1, H, W)
        
        # Filtrage spatial
        spatial_responses = []
        for t in range(time_steps):
            frame = video_sequence[:, t, :, :, :]
            if height >= 9 and width >= 9:
                spatial_response = F.conv2d(frame, self.spatial_filter, padding=4)
            else:
                frame_resized = F.interpolate(frame, size=(max(9, height), max(9, width)), mode='bilinear')
                spatial_response = F.conv2d(frame_resized, self.spatial_filter, padding=4)
                spatial_response = F.interpolate(spatial_response, size=(height, width), mode='bilinear')
            spatial_responses.append(spatial_response)
        
        # Stack temporel
        spatial_stacked = torch.stack(spatial_responses, dim=1)  # (B, T, 1, H, W)
        
        # Filtrage temporel (convolution 1D dans la dimension temporelle)
        # Réorganiser pour la convolution 3D: (B, C, T, H, W)
        spatial_stacked = spatial_stacked.permute(0, 2, 1, 3, 4)  # (B, 1, T, H, W)
        
        # Appliquer le filtre temporel
        temporal_response = F.conv3d(spatial_stacked, self.temporal_filter, padding=(2, 0, 0))
        
        # Prendre le maximum temporel
        motion_response, _ = torch.max(temporal_response, dim=2)  # (B, 1, H, W)
        
        # Ajustement pour la vitesse
        speed_factor = torch.exp(-(self.speed_tuning - 1.0)**2 / 2.0)
        motion_response = motion_response * speed_factor
        
        return F.relu(motion_response).squeeze()


def create_cortical_hierarchy(input_shape: Tuple[int, int],
                             device: str = 'cpu') -> nn.ModuleDict:
    """
    Crée une hiérarchie corticale complète (V1 -> V2 -> V4).
    
    Args:
        input_shape: Forme d'entrée (H, W)
        device: Device
        
    Returns:
        Dictionnaire avec les couches corticales
    """
    hierarchy = nn.ModuleDict()
    
    # V1 - Orientation
    hierarchy['v1'] = Hypercolumn(
        input_shape=input_shape,
        column_size=32,
        stride=16,
        n_orientations=8,
        device=device
    )
    
    # V2 - Formes simples
    hierarchy['v2'] = nn.ModuleDict({
        'contour': V2Cell(feature_type='contour', device=device),
        'angle': V2Cell(feature_type='angle', device=device),
        'junction': V2Cell(feature_type='junction', device=device)
    })
    
    # V4 - Formes complexes
    hierarchy['v4'] = nn.ModuleDict({
        'curve': V4Cell(shape_type='curve', device=device),
        'spiral': V4Cell(shape_type='spiral', device=device),
        'star': V4Cell(shape_type='star', device=device)
    })
    
    return hierarchy
