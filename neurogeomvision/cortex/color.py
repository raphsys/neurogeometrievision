"""
Module color.py - Traitement de la couleur dans le cortex
Opponence des couleurs, constance des couleurs, voies ventrales/dorsales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import math


class ColorOpponency(nn.Module):
    """
    Opponence des couleurs - Transforme RGB en canaux d'opposition.
    Rouge-vert, bleu-jaune, luminance.
    """
    
    def __init__(self,
                 opponent_type: str = 'dkl',  # 'dkl' (Derrington-Krauskopf-Lennie) ou 'lab'
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.opponent_type = opponent_type
        self.device = device
        
        # Matrices de transformation
        if opponent_type == 'dkl':
            # Transformation DKL (espace de chromaticité)
            self.transform = nn.Parameter(torch.tensor([
                [0.299, 0.587, 0.114],    # Luminance (L+M)
                [0.707, -0.707, 0.000],   # Rouge-vert (L-M)
                [0.408, 0.408, -0.816]    # Bleu-jaune (S-(L+M))
            ], device=device), requires_grad=False)
        
        elif opponent_type == 'lab':
            # Approximation LAB simplifiée
            self.transform = nn.Parameter(torch.tensor([
                [0.2126, 0.7152, 0.0722],  # L
                [0.5000, 0.5000, -1.0000], # a (rouge-vert)
                [0.2000, 0.0000, -0.2000]  # b (bleu-jaune)
            ], device=device), requires_grad=False)
        
        else:  # 'simple'
            # Opposition simple
            self.transform = nn.Parameter(torch.tensor([
                [1/3, 1/3, 1/3],          # Luminance
                [1.0, -1.0, 0.0],         # Rouge-vert
                [0.5, 0.5, -1.0]          # Bleu-jaune
            ], device=device), requires_grad=False)
    
    def forward(self, rgb_image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Transforme RGB en canaux d'opposition.
        
        Args:
            rgb_image: Image RGB (3, H, W) ou (B, 3, H, W)
            
        Returns:
            Canaux d'opposition
        """
        if len(rgb_image.shape) == 3:
            rgb_image = rgb_image.unsqueeze(0)  # (1, 3, H, W)
        
        batch_size, channels, height, width = rgb_image.shape
        
        if channels == 1:
            rgb_image = rgb_image.repeat(1, 3, 1, 1)
            channels = 3
        
        if channels != 3:
            raise ValueError(f"Attendu 3 canaux RGB, obtenu {channels}")
        
        # Aplatir spatialement pour la transformation matricielle
        rgb_flat = rgb_image.view(batch_size, 3, -1)  # (B, 3, H*W)
        
        # Appliquer la transformation
        opponent_flat = torch.bmm(self.transform.unsqueeze(0).expand(batch_size, -1, -1), rgb_flat)
        
        # Reformer en image
        opponent_image = opponent_flat.view(batch_size, 3, height, width)
        
        # Séparer les canaux
        luminance = opponent_image[:, 0:1, :, :]
        rg_opponent = opponent_image[:, 1:2, :, :]
        by_opponent = opponent_image[:, 2:3, :, :]
        
        # Rectification (réponses ON/OFF séparées)
        rg_on = F.relu(rg_opponent)  # Rouge > Vert
        rg_off = F.relu(-rg_opponent)  # Vert > Rouge
        
        by_on = F.relu(by_opponent)  # Jaune > Bleu
        by_off = F.relu(-by_opponent)  # Bleu > Jaune
        
        # Luminance ON/OFF
        lum_on = F.relu(luminance)
        lum_off = F.relu(-luminance)
        
        return {
            'opponent_image': opponent_image,
            'luminance': luminance,
            'rg_opponent': rg_opponent,
            'by_opponent': by_opponent,
            'rg_on': rg_on,
            'rg_off': rg_off,
            'by_on': by_on,
            'by_off': by_off,
            'lum_on': lum_on,
            'lum_off': lum_off,
            'transform_type': self.opponent_type
        }


class DoubleOpponentCell(nn.Module):
    """
    Cellule à double opposition - Sélective à la couleur et à l'orientation.
    Centre d'une couleur, surround de la couleur opposée.
    """
    
    def __init__(self,
                 preferred_color: str = 'rg',  # 'rg' (rouge-vert) ou 'by' (bleu-jaune)
                 preferred_orientation: float = 0.0,
                 center_color: str = 'on',  # 'on' ou 'off'
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.preferred_color = preferred_color
        self.preferred_orientation = preferred_orientation
        self.center_color = center_color
        self.device = device
        
        # Créer les filtres centre-surround pour la couleur
        self.center_filter, self.surround_filter = self._create_color_filters()
        
        # Filtre d'orientation pour la luminance
        self.orientation_filter = self._create_orientation_filter()
    
    def _create_color_filters(self, size: int = 15) -> Tuple[torch.Tensor, torch.Tensor]:
        """Crée des filtres centre-surround pour la couleur."""
        center = size // 2
        
        # Grille de distances
        y, x = torch.meshgrid(
            torch.arange(size, device=self.device) - center,
            torch.arange(size, device=self.device) - center,
            indexing='ij'
        )
        
        r = torch.sqrt(x**2 + y**2)
        
        # Filtre centre (gaussienne étroite)
        sigma_center = size / 6.0
        center_gauss = torch.exp(-r**2 / (2 * sigma_center**2))
        center_gauss = center_gauss / center_gauss.sum()
        
        # Filtre surround (gaussienne large)
        sigma_surround = size / 3.0
        surround_gauss = torch.exp(-r**2 / (2 * sigma_surround**2))
        surround_gauss = surround_gauss / surround_gauss.sum()
        
        # Pour double opposition, le surround a le signe opposé
        if self.center_color == 'on':
            # Centre positif, surround négatif
            center_filter = center_gauss
            surround_filter = -0.7 * surround_gauss
        else:
            # Centre négatif, surround positif
            center_filter = -center_gauss
            surround_filter = 0.7 * surround_gauss
        
        return (center_filter.unsqueeze(0).unsqueeze(0),
                surround_filter.unsqueeze(0).unsqueeze(0))
    
    def _create_orientation_filter(self, size: int = 11) -> torch.Tensor:
        """Crée un filtre Gabor pour l'orientation."""
        center = size // 2
        
        y, x = torch.meshgrid(
            torch.arange(size, device=self.device) - center,
            torch.arange(size, device=self.device) - center,
            indexing='ij'
        )
        
        # Rotation selon l'orientation préférée
        x_rot = x * math.cos(self.preferred_orientation) + y * math.sin(self.preferred_orientation)
        y_rot = -x * math.sin(self.preferred_orientation) + y * math.cos(self.preferred_orientation)
        
        # Filtre Gabor
        sigma_x = size / 4.0
        sigma_y = size / 2.0
        spatial_freq = 0.15
        
        gaussian = torch.exp(-0.5 * (x_rot**2 / sigma_x**2 + y_rot**2 / sigma_y**2))
        carrier = torch.cos(2 * math.pi * spatial_freq * x_rot)
        
        gabor = gaussian * carrier
        gabor = gabor - gabor.mean()
        gabor = gabor / (gabor.abs().sum() + 1e-8)
        
        return gabor.unsqueeze(0).unsqueeze(0)
    
    def forward(self, color_opponent: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Réponse de la cellule à double opposition.
        
        Args:
        color_opponent: Sortie de ColorOpponency
        
        Returns:
            Réponse de la cellule
        """
        # Sélectionner le canal couleur approprié
        if self.preferred_color == 'rg':
            if self.center_color == 'on':
                color_channel = color_opponent['rg_on']
            else:
                color_channel = color_opponent['rg_off']
        else:  # 'by'
            if self.center_color == 'on':
                color_channel = color_opponent['by_on']
            else:
                color_channel = color_opponent['by_off']
        
        # Canal de luminance pour l'orientation
        luminance = color_opponent['luminance']
        
        if len(color_channel.shape) == 3:
            color_channel = color_channel.unsqueeze(0)
            luminance = luminance.unsqueeze(0)
        
        batch_size, _, height, width = color_channel.shape
        
        # Utiliser le même padding pour tous les filtres
        # Déterminer la taille de padding maximale
        center_filter_size = self.center_filter.shape[-1]
        surround_filter_size = self.surround_filter.shape[-1]
        orientation_filter_size = self.orientation_filter.shape[-1]
        
        max_filter_size = max(center_filter_size, surround_filter_size, orientation_filter_size)
        padding = max_filter_size // 2
        
        # Appliquer les filtres avec le même padding
        center_response = F.conv2d(color_channel, self.center_filter, padding=padding)
        surround_response = F.conv2d(color_channel, self.surround_filter, padding=padding)
        
        # Opposition centre-surround pour la couleur
        color_response = center_response + surround_response
        
        # Appliquer le filtre d'orientation avec le même padding
        orientation_response = F.conv2d(luminance, self.orientation_filter, padding=padding)
        
        # Redimensionner à la même taille si nécessaire
        # (en théorie, avec le même padding, les tailles devraient être identiques)
        if orientation_response.shape[-2:] != color_response.shape[-2:]:
            # Prendre la taille la plus petite et redimensionner les deux
            target_h = min(color_response.shape[-2], orientation_response.shape[-2])
            target_w = min(color_response.shape[-1], orientation_response.shape[-1])
            
            color_response = F.interpolate(color_response, size=(target_h, target_w), mode='bilinear')
            orientation_response = F.interpolate(orientation_response, size=(target_h, target_w), mode='bilinear')
        
        # Combiner couleur et orientation (multiplication pour AND)
        combined_response = color_response * F.relu(orientation_response)
        
        # Normalisation locale
        local_mean = F.avg_pool2d(combined_response.abs(), kernel_size=5, stride=1, padding=2)
        normalized_response = combined_response / (local_mean + 1e-8)
        
        # Appliquer ReLU et s'assurer des bonnes dimensions
        response = F.relu(normalized_response)
        
        # S'assurer qu'on retourne la bonne forme
        # Si la réponse est (batch_size, 1, H, W), on garde
        # Si c'est (batch_size, H, W), on ajoute la dimension du canal
        if len(response.shape) == 3:
            response = response.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        
        return response


class ColorConstancy(nn.Module):
    """
    Constance des couleurs - Compensation de l'illumination.
    Modèle de Retinex simplifié.
    """
    
    def __init__(self,
                 scale_levels: int = 3,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.scale_levels = scale_levels
        self.device = device
        
        # Pyramide gaussienne pour différentes échelles
        self.gaussian_pyramid = nn.ModuleList()
        
        for level in range(scale_levels):
            sigma = 2.0 * (2 ** level)
            kernel_size = int(2 * 3 * sigma) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Créer un noyau gaussien
            gaussian = self._create_gaussian_kernel(sigma, kernel_size)
            conv = nn.Conv2d(3, 3, kernel_size, padding=kernel_size//2, groups=3, bias=False)
            conv.weight.data = gaussian
            conv.weight.requires_grad = False
            
            self.gaussian_pyramid.append(conv)
    
    def _create_gaussian_kernel(self, sigma: float, size: int) -> torch.Tensor:
        """Crée un noyau gaussien 2D."""
        center = size // 2
        
        y, x = torch.meshgrid(
            torch.arange(size, device=self.device) - center,
            torch.arange(size, device=self.device) - center,
            indexing='ij'
        )
        
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        # Pour convolution groupée (un filtre par canal)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        kernel = kernel.expand(3, 1, -1, -1)  # (3, 1, H, W)
        
        return kernel
    
    def forward(self, rgb_image: torch.Tensor) -> torch.Tensor:
        """
        Applique la constance des couleurs (Retinex).
        
        Args:
            rgb_image: Image RGB
            
        Returns:
            Image avec constance des couleurs
        """
        if len(rgb_image.shape) == 3:
            rgb_image = rgb_image.unsqueeze(0)  # (1, 3, H, W)
        
        batch_size, channels, height, width = rgb_image.shape
        
        if channels == 1:
            rgb_image = rgb_image.repeat(1, 3, 1, 1)
            channels = 3
        
        if channels != 3:
            raise ValueError(f"Attendu 3 canaux RGB, obtenu {channels}")
        
        # Convertir en espace log pour séparer réflexion et illumination
        log_image = torch.log(rgb_image + 1e-8)
        
        # Calculer l'illumination estimée à différentes échelles
        illumination_estimates = []
        
        for gaussian in self.gaussian_pyramid:
            # Lisser l'image (estimation de l'illumination)
            smoothed = gaussian(log_image)
            illumination_estimates.append(smoothed)
        
        # Combiner les estimations (moyenne pondérée)
        weights = torch.linspace(1.0, 0.5, len(illumination_estimates), device=self.device)
        weights = weights / weights.sum()
        
        combined_illumination = torch.zeros_like(log_image)
        for w, illum in zip(weights, illumination_estimates):
            combined_illumination += w * illum
        
        # Soustraire l'illumination pour obtenir la réflexion (couleur constante)
        reflection = log_image - combined_illumination
        
        # Normalisation adaptative
        # Recentrer et remettre à l'échelle chaque canal
        normalized = torch.zeros_like(reflection)
        for c in range(3):
            channel = reflection[:, c:c+1, :, :]
            channel_mean = channel.mean(dim=[2, 3], keepdim=True)
            channel_std = channel.std(dim=[2, 3], keepdim=True)
            normalized[:, c:c+1, :, :] = (channel - channel_mean) / (channel_std + 1e-8)
        
        # Convertir de retour en espace linéaire
        result = torch.exp(normalized)
        
        # Limiter les valeurs extrêmes
        result = torch.clamp(result, 0, 1)
        
        return result.squeeze()


class ColorProcessingStream(nn.Module):
    """
    Voie de traitement de la couleur - Modélise la voie ventrale (quoi).
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int],
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.input_shape = input_shape
        self.device = device
        
        # Étape 1: Opponence des couleurs
        self.color_opponency = ColorOpponency(opponent_type='dkl', device=device)
        
        # Étape 2: Constance des couleurs
        self.color_constancy = ColorConstancy(scale_levels=3, device=device)
        
        # Étape 3: Cellules à double opposition à différentes orientations
        self.double_opponent_cells = nn.ModuleList()
        
        n_orientations = 4
        color_channels = ['rg', 'by']
        center_types = ['on', 'off']
        
        for orientation in torch.linspace(0, math.pi, n_orientations + 1)[:n_orientations]:
            for color_channel in color_channels:
                for center_type in center_types:
                    cell = DoubleOpponentCell(
                        preferred_color=color_channel,
                        preferred_orientation=orientation.item(),
                        center_color=center_type,
                        device=device
                    )
                    self.double_opponent_cells.append(cell)
        
        # Nombre de features = nombre de cellules
        self.n_features = len(self.double_opponent_cells)
        print(f"ColorProcessingStream: {self.n_features} cellules créées")  # DEBUG
        
        # Étape 4: Intégration des caractéristiques de couleur
        self.feature_integration = nn.Sequential(
            nn.Conv2d(self.n_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Classification des couleurs (exemple: 11 catégories de couleur de base)
        self.color_classifier = nn.Linear(8, 11)
    
    def forward(self, rgb_image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Traitement complet de la couleur.
        
        Args:
            rgb_image: Image RGB
            
        Returns:
            Caractéristiques et classification de couleur
        """
        if len(rgb_image.shape) == 3:
            rgb_image = rgb_image.unsqueeze(0)
        
        batch_size, channels, height, width = rgb_image.shape
        
        if channels == 1:
            rgb_image = rgb_image.repeat(1, 3, 1, 1)
            channels = 3
        
        if channels != 3:
            raise ValueError(f"Attendu 3 canaux RGB, obtenu {channels}")
        
        print(f"ColorProcessingStream forward: input shape {rgb_image.shape}")  # DEBUG
        
        # 1. Constance des couleurs
        color_constant = self.color_constancy(rgb_image)
        print(f"  Après constance: {color_constant.shape}")  # DEBUG
        
        # 2. Opponence des couleurs
        opponent = self.color_opponency(color_constant)
        print(f"  Après opponence: {len(opponent)} canaux")  # DEBUG
        
        # 3. Réponses des cellules à double opposition
        double_opponent_responses = []
        for i, cell in enumerate(self.double_opponent_cells):
            response = cell(opponent)
            
            # DEBUG: Vérifier la forme
            print(f"  Cellule {i}: raw response shape {response.shape}")
            
            # Normaliser à 4 dimensions: (batch_size, 1, height, width)
            if len(response.shape) == 2:
                # (height, width) -> (1, 1, height, width)
                response = response.unsqueeze(0).unsqueeze(0)
            elif len(response.shape) == 3:
                # Cas 1: (1, height, width) -> (1, 1, height, width)
                # Cas 2: (batch_size, height, width) -> (batch_size, 1, height, width)
                if response.shape[0] == 1:
                    # Single batch
                    response = response.unsqueeze(0)  # (1, 1, H, W) si déjà (1, H, W)? Non...
                    # En fait si response est (1, H, W), on veut (1, 1, H, W)
                    response = response.unsqueeze(1)
                else:
                    # Multi-batch: (B, H, W) -> (B, 1, H, W)
                    response = response.unsqueeze(1)
            
            # Après normalisation, vérifier
            if len(response.shape) != 4 or response.shape[1] != 1:
                print(f"  WARNING: Cellule {i} shape après normalisation: {response.shape}")
            
            double_opponent_responses.append(response)
        
        # Stacker toutes les réponses
        if double_opponent_responses:
            all_responses = torch.cat(double_opponent_responses, dim=1)  # (B, n_cells, H, W)
            print(f"  All responses shape: {all_responses.shape}")  # DEBUG
            print(f"  Nombre de cellules: {len(self.double_opponent_cells)}")  # DEBUG
        else:
            all_responses = torch.zeros(batch_size, 1, height, width, device=self.device)
        
        # 4. Intégration des caractéristiques
        features = self.feature_integration(all_responses)
        print(f"  Features shape: {features.shape}")  # DEBUG
        
        # 5. Classification (optionnelle)
        color_probs = F.softmax(self.color_classifier(features), dim=1)
        
        return {
            'color_constant': color_constant,
            'opponent_channels': opponent,
            'double_opponent_responses': all_responses,
            'color_features': features,
            'color_probabilities': color_probs,
            'n_cells': len(self.double_opponent_cells)
        }
