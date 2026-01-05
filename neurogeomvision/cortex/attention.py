"""
Module attention.py - Mécanismes d'attention visuelle
Attention spatiale, basée sur les caractéristiques, salience
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import math


class SaliencyMap(nn.Module):
    """
    Carte de saillance - Combine plusieurs caractéristiques pour détecter les régions saillantes.
    Modèle d'Itti, Koch & Niebur.
    """
    
    def __init__(self,
                 feature_maps: Dict[str, torch.Tensor] = None,
                 weights: Dict[str, float] = None,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.device = device
        
        # Poids par défaut pour différentes caractéristiques
        self.default_weights = {
            'intensity': 1.0,
            'orientation': 1.0,
            'color': 1.2,
            'motion': 1.5,
            'faces': 2.0,
            'text': 1.0
        }
        
        if weights is not None:
            self.default_weights.update(weights)
    
    def compute_feature_maps(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calcule les cartes de caractéristiques pour la saillance."""
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        batch_size, channels, height, width = image.shape
        
        feature_maps = {}
        
        # 1. Intensité (luminance)
        if channels == 3:
            # Convertir RGB en luminance
            intensity = 0.299 * image[:, 0:1, :, :] + 0.587 * image[:, 1:2, :, :] + 0.114 * image[:, 2:3, :, :]
        elif channels > 3:
            # Multi-canal (ex: cortical image): moyenne des canaux
            intensity = image.mean(dim=1, keepdim=True)
        else:
            intensity = image
        
        # Contrastes d'intensité à différentes échelles
        intensity_pyramid = self._create_pyramid(intensity, n_levels=4)
        intensity_conspicuity = self._compute_conspicuity(intensity_pyramid)
        feature_maps['intensity'] = intensity_conspicuity
        
        # 2. Couleur (si disponible)
        if channels == 3:
            # Canaux d'opposition rouge-vert et bleu-jaune
            rg = image[:, 0:1, :, :] - image[:, 1:2, :, :]
            by = image[:, 2:3, :, :] - (image[:, 0:1, :, :] + image[:, 1:2, :, :]) / 2
            
            rg_pyramid = self._create_pyramid(rg, n_levels=4)
            by_pyramid = self._create_pyramid(by, n_levels=4)
            
            rg_conspicuity = self._compute_conspicuity(rg_pyramid)
            by_conspicuity = self._compute_conspicuity(by_pyramid)
            
            # Combiner les canaux de couleur
            color_conspicuity = (rg_conspicuity + by_conspicuity) / 2
            feature_maps['color'] = color_conspicuity
        
        # 3. Orientation (nécessite des filtres Gabor)
        # Pour simplifier, utilisons des gradients
        sobel_x = torch.tensor([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], device=self.device).float().view(1, 1, 3, 3) / 8.0
        
        sobel_y = torch.tensor([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]], device=self.device).float().view(1, 1, 3, 3) / 8.0
        
        gradient_x = F.conv2d(intensity, sobel_x, padding=1)
        gradient_y = F.conv2d(intensity, sobel_y, padding=1)
        
        orientation_magnitude = torch.sqrt(gradient_x**2 + gradient_y**2 + 1e-8)
        
        orientation_pyramid = self._create_pyramid(orientation_magnitude, n_levels=4)
        orientation_conspicuity = self._compute_conspicuity(orientation_pyramid)
        feature_maps['orientation'] = orientation_conspicuity
        
        return feature_maps
    
    def _create_pyramid(self, image: torch.Tensor, n_levels: int = 4) -> List[torch.Tensor]:
        """Crée une pyramide gaussienne."""
        pyramid = [image]
        
        for level in range(1, n_levels):
            # Sous-échantillonner par 2
            h, w = pyramid[-1].shape[-2:]
            if h > 1 and w > 1:
                downsampled = F.avg_pool2d(pyramid[-1], kernel_size=2, stride=2)
                pyramid.append(downsampled)
            else:
                break
        
        return pyramid
    

    def _compute_conspicuity(self, pyramid: List[torch.Tensor]) -> torch.Tensor:
        """Calcule la conspicuité à partir d'une pyramide."""
        if not pyramid:
            return torch.zeros(1, 1, 1, 1, device=self.device)
    
        # Utiliser la taille du niveau le plus fin comme référence
        reference_size = pyramid[0].shape[-2:]
    
        # Calculer les différences entre les niveaux
        conspicuity_maps = []
    
        for i in range(len(pyramid)):
            for j in range(i + 1, len(pyramid)):
                # Redimensionner à la taille de référence
                level_i_resized = F.interpolate(pyramid[i], size=reference_size, mode='bilinear')
                level_j_resized = F.interpolate(pyramid[j], size=reference_size, mode='bilinear')
            
                # Différence absolue
                diff = torch.abs(level_i_resized - level_j_resized)
            
                # Normaliser
                diff_min = diff.min()
                diff_max = diff.max()
                if diff_max - diff_min > 1e-8:
                    diff_normalized = (diff - diff_min) / (diff_max - diff_min)
                else:
                    diff_normalized = diff
            
                conspicuity_maps.append(diff_normalized)
    
        if conspicuity_maps:
            # Moyenne sur toutes les comparaisons
            combined = torch.stack(conspicuity_maps, dim=0).mean(dim=0)
            return combined
        else:
            return pyramid[0]
        
            
    def forward(self,
                image: torch.Tensor,
                feature_maps: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Calcule la carte de saillance.
        
        Args:
            image: Image d'entrée
            feature_maps: Cartes de caractéristiques pré-calculées (optionnel)
            
        Returns:
            Carte de saillance et cartes intermédiaires
        """
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        batch_size, channels, height, width = image.shape
        
        # Calculer ou utiliser les cartes de caractéristiques fournies
        if feature_maps is None:
            feature_maps = self.compute_feature_maps(image)
        
        # Normaliser chaque carte de caractéristiques
        normalized_maps = {}
        for name, feature_map in feature_maps.items():
            if feature_map.numel() > 0:
                # Normalisation locale
                local_mean = F.avg_pool2d(feature_map, kernel_size=7, stride=1, padding=3)
                local_std = torch.sqrt(F.avg_pool2d(feature_map**2, kernel_size=7, stride=1, padding=3) - local_mean**2 + 1e-8)
                
                normalized = (feature_map - local_mean) / (local_std + 1e-8)
                normalized = torch.sigmoid(normalized)  # [0, 1]
                
                normalized_maps[name] = normalized
        
        # Combiner les cartes normalisées avec pondération
        saliency = torch.zeros(batch_size, 1, height, width, device=self.device)
        total_weight = 0.0
        
        for name, normalized_map in normalized_maps.items():
            weight = self.default_weights.get(name, 1.0)
            
            # Redimensionner si nécessaire
            if normalized_map.shape[-2:] != (height, width):
                normalized_map = F.interpolate(normalized_map, size=(height, width), mode='bilinear')
            
            saliency += weight * normalized_map
            total_weight += weight
        
        if total_weight > 0:
            saliency = saliency / total_weight
        
        # Lissage gaussien final
        saliency_smoothed = F.avg_pool2d(saliency, kernel_size=5, stride=1, padding=2)
        
        return {
            'saliency_map': saliency_smoothed,
            'feature_maps': feature_maps,
            'normalized_maps': normalized_maps,
            'weights': self.default_weights
        }


class SpatialAttention(nn.Module):
    """
    Attention spatiale - Modèle de déplacement de l'attention dans l'espace.
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int],
                 input_channels: int = 1,
                 attention_field: Tuple[int, int] = (32, 32),
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.input_shape = input_shape
        self.input_channels = input_channels
        self.attention_field = attention_field
        self.device = device
        
        # Carte d'attention (peut être apprise)
        self.attention_map = nn.Parameter(torch.randn(1, 1, input_shape[0], input_shape[1], device=device))
        
        # Mécanisme de déplacement d'attention
        self.attention_shift = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 2, kernel_size=3, padding=1),  # Prédit dx, dy
            nn.Tanh()  # Normalisé entre -1 et 1
        )
        
        # Amplification des caractéristiques dans la région d'attention
        self.feature_amplification = 2.0
    
    def compute_attention_region(self,
                                center: Tuple[float, float],
                                sigma: float = 10.0) -> torch.Tensor:
        """Crée une région d'attention gaussienne autour d'un centre."""
        height, width = self.input_shape
        
        # Grille de coordonnées
        y, x = torch.meshgrid(
            torch.linspace(0, height-1, height, device=self.device),
            torch.linspace(0, width-1, width, device=self.device),
            indexing='ij'
        )
        
        # Centre normalisé
        center_y, center_x = center
        
        # Champ d'attention gaussien
        attention = torch.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
        
        return attention.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    def shift_attention(self,
                       current_center: Tuple[float, float],
                       context_features: torch.Tensor) -> Tuple[Tuple[float, float], torch.Tensor]:
        """
        Déplace le centre d'attention en fonction des caractéristiques contextuelles.
        
        Args:
            current_center: Centre d'attention actuel (y, x)
            context_features: Caractéristiques visuelles contextuelles
            
        Returns:
            Nouveau centre et carte d'attention
        """
        if len(context_features.shape) == 3:
            context_features = context_features.unsqueeze(0)
        
        # Prédire le déplacement à partir des caractéristiques
        displacement = self.attention_shift(context_features)
        
        # Extraire dx, dy (normalisés entre -1 et 1)
        dx = displacement[:, 0:1, :, :].mean()
        dy = displacement[:, 1:2, :, :].mean()
        
        # Convertir en pixels
        max_shift = min(self.input_shape) * 0.2  # Déplacement maximal de 20% de l'image
        dx_pixels = dx * max_shift
        dy_pixels = dy * max_shift
        
        # Nouveau centre
        current_y, current_x = current_center
        new_y = torch.clamp(torch.tensor(current_y + dy_pixels, device=self.device), 0, self.input_shape[0]-1)
        new_x = torch.clamp(torch.tensor(current_x + dx_pixels, device=self.device), 0, self.input_shape[1]-1)
        
        new_center = (new_y.item(), new_x.item())
        
        # Créer la nouvelle carte d'attention
        new_attention = self.compute_attention_region(new_center)
        
        return new_center, new_attention
    
    def forward(self,
                features: torch.Tensor,
                attention_center: Optional[Tuple[float, float]] = None,
                saliency_map: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Applique l'attention spatiale aux caractéristiques.
        
        Args:
            features: Caractéristiques visuelles (B, C, H, W)
            attention_center: Centre d'attention initial (optionnel)
            saliency_map: Carte de saillance pour guider l'attention (optionnel)
            
        Returns:
            Caractéristiques avec attention, carte d'attention, nouveau centre
        """
        if len(features.shape) == 3:
            features = features.unsqueeze(0)
        
        batch_size, channels, height, width = features.shape
        
        # Initialiser le centre d'attention
        if attention_center is None:
            # Par défaut au centre de l'image
            attention_center = (height / 2, width / 2)
        
        # Si une carte de saillance est fournie, l'utiliser pour guider l'attention
        if saliency_map is not None:
            if len(saliency_map.shape) == 2:
                saliency_map = saliency_map.unsqueeze(0).unsqueeze(0)
            
            # Trouver le point le plus saillant
            saliency_flat = saliency_map.view(batch_size, -1)
            max_idx = saliency_flat.argmax(dim=1)
            
            max_y = max_idx // width
            max_x = max_idx % width
            
            # Mettre à jour le centre d'attention
            attention_center = (max_y.float().mean().item(), max_x.float().mean().item())
        
        # Créer la carte d'attention
        attention_map = self.compute_attention_region(attention_center)
        
        # Amplifier les caractéristiques dans la région d'attention
        weighted_features = features * (1.0 + (self.feature_amplification - 1.0) * attention_map)
        
        # Déplacer l'attention pour le prochain time step
        new_center, new_attention = self.shift_attention(attention_center, features)
        
        return {
            'attended_features': weighted_features,
            'attention_map': attention_map,
            'attention_center': attention_center,
            'new_attention_map': new_attention,
            'new_attention_center': new_center,
            'feature_amplification': self.feature_amplification
        }


class FeatureBasedAttention(nn.Module):
    """
    Attention basée sur les caractéristiques - Sélectionne certaines caractéristiques.
    """
    
    def __init__(self,
                 n_features: int,
                 attention_dim: int = 32,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.n_features = n_features
        self.device = device
        
        # Réseau pour générer les poids d'attention
        self.attention_network = nn.Sequential(
            nn.Linear(n_features, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, attention_dim // 2),
            nn.ReLU(),
            nn.Linear(attention_dim // 2, n_features),
            nn.Softmax(dim=-1)
        )
        
        # Mécanisme de modulation
        self.modulation_gain = nn.Parameter(torch.ones(1, n_features, 1, 1, device=device))
        self.modulation_bias = nn.Parameter(torch.zeros(1, n_features, 1, 1, device=device))
    
    def forward(self,
                features: torch.Tensor,
                attention_query: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Applique l'attention basée sur les caractéristiques.
        
        Args:
            features: Caractéristiques (B, C, H, W) ou (B, C)
            attention_query: Vecteur de requête d'attention (optionnel)
            
        Returns:
            Caractéristiques modulées, poids d'attention
        """
        original_shape = features.shape
        
        if len(features.shape) == 4:
            # Caractéristiques spatiales: (B, C, H, W)
            batch_size, channels, height, width = features.shape
            
            # Global average pooling pour obtenir un vecteur par canal
            features_pooled = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)  # (B, C)
            
            spatial_mode = True
        else:
            # Déjà un vecteur: (B, C)
            features_pooled = features
            spatial_mode = False
        
        batch_size, n_channels = features_pooled.shape
        
        if n_channels != self.n_features:
            raise ValueError(f"Attendu {self.n_features} caractéristiques, obtenu {n_channels}")
        
        # Générer les poids d'attention
        if attention_query is not None:
            # Combiner les caractéristiques avec la requête
            combined = features_pooled + attention_query
        else:
            combined = features_pooled
        
        attention_weights = self.attention_network(combined)  # (B, C)
        
        if spatial_mode:
            # Étendre les poids pour la modulation spatiale
            attention_weights = attention_weights.view(batch_size, self.n_features, 1, 1)
            
            # Moduler les caractéristiques spatiales
            modulated = features * (self.modulation_gain * attention_weights + self.modulation_bias)
            
            # Re-pooler pour le vecteur modulé
            modulated_pooled = F.adaptive_avg_pool2d(modulated, 1).squeeze(-1).squeeze(-1)
        else:
            # Moduler directement le vecteur
            modulated = features_pooled * (self.modulation_gain.squeeze() * attention_weights + self.modulation_bias.squeeze())
            modulated_pooled = modulated
        
        return {
            'modulated_features': modulated if spatial_mode else modulated_pooled,
            'attention_weights': attention_weights,
            'original_features': features_pooled,
            'spatial_mode': spatial_mode
        }


class AttentionModel(nn.Module):
    """
    Modèle d'attention complet combinant spatial et caractéristiques.
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int],
                 feature_channels: int = 64,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.input_shape = input_shape
        self.feature_channels = feature_channels
        self.device = device
        
        # Sous-modules d'attention
        self.saliency = SaliencyMap(device=device)
        self.spatial_attention = SpatialAttention(
            input_shape, 
            input_channels=feature_channels,
            device=device
        )
        self.feature_attention = FeatureBasedAttention(feature_channels, device=device)
        
        # Intégration des différentes formes d'attention
        self.integration = nn.Sequential(
            nn.Conv2d(feature_channels * 2, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Prédiction de l'action suivante (ex: saccade)
        self.action_predictor = nn.Sequential(
            nn.Linear(feature_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # dx, dy, zoom in, zoom out
            nn.Tanh()
        )
    
    def forward(self,
                image: torch.Tensor,
                features: torch.Tensor,
                previous_attention: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        Applique le modèle d'attention complet.
        
        Args:
            image: Image brute (pour la saillance)
            features: Caractéristiques extraites (pour l'attention spatiale et caractéristiques)
            previous_attention: État d'attention précédent (optionnel)
            
        Returns:
            Résultats d'attention complets
        """
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        if len(features.shape) == 3:
            features = features.unsqueeze(0)
        
        batch_size, channels, height, width = features.shape
        
        # 1. Calcul de la saillance
        saliency_results = self.saliency(image)
        saliency_map = saliency_results['saliency_map']
        
        # 2. Attention spatiale
        if previous_attention is not None:
            initial_center = previous_attention.get('new_attention_center', None)
        else:
            initial_center = None
        
        spatial_results = self.spatial_attention(
            features, 
            attention_center=initial_center,
            saliency_map=saliency_map
        )
        
        # 3. Attention basée sur les caractéristiques
        # Utiliser la région d'attention comme requête
        attention_region = spatial_results['attention_map']
        
        # Pooling sur la région d'attention pour obtenir un vecteur de requête
        if attention_region.shape[-2:] != (height, width):
            attention_region = F.interpolate(attention_region, size=(height, width), mode='bilinear')
        
        # Caractéristiques pondérées par l'attention spatiale
        spatially_attended = spatial_results['attended_features']
        
        # Vecteur de requête: caractéristiques moyennes dans la région d'attention
        query = (spatially_attended * attention_region).sum(dim=[2, 3]) / (attention_region.sum(dim=[2, 3]) + 1e-8)
        
        feature_results = self.feature_attention(spatially_attended, attention_query=query)
        
        # 4. Intégration des différentes attentions
        modulated_features = feature_results['modulated_features']
        
        if feature_results['spatial_mode']:
            # Caractéristiques déjà spatiales
            integrated_input = torch.cat([spatially_attended, modulated_features], dim=1)
            integrated_features = self.integration(integrated_input)
        else:
            # Nécessite un remodelage
            integrated_features = modulated_features
        
        # 5. Prédiction de l'action suivante
        action_prediction = self.action_predictor(
            F.adaptive_avg_pool2d(integrated_features, 1).squeeze(-1).squeeze(-1)
        )
        
        return {
            'saliency': saliency_results,
            'spatial_attention': spatial_results,
            'feature_attention': feature_results,
            'integrated_features': integrated_features,
            'action_prediction': action_prediction,
            'attention_history': {
                'center': spatial_results['attention_center'],
                'map': spatial_results['attention_map']
            }
        }
