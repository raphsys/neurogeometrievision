"""
Module motion.py - Détection et analyse du mouvement
Sélectivité directionnelle, filtres spatio-temporels, flux optique
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import math


class MotionEnergyFilter(nn.Module):
    """
    Filtre d'énergie de mouvement - Détecte le mouvement directionnel.
    Modèle de Adelson & Bergen (1985).
    """
    
    def __init__(self,
                 direction: float = 0.0,  # Direction préférée en radians
                 speed: float = 1.0,  # Vitesse préférée (pixels/frame)
                 temporal_freq: float = 0.25,  # Fréquence temporelle
                 spatial_freq: float = 0.1,  # Fréquence spatiale
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.direction = direction
        self.speed = speed
        self.temporal_freq = temporal_freq
        self.spatial_freq = spatial_freq
        self.device = device
        
        # Créer les filtres spatio-temporels
        self.filters = self._create_spatiotemporal_filters()
    
    def _create_spatiotemporal_filters(self,
                                      spatial_size: int = 11,
                                      temporal_size: int = 5) -> Dict[str, torch.Tensor]:
        """Crée une paire de filtres spatio-temporels en quadrature."""
        filters = {}
        
        # Coordonnées spatio-temporelles
        y, x, t = torch.meshgrid(
            torch.linspace(-1, 1, spatial_size, device=self.device),
            torch.linspace(-1, 1, spatial_size, device=self.device),
            torch.linspace(-1, 1, temporal_size, device=self.device),
            indexing='ij'
        )
        
        # Direction dans l'espace
        x_dir = x * math.cos(self.direction) + y * math.sin(self.direction)
        y_dir = -x * math.sin(self.direction) + y * math.cos(self.direction)
        
        # Coordonnées dans l'espace-temps selon la vitesse
        # Pour un mouvement dans la direction préférée à la vitesse préférée
        st_x = x_dir - self.speed * t
        
        # Enveloppe gaussienne 3D
        sigma_xy = 0.3
        sigma_t = 0.4
        gaussian = torch.exp(-0.5 * (x_dir**2 / sigma_xy**2 + 
                                    y_dir**2 / (sigma_xy*2)**2 + 
                                    t**2 / sigma_t**2))
        
        # Filtres en quadrature
        # Filtre pair (cos-cos)
        spatial_carrier = torch.cos(2 * math.pi * self.spatial_freq * st_x)
        temporal_carrier = torch.cos(2 * math.pi * self.temporal_freq * t)
        filter_even = gaussian * spatial_carrier * temporal_carrier
        
        # Filtre impair (sin-sin)
        spatial_carrier = torch.sin(2 * math.pi * self.spatial_freq * st_x)
        temporal_carrier = torch.sin(2 * math.pi * self.temporal_freq * t)
        filter_odd = gaussian * spatial_carrier * temporal_carrier
        
        # Normalisation
        filter_even = filter_even - filter_even.mean()
        filter_even = filter_even / (filter_even.abs().sum() + 1e-8)
        
        filter_odd = filter_odd - filter_odd.mean()
        filter_odd = filter_odd / (filter_odd.abs().sum() + 1e-8)
        
        # Réorganiser pour la convolution 3D: (out_channels, in_channels, T, H, W)
        filter_even = filter_even.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # (1, 1, T, H, W)
        filter_odd = filter_odd.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        
        return {'even': filter_even, 'odd': filter_odd}
    
    def forward(self, video_sequence: torch.Tensor) -> torch.Tensor:
        """
        Calcule l'énergie de mouvement.
        
        Args:
            video_sequence: Séquence vidéo (T, H, W) ou (B, T, H, W)
            
        Returns:
            Énergie de mouvement
        """
        if len(video_sequence.shape) == 3:
            video_sequence = video_sequence.unsqueeze(0)  # (1, T, H, W)
        
        batch_size, time_steps, height, width = video_sequence.shape
        
        # Ajouter dimension canal
        video_sequence = video_sequence.unsqueeze(1)  # (B, 1, T, H, W)
        
        # Vérifier la longueur temporelle
        temporal_size = self.filters['even'].shape[2]
        if time_steps < temporal_size:
            # Pad temporel
            pad_front = (temporal_size - time_steps) // 2
            pad_back = temporal_size - time_steps - pad_front
            video_sequence = F.pad(video_sequence, (0, 0, 0, 0, pad_front, pad_back), mode='replicate')
            time_steps = temporal_size
        
        # Convolution 3D avec les filtres
        padding = (temporal_size // 2, self.filters['even'].shape[-1] // 2, self.filters['even'].shape[-1] // 2)
        
        response_even = F.conv3d(video_sequence, self.filters['even'], padding=padding)
        response_odd = F.conv3d(video_sequence, self.filters['odd'], padding=padding)
        
        # Énergie de mouvement (carré et somme)
        energy = response_even**2 + response_odd**2
        
        # Prendre le maximum sur la dimension temporelle
        motion_energy, _ = energy.max(dim=2)  # (B, 1, H, W)
        
        return motion_energy.squeeze()


class DirectionSelectivity(nn.Module):
    """
    Sélectivité directionnelle - Population de détecteurs de mouvement.
    """
    
    def __init__(self,
                 n_directions: int = 8,
                 speeds: List[float] = None,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.n_directions = n_directions
        self.device = device
        
        self.speeds = speeds or [0.5, 1.0, 2.0, 4.0]
        
        # Créer une population de filtres de mouvement
        self.motion_filters = nn.ModuleList()
        self.directions = torch.linspace(0, 2 * math.pi, n_directions + 1)[:n_directions]
        
        for direction in self.directions:
            for speed in self.speeds:
                filter_cell = MotionEnergyFilter(
                    direction=direction.item(),
                    speed=speed,
                    device=device
                )
                self.motion_filters.append(filter_cell)
    
    def forward(self, video_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calcule les réponses directionnelles.
        
        Args:
            video_sequence: Séquence vidéo
            
        Returns:
            Réponses directionnelles, carte de direction, carte de vitesse
        """
        if len(video_sequence.shape) == 3:
            video_sequence = video_sequence.unsqueeze(0)  # (1, T, H, W)
        
        batch_size, time_steps, height, width = video_sequence.shape
        
        # Calculer les réponses de tous les filtres
        all_responses = []
        for filter_cell in self.motion_filters:
            response = filter_cell(video_sequence)
            if len(response.shape) == 2:
                response = response.unsqueeze(0)
            all_responses.append(response)
        
        # Stacker les réponses
        responses = torch.stack(all_responses, dim=1)  # (B, n_filters, H, W)
        
        # Reshaper pour séparer direction et vitesse
        n_filters_per_direction = len(self.speeds)
        responses_reshaped = responses.view(
            batch_size, self.n_directions, n_filters_per_direction, height, width
        )
        
        # Direction préférée (maximum sur toutes les vitesses)
        direction_responses, speed_idx = responses_reshaped.max(dim=2)  # (B, n_directions, H, W)
        direction_strength, direction_idx = direction_responses.max(dim=1)  # (B, H, W)
        
        # Vitesse préférée
        speed_responses, _ = responses_reshaped.max(dim=1)  # (B, n_speeds, H, W)
        speed_strength, speed_idx_flat = speed_responses.max(dim=1)  # (B, H, W)
        
        # Convertir les indices en valeurs
        direction_map = self.directions[direction_idx]
        speed_map = torch.tensor(self.speeds, device=self.device)[speed_idx_flat]
        
        # Vecteur de mouvement (direction et magnitude)
        motion_x = direction_strength * torch.cos(direction_map)
        motion_y = direction_strength * torch.sin(direction_map)
        
        return {
            'responses': responses,
            'direction_map': direction_map,
            'direction_strength': direction_strength,
            'speed_map': speed_map,
            'speed_strength': speed_strength,
            'motion_vector': torch.stack([motion_x, motion_y], dim=1),  # (B, 2, H, W)
            'n_directions': self.n_directions,
            'n_speeds': len(self.speeds)
        }


class MotionDetector(nn.Module):
    """
    Détecteur de mouvement - Combine plusieurs directions pour une détection robuste.
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int],
                 n_directions: int = 8,
                 pyramid_levels: int = 3,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.input_shape = input_shape
        self.n_directions = n_directions
        self.pyramid_levels = pyramid_levels
        self.device = device
        
        # Pyramide de détection à différentes échelles
        self.direction_selectivity = nn.ModuleList()
        
        for level in range(pyramid_levels):
            scale_factor = 2 ** level
            
            # Calculer la taille à cette échelle
            level_height = input_shape[0] // scale_factor
            level_width = input_shape[1] // scale_factor
            
            if level_height >= 16 and level_width >= 16:
                detector = DirectionSelectivity(
                    n_directions=n_directions,
                    device=device
                )
                self.direction_selectivity.append(detector)
            else:
                break
        
        # Intégration multi-échelle
        n_speeds = 4
        if len(self.direction_selectivity) > 0:
            n_speeds = len(self.direction_selectivity[0].speeds)
            
        self.integration = nn.Sequential(
            nn.Conv2d(self.n_directions * n_speeds * len(self.direction_selectivity), 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, kernel_size=3, padding=1),  # Flux optique (u, v)
            nn.Tanh()  # Normaliser entre -1 et 1
        )
    
    def forward(self, video_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Détecte le mouvement à plusieurs échelles.
        
        Args:
            video_sequence: Séquence vidéo (T, H, W) ou (B, T, H, W)
            
        Returns:
            Détection de mouvement multi-échelle
        """
        if len(video_sequence.shape) == 3:
            video_sequence = video_sequence.unsqueeze(0)
        
        batch_size, time_steps, height, width = video_sequence.shape
        
        all_level_responses = []
        
        for level, detector in enumerate(self.direction_selectivity):
            scale_factor = 2 ** level
            
            # Sous-échantillonner la séquence
            if scale_factor > 1:
                level_height = height // scale_factor
                level_width = width // scale_factor
                
                video_resized = F.interpolate(
                    video_sequence.view(batch_size * time_steps, 1, height, width),
                    size=(level_height, level_width),
                    mode='bilinear'
                ).view(batch_size, time_steps, level_height, level_width)
            else:
                video_resized = video_sequence
            
            # Détection de mouvement à cette échelle
            level_results = detector(video_resized)
            
            # Extraire les réponses directionnelles
            direction_responses = level_results['responses']
            
            # Ré-échantillonner à la taille d'origine
            direction_responses_resized = F.interpolate(
                direction_responses,
                size=(height, width),
                mode='bilinear'
            )
            
            all_level_responses.append(direction_responses_resized)
        
        # Combiner toutes les échelles
        if all_level_responses:
            combined = torch.cat(all_level_responses, dim=1)  # (B, n_directions * n_levels, H, W)
            
            # Intégration pour obtenir le flux optique
            optical_flow = self.integration(combined)  # (B, 2, H, W)
            
            # Séparer les composantes u et v
            flow_u = optical_flow[:, 0:1, :, :]
            flow_v = optical_flow[:, 1:2, :, :]
            
            # Magnitude et direction du flux
            flow_magnitude = torch.sqrt(flow_u**2 + flow_v**2 + 1e-8)
            flow_direction = torch.atan2(flow_v, flow_u)
            
            return {
                'optical_flow': optical_flow,
                'flow_u': flow_u,
                'flow_v': flow_v,
                'flow_magnitude': flow_magnitude,
                'flow_direction': flow_direction,
                'level_responses': all_level_responses,
                'n_levels': len(self.direction_selectivity)
            }
        else:
            # Retourner des zéros si aucune échelle valide
            zeros = torch.zeros(batch_size, 2, height, width, device=self.device)
            return {
                'optical_flow': zeros,
                'flow_u': zeros[:, 0:1, :, :],
                'flow_v': zeros[:, 1:2, :, :],
                'flow_magnitude': zeros[:, 0:1, :, :],
                'flow_direction': zeros[:, 0:1, :, :],
                'level_responses': [],
                'n_levels': 0
            }


class OpticalFlow(nn.Module):
    """
    Estimation de flux optique basée sur des gradients.
    Implémentation simplifiée de la méthode de Horn & Schunck.
    """
    
    def __init__(self,
                 smoothness_weight: float = 0.01,
                 n_iterations: int = 20,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.smoothness_weight = smoothness_weight
        self.n_iterations = n_iterations
        self.device = device
        
        # Filtres pour les gradients
        self.sobel_x = torch.tensor([[-1, 0, 1],
                                     [-2, 0, 2],
                                     [-1, 0, 1]], device=device).float().view(1, 1, 3, 3) / 8.0
        
        self.sobel_y = torch.tensor([[-1, -2, -1],
                                     [0, 0, 0],
                                     [1, 2, 1]], device=device).float().view(1, 1, 3, 3) / 8.0
        
        self.sobel_t = torch.ones(1, 1, 3, 3, device=device).float() / 9.0
        
        # Filtre de lissage (moyenne)
        self.smooth_filter = torch.ones(1, 1, 3, 3, device=device).float() / 9.0
    
    def compute_gradients(self, frame1: torch.Tensor, frame2: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Calcule les gradients spatio-temporels."""
        if len(frame1.shape) == 2:
            frame1 = frame1.unsqueeze(0).unsqueeze(0)
            frame2 = frame2.unsqueeze(0).unsqueeze(0)
        
        # Gradients spatiaux
        Ix = 0.5 * (F.conv2d(frame1, self.sobel_x, padding=1) + 
                    F.conv2d(frame2, self.sobel_x, padding=1))
        
        Iy = 0.5 * (F.conv2d(frame1, self.sobel_y, padding=1) + 
                    F.conv2d(frame2, self.sobel_y, padding=1))
        
        # Gradient temporel
        It = F.conv2d(frame2 - frame1, self.sobel_t, padding=1)
        
        return Ix, Iy, It
    
    def forward(self, frame1: torch.Tensor, frame2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Estime le flux optique entre deux frames.
        
        Args:
            frame1: Première frame
            frame2: Deuxième frame
            
        Returns:
            Flux optique estimé
        """
        if len(frame1.shape) == 2:
            frame1 = frame1.unsqueeze(0).unsqueeze(0)
            frame2 = frame2.unsqueeze(0).unsqueeze(0)
        
        batch_size, channels, height, width = frame1.shape
        
        # Si multi-canal, convertir en niveaux de gris
        if channels > 1:
            frame1_gray = 0.299 * frame1[:, 0:1, :, :] + 0.587 * frame1[:, 1:2, :, :] + 0.114 * frame1[:, 2:3, :, :]
            frame2_gray = 0.299 * frame2[:, 0:1, :, :] + 0.587 * frame2[:, 1:2, :, :] + 0.114 * frame2[:, 2:3, :, :]
        else:
            frame1_gray = frame1
            frame2_gray = frame2
        
        # Calculer les gradients
        Ix, Iy, It = self.compute_gradients(frame1_gray, frame2_gray)
        
        # Initialiser le flux optique
        u = torch.zeros(batch_size, 1, height, width, device=self.device)
        v = torch.zeros(batch_size, 1, height, width, device=self.device)
        
        # Itération de Horn & Schunck
        for _ in range(self.n_iterations):
            # Moyenne du flux voisin (lissage)
            u_avg = F.conv2d(u, self.smooth_filter, padding=1)
            v_avg = F.conv2d(v, self.smooth_filter, padding=1)
            
            # Mettre à jour le flux
            denominator = Ix**2 + Iy**2 + self.smoothness_weight
            u = u_avg - Ix * (Ix * u_avg + Iy * v_avg + It) / denominator
            v = v_avg - Iy * (Ix * u_avg + Iy * v_avg + It) / denominator
        
        # Calculer la magnitude et la direction
        magnitude = torch.sqrt(u**2 + v**2 + 1e-8)
        direction = torch.atan2(v, u)
        
        return {
            'flow_u': u,
            'flow_v': v,
            'flow_magnitude': magnitude,
            'flow_direction': direction,
            'gradients': {'Ix': Ix, 'Iy': Iy, 'It': It},
            'n_iterations': self.n_iterations
        }
