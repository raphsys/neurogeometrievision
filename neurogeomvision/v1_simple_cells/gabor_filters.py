"""
Module gabor_filters.py - Banque de filtres de Gabor orientés
VERSION COMPLÈTEMENT OPTIMISÉE
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
import math


class GaborFilterBank:
    """
    Banque de filtres de Gabor pour détecter les orientations dans V1.
    VERSION OPTIMISÉE avec pré-calcul et convolution batch.
    """
    
    def __init__(self, 
                 img_size: Tuple[int, int],
                 n_orientations: int = 8,
                 spatial_freqs: List[float] = None,
                 phases: List[float] = None,
                 device: str = 'cpu'):
        """
        Args:
            img_size: (height, width) de l'image
            n_orientations: Nombre d'orientations
            spatial_freqs: Fréquences spatiales
            phases: Phases (en radians)
            device: 'cpu' ou 'cuda'
        """
        self.img_size = img_size
        self.height, self.width = img_size
        self.n_orientations = n_orientations
        self.device = device
        
        if spatial_freqs is None:
            spatial_freqs = [0.1, 0.2, 0.3]
        self.spatial_freqs = spatial_freqs

        if phases is None:
            phases = [0.0]
        self.phases = phases
        
        # Paramètres des filtres
        self.sigma_x = 3.0
        self.sigma_y = 1.5
        
        # PRÉ-CALCULE TOUS LES FILTRES (optimisation clé)
        self.filters, self.filter_metadata = self._precompute_all_filters()
    
    def _precompute_all_filters(self) -> Tuple[List[torch.Tensor], List[Dict]]:
        """Pré-calcule tous les filtres de Gabor."""
        filters_list = []
        metadata_list = []
        
        for freq in self.spatial_freqs:
            for phi in self.phases:
                for orientation_idx in range(self.n_orientations):
                    theta = orientation_idx * math.pi / self.n_orientations
                    
                    # Crée le filtre
                    gabor_filter = self._create_gabor_filter_fast(theta, freq, phi)
                    
                    filters_list.append(gabor_filter)
                    metadata_list.append({
                        'theta': theta,
                        'freq': freq,
                        'phase': phi,
                        'orientation_idx': orientation_idx
                    })
        
        return filters_list, metadata_list
    
    def _create_gabor_filter_fast(self,
                                 theta: float,
                                 freq: float,
                                 phi: float = 0.0) -> torch.Tensor:
        """
        Crée un filtre de Gabor 2D - VERSION OPTIMISÉE avec broadcasting.
        """
        # Taille basée sur les sigmas
        filter_size = int(2 * max(self.sigma_x, self.sigma_y) * 3) + 1
        if filter_size % 2 == 0:
            filter_size += 1
        
        half = filter_size // 2
        
        # Crée les coordonnées avec broadcasting (PLUS RAPIDE que meshgrid)
        y_coords, x_coords = torch.meshgrid(
            torch.arange(-half, half + 1, device=self.device).float(),
            torch.arange(-half, half + 1, device=self.device).float(),
            indexing='ij'
        )
        
        # Rotation des coordonnées (vectorisé)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        x_theta = x_coords * cos_theta + y_coords * sin_theta
        y_theta = -x_coords * sin_theta + y_coords * cos_theta
        
        # Terme gaussien (vectorisé)
        gaussian = torch.exp(
            -0.5 * (x_theta**2 / self.sigma_x**2 + y_theta**2 / self.sigma_y**2)
        )
        
        # Terme sinusoïdal (vectorisé)
        sinusoidal = torch.cos(2 * math.pi * freq * x_theta + phi)
        
        # Filtre complet
        gabor = gaussian * sinusoidal
        
        # Normalise pour somme nulle
        gabor = gabor - gabor.mean()
        
        # Normalise l'énergie
        energy = torch.sqrt(torch.sum(gabor**2))
        if energy > 1e-8:
            gabor = gabor / energy
        
        return gabor
    
    def apply_filters(self, image: torch.Tensor) -> Dict:
        """
        Applique tous les filtres de Gabor à une image - VERSION OPTIMISÉE.
        Utilise la convolution batch pour une performance maximale.
        """
        # Normalise l'image
        if len(image.shape) == 3:
            image = image.mean(dim=0)  # Convertit RGB en luminance
        
        image = image.to(self.device)
        
        # Prépare l'image pour la convolution
        image_4d = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        # Prépare les filtres en batch
        if not self.filters:
            raise ValueError("Aucun filtre pré-calculé")
        
        # Taille des filtres
        filter_h, filter_w = self.filters[0].shape
        filters_batch = torch.stack(self.filters).unsqueeze(1)  # (n_filters, 1, Hf, Wf)
        
        # Padding
        pad_h, pad_w = filter_h // 2, filter_w // 2
        
        # CONVOLUTION BATCH (TRÈS RAPIDE)
        start_time = torch.cuda.Event(enable_timing=True) if self.device == 'cuda' else None
        end_time = torch.cuda.Event(enable_timing=True) if self.device == 'cuda' else None
        
        if self.device == 'cuda':
            start_time.record()
        
        responses = torch.nn.functional.conv2d(
            torch.nn.functional.pad(image_4d, (pad_w, pad_w, pad_h, pad_h), mode='reflect'),
            filters_batch,
            padding=0
        )  # (1, n_filters, H, W)
        
        if self.device == 'cuda':
            end_time.record()
            torch.cuda.synchronize()
        
        responses = responses.squeeze(0)  # (n_filters, H, W)
        
        # Trouve l'orientation dominante pour chaque pixel (OPTIMISÉ)
        n_filters = responses.shape[0]
        h, w = responses.shape[1:]
        
        # Utilise torch.max pour la performance
        response_abs = responses.abs()
        max_response, best_filter_idx = torch.max(response_abs, dim=0)
        
        # Crée les cartes d'orientation et de fréquence (vectorisé)
        orientation_map = torch.zeros(h, w, device=self.device)
        frequency_map = torch.zeros(h, w, device=self.device)
        orientation_idx_map = torch.zeros(h, w, device=self.device, dtype=torch.long)
        
        # Remplit les cartes avec l'indexation vectorisée
        for i in range(n_filters):
            mask = (best_filter_idx == i)
            if mask.any():
                orientation_map[mask] = self.filter_metadata[i]['theta']
                frequency_map[mask] = self.filter_metadata[i]['freq']
                orientation_idx_map[mask] = self.filter_metadata[i]['orientation_idx']
        
        # Calcule la cohérence locale d'orientation
        orientation_coherence = self._compute_orientation_coherence(orientation_idx_map)
        
        # Prépare les réponses individuelles pour la compatibilité (si nécessaire)
        filter_responses = {}
        # Pour limiter la mémoire, on ne stocke que quelques réponses si demandé, 
        # mais ici on va en mettre quelques-unes pour le test
        for i in range(min(5, n_filters)):
            filter_responses[f'filter_{i}'] = {
                'response': responses[i],
                'params': self.filter_metadata[i]
            }
        
        return {
            'responses': responses,  # (n_filters, H, W)
            'dominant_orientation': {
                'angle': orientation_map,
                'amplitude': max_response,
                'index': orientation_idx_map,
                'coherence': orientation_coherence
            },
            'orientation_map': orientation_map,
            'dominant_frequency': frequency_map,
            'orientation_coherence': orientation_coherence,
            'filter_responses': filter_responses,
            'filter_bank': self.filters,
            'metadata': self.filter_metadata
        }
    
    def _compute_orientation_coherence(self, orientation_idx_map: torch.Tensor) -> torch.Tensor:
        """
        Calcule la cohérence locale d'orientation - VERSION OPTIMISÉE.
        """
        h, w = orientation_idx_map.shape
        
        # Convertit les indices en vecteurs d'orientation
        angles = orientation_idx_map.float() * math.pi / self.n_orientations
        cos_2theta = torch.cos(2 * angles)
        sin_2theta = torch.sin(2 * angles)
        
        # Filtre gaussien pour le lissage
        kernel_size = 5
        sigma = 1.0
        half = kernel_size // 2
        
        # Crée le noyau gaussien
        coords = torch.arange(-half, half + 1, device=self.device).float()
        x = coords.view(1, -1)
        y = coords.view(-1, 1)
        gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        gaussian = gaussian / gaussian.sum()
        
        # Lisse les composantes
        cos_2theta_4d = cos_2theta.unsqueeze(0).unsqueeze(0)
        sin_2theta_4d = sin_2theta.unsqueeze(0).unsqueeze(0)
        
        cos_smoothed = torch.nn.functional.conv2d(
            torch.nn.functional.pad(cos_2theta_4d, (half, half, half, half), mode='reflect'),
            gaussian.unsqueeze(0).unsqueeze(0),
            padding=0
        ).squeeze()
        
        sin_smoothed = torch.nn.functional.conv2d(
            torch.nn.functional.pad(sin_2theta_4d, (half, half, half, half), mode='reflect'),
            gaussian.unsqueeze(0).unsqueeze(0),
            padding=0
        ).squeeze()
        
        # Cohérence = magnitude du vecteur moyen
        coherence = torch.sqrt(cos_smoothed**2 + sin_smoothed**2)
        
        return coherence
    
    def get_orientation_histogram(self, orientation_map: torch.Tensor) -> torch.Tensor:
        """
        Calcule l'histogramme des orientations - OPTIMISÉ avec torch.bincount.
        """
        n_bins = 36
        
        # Discrétise les orientations
        bins = torch.linspace(0, math.pi, n_bins + 1, device=self.device)
        
        # Utilise bucketize pour une discrétisation rapide
        flat_orientations = orientation_map.flatten()
        indices = torch.bucketize(flat_orientations, bins)
        indices = torch.clamp(indices - 1, 0, n_bins - 1)
        
        # Compte avec bincount (optimisé en C++)
        histogram = torch.bincount(indices, minlength=n_bins).float()
        
        # Normalise
        if histogram.sum() > 0:
            histogram = histogram / histogram.sum()
        
        return histogram
    
    def get_filter_summary(self) -> Dict:
        """Retourne un résumé des paramètres de la banque de filtres."""
        orientations = [m['theta'] for m in self.filter_metadata]
        frequencies = [m['freq'] for m in self.filter_metadata]
        sizes = [f.shape[0] for f in self.filters]
        
        return {
            'total_filters': len(self.filters),
            'orientations': {
                'min': min(orientations),
                'max': max(orientations),
                'count': self.n_orientations
            },
            'frequencies': {
                'min': min(frequencies),
                'max': max(frequencies),
                'count': len(self.spatial_freqs)
            },
            'sizes': {
                'min': min(sizes),
                'max': max(sizes)
            }
        }

    def visualize_filters(self, n_filters: int = 12):
        """
        Visualise un sous-ensemble des filtres.
        """
        import matplotlib.pyplot as plt
        
        n_filters_total = len(self.filters)
        n_cols = min(4, n_filters)
        n_rows = (n_filters + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        for idx in range(min(n_filters, n_filters_total)):
            ax = axes[idx]
            filter_img = self.filters[idx].cpu().numpy()
            
            im = ax.imshow(filter_img, cmap='RdBu_r')
            ax.set_title(f"θ={self.filter_metadata[idx]['theta']*180/math.pi:.0f}°, f={self.filter_metadata[idx]['freq']:.2f}")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Cache les axes non utilisés
        for idx in range(min(n_filters, n_filters_total), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f"Filtres de Gabor ({n_filters_total} au total)", fontsize=12)
        plt.tight_layout()
        
        return fig
