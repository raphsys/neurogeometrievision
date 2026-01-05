"""
Module retinal_processing.py - Fonctions de traitement rétinien
Filtrage, traitement centre-surround, réponse temporelle, normalisation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional, Union, Callable
import math


def retinal_filtering(image: torch.Tensor,
                      filter_type: str = 'dog',
                      filter_size: int = 15,
                      sigma_center: float = 2.0,
                      sigma_surround: float = 5.0,
                      device: str = 'cpu') -> torch.Tensor:
    """
    Applique un filtrage rétinien à une image.
    
    Args:
        image: Image d'entrée (H, W) ou (C, H, W)
        filter_type: Type de filtre ('dog', 'gaussian', 'laplacian')
        filter_size: Taille du filtre
        sigma_center: Sigma du centre (pour DoG)
        sigma_surround: Sigma du surround (pour DoG)
        device: Device
        
    Returns:
        Image filtrée
    """
    if len(image.shape) == 2:
        image = image.unsqueeze(0)  # (1, H, W)
    
    channels, height, width = image.shape
    
    # Créer le filtre
    if filter_type == 'dog':
        filter_kernel = _create_dog_filter(filter_size, sigma_center, sigma_surround, device)
    elif filter_type == 'gaussian':
        filter_kernel = _create_gaussian_filter(filter_size, sigma_center, device)
    elif filter_type == 'laplacian':
        filter_kernel = _create_laplacian_filter(filter_size, device)
    else:
        raise ValueError(f"Type de filtre inconnu: {filter_type}")
    
    # Appliquer la convolution à chaque canal
    filtered = []
    for c in range(channels):
        channel_input = image[c:c+1].unsqueeze(0)  # (1, 1, H, W)
        
        if height >= filter_size and width >= filter_size:
            channel_filtered = F.conv2d(
                channel_input,
                filter_kernel,
                padding=filter_size // 2
            ).squeeze()
        else:
            channel_filtered = channel_input.squeeze()
        
        filtered.append(channel_filtered)
    
    if channels == 1:
        return filtered[0]
    else:
        return torch.stack(filtered, dim=0)


def _create_dog_filter(size: int, sigma_center: float, sigma_surround: float, device: str) -> torch.Tensor:
    """Crée un filtre DoG (Difference of Gaussians)."""
    center = size // 2
    
    y, x = torch.meshgrid(
        torch.arange(size, device=device) - center,
        torch.arange(size, device=device) - center,
        indexing='ij'
    )
    
    r = torch.sqrt(x**2 + y**2)
    
    # Gaussienne centre
    center_gauss = torch.exp(-r**2 / (2 * sigma_center**2))
    center_gauss = center_gauss / center_gauss.sum()
    
    # Gaussienne surround
    surround_gauss = torch.exp(-r**2 / (2 * sigma_surround**2))
    surround_gauss = surround_gauss / surround_gauss.sum()
    
    # DoG
    dog_filter = center_gauss - 0.7 * surround_gauss
    
    return dog_filter.unsqueeze(0).unsqueeze(0)


def _create_gaussian_filter(size: int, sigma: float, device: str) -> torch.Tensor:
    """Crée un filtre gaussien."""
    center = size // 2
    
    y, x = torch.meshgrid(
        torch.arange(size, device=device) - center,
        torch.arange(size, device=device) - center,
        indexing='ij'
    )
    
    r = torch.sqrt(x**2 + y**2)
    gaussian = torch.exp(-r**2 / (2 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    
    return gaussian.unsqueeze(0).unsqueeze(0)


def _create_laplacian_filter(size: int, device: str) -> torch.Tensor:
    """Crée un filtre Laplacien."""
    center = size // 2
    
    y, x = torch.meshgrid(
        torch.arange(size, device=device) - center,
        torch.arange(size, device=device) - center,
        indexing='ij'
    )
    
    r = torch.sqrt(x**2 + y**2)
    
    # Laplacien de Gaussian (approximé)
    sigma = size / 4.0
    log_filter = (r**2 - 2 * sigma**2) / (sigma**4) * torch.exp(-r**2 / (2 * sigma**2))
    log_filter = log_filter / log_filter.abs().sum()
    
    return log_filter.unsqueeze(0).unsqueeze(0)


def center_surround_processing(image: torch.Tensor,
                              center_size: int = 5,
                              surround_size: int = 15,
                              on_off_balance: float = 0.5,
                              device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """
    Traitement centre-surround (réponses ON et OFF).
    
    Args:
        image: Image d'entrée
        center_size: Taille du centre
        surround_size: Taille du surround
        on_off_balance: Balance entre ON et OFF (0=OFF seulement, 1=ON seulement)
        device: Device
        
    Returns:
        Dictionnaire avec réponses ON et OFF
    """
    if len(image.shape) == 2:
        image = image.unsqueeze(0)  # (1, H, W)
    
    channels, height, width = image.shape
    
    # Filtres centre et surround
    center_filter = _create_gaussian_filter(center_size, center_size/3.0, device)
    surround_filter = _create_gaussian_filter(surround_size, surround_size/4.0, device)
    
    # Traiter chaque canal
    on_responses = []
    off_responses = []
    
    for c in range(channels):
        channel_input = image[c:c+1].unsqueeze(0)  # (1, 1, H, W)
        
        # Réponse du centre
        if height >= center_size and width >= center_size:
            center_response = F.conv2d(
                channel_input,
                center_filter,
                padding=center_size // 2
            ).squeeze()
        else:
            center_response = channel_input.squeeze()
        
        # Réponse du surround
        if height >= surround_size and width >= surround_size:
            surround_response = F.conv2d(
                channel_input,
                surround_filter,
                padding=surround_size // 2
            ).squeeze()
        else:
            surround_response = channel_input.mean() * torch.ones_like(channel_input.squeeze())
        
        # Réponses ON et OFF
        # ON = centre - surround (réponse aux augmentations)
        # OFF = surround - centre (réponse aux diminutions)
        
        on_response = torch.relu(center_response - 0.7 * surround_response)
        off_response = torch.relu(0.7 * surround_response - center_response)
        
        # Ajuster la balance
        on_response = on_response * on_off_balance
        off_response = off_response * (1.0 - on_off_balance)
        
        on_responses.append(on_response)
        off_responses.append(off_response)
    
    if channels == 1:
        on_output = on_responses[0]
        off_output = off_responses[0]
    else:
        on_output = torch.stack(on_responses, dim=0)
        off_output = torch.stack(off_responses, dim=0)
    
    return {
        'on_response': on_output,
        'off_response': off_output,
        'center_response': center_response if channels == 1 else None,
        'surround_response': surround_response if channels == 1 else None
    }


def temporal_response(input_signal: torch.Tensor,
                     tau_fast: float = 10.0,
                     tau_slow: float = 50.0,
                     dt: float = 1.0,
                     device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """
    Simule la réponse temporelle des cellules rétiniennes.
    
    Args:
        input_signal: Signal d'entrée (peut avoir dimensions temporelles)
        tau_fast: Constante de temps rapide (ms)
        tau_slow: Constante de temps lente (ms)
        dt: Pas de temps (ms)
        device: Device
        
    Returns:
        Dictionnaire avec réponses temporelles
    """
    if len(input_signal.shape) == 2:
        # Ajouter une dimension temporelle
        input_signal = input_signal.unsqueeze(0)  # (1, H, W)
    
    # Initialiser les états temporels
    fast_state = torch.zeros_like(input_signal[0], device=device)
    slow_state = torch.zeros_like(input_signal[0], device=device)
    
    # Simuler la réponse temporelle
    fast_responses = []
    slow_responses = []
    transient_responses = []
    sustained_responses = []
    
    for t in range(input_signal.shape[0]):
        current_input = input_signal[t]
        
        # Filtres temporels (simples filtres exponentiels)
        alpha_fast = math.exp(-dt / tau_fast)
        alpha_slow = math.exp(-dt / tau_slow)
        
        fast_state = alpha_fast * fast_state + (1 - alpha_fast) * current_input
        slow_state = alpha_slow * slow_state + (1 - alpha_slow) * current_input
        
        # Composantes transitoire et soutenue
        transient = fast_state - slow_state  # Réponse transitoire
        sustained = slow_state  # Réponse soutenue
        
        fast_responses.append(fast_state.clone())
        slow_responses.append(slow_state.clone())
        transient_responses.append(transient.clone())
        sustained_responses.append(sustained.clone())
    
    # Convertir en tenseurs
    fast_responses = torch.stack(fast_responses, dim=0)
    slow_responses = torch.stack(slow_responses, dim=0)
    transient_responses = torch.stack(transient_responses, dim=0)
    sustained_responses = torch.stack(sustained_responses, dim=0)
    
    return {
        'fast_response': fast_responses,
        'slow_response': slow_responses,
        'transient_response': transient_responses,
        'sustained_response': sustained_responses,
        'tau_fast': tau_fast,
        'tau_slow': tau_slow
    }


def contrast_normalization(image: torch.Tensor,
                          local_window_size: int = 20,
                          epsilon: float = 1e-6,
                          device: str = 'cpu') -> torch.Tensor:
    """
    Normalisation de contraste locale (similaire à la rétine biologique).
    
    Args:
        image: Image d'entrée
        local_window_size: Taille de la fenêtre locale
        epsilon: Petite valeur pour éviter la division par zéro
        device: Device
        
    Returns:
        Image normalisée
    """
    if len(image.shape) == 2:
        image = image.unsqueeze(0)  # (1, H, W)
    
    channels, height, width = image.shape
    
    # S'assurer que local_window_size est impair pour un padding symétrique
    if local_window_size % 2 == 0:
        local_window_size += 1  # Rendre impair
    
    normalized = []
    
    for c in range(channels):
        channel_input = image[c:c+1].unsqueeze(0)  # (1, 1, H, W)
        
        # Calculer la moyenne locale
        if height >= local_window_size and width >= local_window_size:
            # Padding pour average pooling
            padding = local_window_size // 2  # Maintenant symétrique car impair
            
            # Average pooling pour la moyenne
            local_mean = F.avg_pool2d(
                F.pad(channel_input, (padding, padding, padding, padding), mode='reflect'),
                kernel_size=local_window_size,
                stride=1
            )
            
            # Average pooling pour le carré moyen
            local_sq_mean = F.avg_pool2d(
                F.pad(channel_input ** 2, (padding, padding, padding, padding), mode='reflect'),
                kernel_size=local_window_size,
                stride=1
            )
            
            # Calculer variance et écart-type
            local_variance = local_sq_mean - local_mean ** 2
            local_std = torch.sqrt(torch.clamp(local_variance, min=0.0) + epsilon)
            
            # Normalisation
            channel_normalized = (channel_input - local_mean) / (local_std + epsilon)
        else:
            # Normalisation globale
            channel_mean = channel_input.mean()
            channel_std = channel_input.std()
            channel_normalized = (channel_input - channel_mean) / (channel_std + epsilon)
        
        # Limiter les valeurs extrêmes
        channel_normalized = torch.tanh(channel_normalized)
        
        normalized.append(channel_normalized.squeeze())
    
    if channels == 1:
        return normalized[0]
    else:
        return torch.stack(normalized, dim=0)
        

def retinal_adaptation(image: torch.Tensor,
                      adaptation_rate: float = 0.1,
                      light_level: float = 1.0,
                      dt: float = 1.0,
                      device: str = 'cpu') -> torch.Tensor:
    """
    Simule l'adaptation rétinienne à différents niveaux de lumière.
    
    Args:
        image: Image d'entrée
        adaptation_rate: Taux d'adaptation
        light_level: Niveau de lumière (0=obscurité, 1=pleine lumière)
        dt: Pas de temps
        device: Device
        
    Returns:
        Image adaptée
    """
    if len(image.shape) == 2:
        image = image.unsqueeze(0)  # (1, H, W)
    
    # Niveau d'adaptation (simule la sensibilité des photorécepteurs)
    # Plus la lumière est forte, plus la sensibilité diminue
    target_sensitivity = 1.0 / (1.0 + light_level * 10.0)
    
    # Filtrer pour simuler l'adaptation
    adapted_image = image * target_sensitivity
    
    # Compensation non-linéaire (simule les mécanismes d'adaptation)
    # Fonction de compression logarithmique approximative
    adapted_image = torch.log(1.0 + adapted_image * 10.0) / torch.log(torch.tensor(11.0))
    
    return adapted_image.squeeze() if image.shape[0] == 1 else adapted_image


def create_retinal_processing_pipeline(steps: List[str] = None,
                                      device: str = 'cpu') -> Callable:
    """
    Crée un pipeline de traitement rétinien configurable.
    
    Args:
        steps: Liste des étapes de traitement
        device: Device
        
    Returns:
        Fonction de traitement
    """
    if steps is None:
        steps = ['filtering', 'center_surround', 'normalization', 'adaptation']
    
    def pipeline(image: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Applique le pipeline de traitement.
        
        Args:
            image: Image d'entrée
            **kwargs: Paramètres optionnels pour chaque étape
            
        Returns:
            Résultats du traitement
        """
        results = {'input': image}
        current = image
        
        for step in steps:
            if step == 'filtering':
                filtered = retinal_filtering(
                    current,
                    filter_type=kwargs.get('filter_type', 'dog'),
                    filter_size=kwargs.get('filter_size', 15),
                    device=device
                )
                results['filtered'] = filtered
                current = filtered
            
            elif step == 'center_surround':
                cs_results = center_surround_processing(
                    current,
                    center_size=kwargs.get('center_size', 5),
                    surround_size=kwargs.get('surround_size', 15),
                    on_off_balance=kwargs.get('on_off_balance', 0.5),
                    device=device
                )
                results.update(cs_results)
                # Utiliser la réponse ON comme sortie par défaut
                current = cs_results['on_response']
            
            elif step == 'normalization':
                normalized = contrast_normalization(
                    current,
                    local_window_size=kwargs.get('local_window_size', 20),
                    device=device
                )
                results['normalized'] = normalized
                current = normalized
            
            elif step == 'adaptation':
                adapted = retinal_adaptation(
                    current,
                    adaptation_rate=kwargs.get('adaptation_rate', 0.1),
                    light_level=kwargs.get('light_level', 1.0),
                    device=device
                )
                results['adapted'] = adapted
                current = adapted
            
            elif step == 'temporal':
                # Pour les séquences temporelles
                if 'temporal_input' in kwargs:
                    temporal_results = temporal_response(
                        kwargs['temporal_input'],
                        tau_fast=kwargs.get('tau_fast', 10.0),
                        tau_slow=kwargs.get('tau_slow', 50.0),
                        device=device
                    )
                    results.update(temporal_results)
                    # Utiliser la réponse soutenue
                    current = temporal_results['sustained_response'][-1]
            
            else:
                raise ValueError(f"Étape inconnue: {step}")
        
        results['output'] = current
        return results
    
    return pipeline
