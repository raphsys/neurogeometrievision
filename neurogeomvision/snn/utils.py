"""
Module utils.py - Utilitaires pour SNN
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt


def encode_image_to_spikes(image: torch.Tensor,
                          n_timesteps: int = 10,
                          max_rate: float = 100.0) -> torch.Tensor:
    """
    Encode une image en trains de spikes (rate coding).
    
    Args:
        image: Image à encoder (H, W) ou (C, H, W)
        n_timesteps: Nombre de pas de temps
        max_rate: Fréquence maximale (Hz)
        
    Returns:
        Spikes: (n_timesteps, channels, height, width)
    """
    if len(image.shape) == 2:
        image = image.unsqueeze(0)  # (1, H, W)
    
    channels, height, width = image.shape
    
    # Normalisation
    image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # Génération de spikes
    spikes = torch.zeros(n_timesteps, channels, height, width)
    
    for t in range(n_timesteps):
        # Probabilité proportionnelle à l'intensité
        probabilities = image_norm * (max_rate / 1000.0) * n_timesteps
        spikes[t] = (torch.rand_like(probabilities) < probabilities).float()
    
    return spikes


def calculate_spike_stats(spikes: torch.Tensor) -> Dict:
    """
    Calcule des statistiques sur les spikes.
    
    Args:
        spikes: Tensors de spikes
        
    Returns:
        Dictionnaire de statistiques
    """
    if len(spikes.shape) == 4:
        # (time, channels, height, width) -> (time, neurons)
        n_time, n_channels, height, width = spikes.shape
        spikes_flat = spikes.view(n_time, n_channels * height * width)
    elif len(spikes.shape) == 2:
        spikes_flat = spikes
    else:
        raise ValueError(f"Shape {spikes.shape} non supporté")
    
    n_time, n_neurons = spikes_flat.shape
    
    # Taux de décharge
    firing_rates = spikes_flat.mean(dim=0) * 1000.0  # Hz
    
    # Statistiques
    stats = {
        'mean_firing_rate': firing_rates.mean().item(),
        'max_firing_rate': firing_rates.max().item(),
        'min_firing_rate': firing_rates.min().item(),
        'total_spikes': spikes_flat.sum().item(),
        'n_neurons': n_neurons,
        'n_timesteps': n_time,
        'firing_rates': firing_rates
    }
    
    return stats


def visualize_spike_train(spikes: torch.Tensor,
                         title: str = "Spike Train",
                         save_path: Optional[str] = None):
    """
    Visualise un train de spikes.
    
    Args:
        spikes: Tensors de spikes
        title: Titre du graphique
        save_path: Chemin pour sauvegarder
    """
    if len(spikes.shape) == 4:
        # (time, channels, height, width) -> (time, neurons)
        n_time, n_channels, height, width = spikes.shape
        spikes_vis = spikes.view(n_time, n_channels * height * width)
    elif len(spikes.shape) == 2:
        spikes_vis = spikes
    else:
        raise ValueError(f"Shape {spikes.shape} non supporté")
    
    n_time, n_neurons = spikes_vis.shape
    
    # Limite pour l'affichage
    max_neurons_show = min(50, n_neurons)
    spikes_show = spikes_vis[:, :max_neurons_show]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Raster plot
    spike_times, neuron_ids = torch.where(spikes_show > 0.5)
    if len(spike_times) > 0:
        axes[0].scatter(spike_times.numpy(), neuron_ids.numpy(), s=1, color='black')
    axes[0].set_xlabel('Time step')
    axes[0].set_ylabel('Neuron')
    axes[0].set_title('Raster Plot')
    axes[0].grid(True, alpha=0.3)
    
    # Firing rates
    firing_rates = spikes_show.mean(dim=0) * 1000.0
    axes[1].bar(range(len(firing_rates)), firing_rates.numpy())
    axes[1].set_xlabel('Neuron')
    axes[1].set_ylabel('Firing rate (Hz)')
    axes[1].set_title(f'Mean: {firing_rates.mean():.1f} Hz')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    
    return fig
