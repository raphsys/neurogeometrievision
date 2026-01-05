import torch
import numpy as np
from typing import Tuple, Optional

class SpikeEncoder:
    """
    Encode une image ou une réponse neuronale en trains d'impulsions (spikes).
    Implémente différentes stratégies de codage inspirées de la biologie.
    """
    
    @staticmethod
    def rate_coding(intensity: torch.Tensor, max_rate: float = 100.0, 
                   time_steps: int = 10) -> torch.Tensor:
        """
        Codage par fréquence (rate coding).
        Convertit l'intensité en probabilité de décharge par pas de temps.
        
        Args:
            intensity: Tensor de valeurs normalisées [0, 1]
            max_rate: Fréquence de décharge maximale (Hz)
            time_steps: Nombre de pas de temps simulés
            
        Returns:
            Tensor binaire de forme (time_steps, *intensity.shape)
        """
        # Normalise l'intensité pour qu'elle représente une fréquence
        intensity = torch.clamp(intensity, 0, 1)
        
        # Calcule la probabilité de décharge par pas de temps
        # (en supposant un pas de temps de 10ms)
        dt = 0.01  # 10ms
        prob = intensity * max_rate * dt
        
        # Génère des spikes aléatoires
        spikes = torch.rand(time_steps, *intensity.shape) < prob
        
        return spikes.float()
    
    @staticmethod
    def rank_coding(intensity: torch.Tensor, time_steps: int = 10) -> torch.Tensor:
        """
        Codage par rang (rank coding) à la Simon Thorpe.
        Les pixels avec les intensités les plus fortes déchargent en premier.
        
        Args:
            intensity: Tensor de valeurs normalisées [0, 1]
            time_steps: Nombre de pas de temps
            
        Returns:
            Tensor binaire de forme (time_steps, *intensity.shape)
        """
        intensity = torch.clamp(intensity, 0, 1)
        spikes = torch.zeros(time_steps, *intensity.shape)
        
        # Flatten pour le traitement
        flat_intensity = intensity.flatten()
        n_pixels = flat_intensity.numel()
        
        # Trie les pixels par intensité décroissante
        sorted_indices = torch.argsort(flat_intensity, descending=True)
        
        # Assigne un temps de décharge basé sur le rang
        for rank, idx in enumerate(sorted_indices):
            time = int((rank / n_pixels) * time_steps)
            if time < time_steps:
                # Convertit l'indice plat en indices multidimensionnels
                idx_tuple = np.unravel_index(idx.item(), intensity.shape)
                spikes_idx = (time,) + idx_tuple
                spikes[spikes_idx] = 1.0
        
        return spikes
    
    @staticmethod
    def latency_coding(intensity: torch.Tensor, max_latency: float = 50.0, 
                      time_steps: int = 10) -> torch.Tensor:
        """
        Codage par latence (latency coding).
        L'intensité détermine le délai avant la première décharge.
        """
        intensity = torch.clamp(intensity, 0, 1)
        spikes = torch.zeros(time_steps, *intensity.shape)
        
        # Calcule la latence (en pas de temps)
        # Forte intensité = courte latence
        latencies = (1.0 - intensity) * max_latency
        
        # Arrondit à l'entier le plus proche pour les indices de temps
        time_indices = torch.round(latencies * (time_steps - 1) / max_latency).long()
        
        # Génère un spike au temps calculé
        for idx in torch.nonzero(time_indices < time_steps, as_tuple=True):
            t = time_indices[idx].item()
            spikes_idx = (t,) + tuple(i.item() for i in idx)
            spikes[spikes_idx] = 1.0
        
        return spikes


class TemporalProcessor:
    """
    Traitement temporel simplifié pour simuler la dynamique des réponses neuronales.
    """
    
    def __init__(self, tau: float = 20.0, dt: float = 1.0, device: str = 'cpu'):
        """
        Args:
            tau: Constante de temps membranaire (ms)
            dt: Pas de temps (ms)
            device: 'cpu' ou 'cuda'
        """
        self.tau = tau
        self.dt = dt
        self.device = device
        # CORRECTION : Utiliser torch.tensor au lieu de flottants
        self.alpha = torch.exp(torch.tensor(-dt / tau, device=device))
    
    def leaky_integrate(self, spikes: torch.Tensor, init_v: float = 0.0) -> torch.Tensor:
        """
        Intégration à fuite des spikes pour simuler le potentiel membranaire.
        
        Args:
            spikes: Tensor de forme (time_steps, ...)
            init_v: Potentiel initial
            
        Returns:
            Potentiel membranaire à chaque pas de temps
        """
        time_steps = spikes.shape[0]
        shape = spikes.shape[1:]
        
        # Initialise le potentiel
        v = torch.full(shape, init_v, dtype=torch.float32, device=self.device)
        voltages = []
        
        for t in range(time_steps):
            # Équation différentielle: dv/dt = -v/tau + input
            v = self.alpha * v + spikes[t]
            voltages.append(v.clone())
        
        return torch.stack(voltages)
    
    def spike_response_model(self, spikes: torch.Tensor, tau_syn: float = 5.0) -> torch.Tensor:
        """
        Modèle de réponse aux spikes plus sophistiqué avec noyau exponentiel.
        
        Args:
            spikes: Tensor de spikes
            tau_syn: Constante de temps synaptique (ms)
            
        Returns:
            Potentiel postsynaptique
        """
        time_steps = spikes.shape[0]
        shape = spikes.shape[1:]
        
        # Noyau exponentiel pour la réponse synaptique
        kernel = torch.zeros(time_steps, device=self.device)
        for t in range(time_steps):
            if t >= 0:
                kernel[t] = torch.exp(torch.tensor(-t * self.dt / tau_syn, device=self.device))
        
        # Convolution 1D le long de l'axe temporel
        response = torch.zeros_like(spikes)
        for i in range(spikes.shape[1]):
            for j in range(spikes.shape[2]):
                response[:, i, j] = torch.conv1d(
                    spikes[:, i, j].unsqueeze(0).unsqueeze(0),
                    kernel.flip(0).unsqueeze(0).unsqueeze(0),
                    padding=time_steps-1
                )[0, 0, :time_steps]
        
        return response
