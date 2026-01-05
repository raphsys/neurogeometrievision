"""
Module learning.py - Algorithmes d'apprentissage pour SNN
"""

import torch
import torch.nn as nn
import math


class STDP_SNN(nn.Module):
    """
    Spike-Timing Dependent Plasticity pour SNN.
    """
    
    def __init__(self,
                 pre_size: int,
                 post_size: int,
                 A_plus: float = 0.01,
                 A_minus: float = 0.0105,
                 tau_plus: float = 20.0,
                 tau_minus: float = 20.0,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.pre_size = pre_size
        self.post_size = post_size
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.device = device
        
        # Poids
        self.weights = nn.Parameter(
            torch.rand(post_size, pre_size, device=device) * 0.1
        )
        
        # Traces
        self.register_buffer('x_trace', torch.zeros(pre_size, device=device))
        self.register_buffer('y_trace', torch.zeros(post_size, device=device))
    
    def reset_traces(self):
        """Réinitialise les traces."""
        self.x_trace = torch.zeros(self.pre_size, device=self.device)
        self.y_trace = torch.zeros(self.post_size, device=self.device)
    
    def stdp_update(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, dt: float = 1.0):
        """
        Mise à jour STDP.
        
        Args:
            pre_spikes: Spikes pré-synaptiques
            post_spikes: Spikes post-synaptiques
            dt: Pas de temps
        """
        # Mise à jour des traces
        self.x_trace = self.x_trace * math.exp(-dt / self.tau_plus) + pre_spikes
        self.y_trace = self.y_trace * math.exp(-dt / self.tau_minus) + post_spikes
        
        # Changement de poids
        delta_w = (self.A_plus * torch.outer(post_spikes, self.x_trace) -
                   self.A_minus * torch.outer(self.y_trace, pre_spikes))
        
        self.weights.data += delta_w
        
        return self.weights


class SurrogateGradient(nn.Module):
    """
    Gradient de substitution pour l'apprentissage par rétropropagation.
    """
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcule le gradient de substitution.
        
        Args:
            x: Entrée
            
        Returns:
            Gradient de substitution
        """
        return torch.sigmoid(self.alpha * x)
