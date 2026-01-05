"""
Module stdp.py - Spike-Timing Dependent Plasticity
Plasticité dépendante du timing des spikes
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
import math


class STDPLearning:
    """
    STDP (Spike-Timing Dependent Plasticity)
    
    Δw = A_+ * exp(-Δt/τ_+) si Δt > 0 (pre → post)
    Δw = -A_- * exp(Δt/τ_-) si Δt < 0 (post → pre)
    """
    
    def __init__(self,
                 n_neurons: int,
                 A_plus: float = 0.01,
                 A_minus: float = 0.0105,
                 tau_plus: float = 20.0,  # ms
                 tau_minus: float = 20.0,  # ms
                 w_max: float = 1.0,
                 w_min: float = 0.0,
                 device: str = 'cpu'):
        
        self.n_neurons = n_neurons
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_max = w_max
        self.w_min = w_min
        self.device = device
        
        # Matrice de poids
        self.weights = torch.rand(n_neurons, n_neurons, device=device) * 0.1
        
        # Derniers temps de spike
        self.last_pre_spike = torch.full((n_neurons,), -1e6, device=device)
        self.last_post_spike = torch.full((n_neurons,), -1e6, device=device)
        
        # Traces d'activité
        self.x_trace = torch.zeros(n_neurons, device=device)  # Pré-synaptique
        self.y_trace = torch.zeros(n_neurons, device=device)  # Post-synaptique
        
    def stdp_update(self,
                   pre_spikes: torch.Tensor,
                   post_spikes: torch.Tensor,
                   current_time: float) -> torch.Tensor:
        """
        Met à jour les poids selon la règle STDP.
        
        Args:
            pre_spikes: Spikes pré-synaptiques (n_neurons,)
            post_spikes: Spikes post-synaptiques (n_neurons,)
            current_time: Temps courant (ms)
            
        Returns:
            Nouveaux poids
        """
        # Met à jour les traces
        dt = 1.0  # Pas de temps
        self.x_trace = self.x_trace * math.exp(-dt / self.tau_plus) + pre_spikes
        self.y_trace = self.y_trace * math.exp(-dt / self.tau_minus) + post_spikes
        
        # Calcule les changements de poids
        delta_w = torch.zeros_like(self.weights)
        
        # Pour chaque paire de neurones
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if pre_spikes[i] > 0 and post_spikes[j] > 0:
                    # Δt = t_post - t_pre
                    delta_t = current_time - self.last_pre_spike[i]
                    
                    if delta_t > 0:  # Pré avant post
                        delta_w[i, j] = self.A_plus * math.exp(-delta_t / self.tau_plus)
                    else:  # Post avant pré
                        delta_w[i, j] = -self.A_minus * math.exp(delta_t / self.tau_minus)
        
        # Met à jour les derniers temps de spike
        self.last_pre_spike[pre_spikes > 0] = current_time
        self.last_post_spike[post_spikes > 0] = current_time
        
        # Applique les changements
        self.weights += delta_w
        
        # Contraint les poids
        self.weights = torch.clamp(self.weights, self.w_min, self.w_max)
        
        return self.weights
    
    def pair_based_stdp(self,
                       pre_spike_times: List[float],
                       post_spike_times: List[float],
                       current_time: float) -> float:
        """
        STDP basé sur les paires de spikes.
        
        Returns:
            Changement total de poids
        """
        delta_w_total = 0.0
        
        for t_pre in pre_spike_times:
            for t_post in post_spike_times:
                delta_t = t_post - t_pre
                
                if delta_t > 0:  # Pré avant post
                    delta_w = self.A_plus * math.exp(-delta_t / self.tau_plus)
                else:  # Post avant pré
                    delta_w = -self.A_minus * math.exp(delta_t / self.tau_minus)
                
                delta_w_total += delta_w
        
        return delta_w_total
    
    def trace_based_stdp(self,
                        pre_trace: float,
                        post_trace: float,
                        pre_spike: float,
                        post_spike: float) -> float:
        """
        STDP basé sur les traces.
        """
        # Mise à jour des traces
        pre_trace_new = pre_trace * math.exp(-1/self.tau_plus) + pre_spike
        post_trace_new = post_trace * math.exp(-1/self.tau_minus) + post_spike
        
        # Changement de poids
        delta_w = self.A_plus * post_trace_new * pre_spike - \
                  self.A_minus * pre_trace_new * post_spike
        
        return delta_w, pre_trace_new, post_trace_new
    
    def simulate_spike_train(self,
                            n_steps: int = 1000,
                            firing_rate: float = 0.1) -> Dict:
        """
        Simule un train de spikes et l'apprentissage STDP.
        """
        spike_history = []
        weight_history = []
        
        for step in range(n_steps):
            # Génère des spikes aléatoires
            pre_spikes = (torch.rand(self.n_neurons, device=self.device) < firing_rate).float()
            post_spikes = (torch.rand(self.n_neurons, device=self.device) < firing_rate).float()
            
            # Met à jour les poids
            self.stdp_update(pre_spikes, post_spikes, float(step))
            
            # Sauvegarde
            spike_history.append((pre_spikes.sum().item(), post_spikes.sum().item()))
            weight_history.append(self.weights.mean().item())
            
            # Affiche la progression
            if (step + 1) % 100 == 0:
                print(f"Step {step + 1}/{n_steps}, "
                      f"Poids moyen: {weight_history[-1]:.4f}")
        
        return {
            'spike_history': spike_history,
            'weight_history': weight_history,
            'final_weights': self.weights
        }


class ExponentialSTDP(STDPLearning):
    """
    STDP exponentielle avec fenêtre temporelle.
    """
    
    def __init__(self, n_neurons: int, **kwargs):
        super().__init__(n_neurons, **kwargs)
        
    def exponential_stdp_kernel(self, delta_t: float) -> float:
        """Noyau exponentiel STDP."""
        if delta_t > 0:
            return self.A_plus * math.exp(-delta_t / self.tau_plus)
        else:
            return -self.A_minus * math.exp(delta_t / self.tau_minus)
    
    def apply_stdp_window(self,
                         pre_times: torch.Tensor,
                         post_times: torch.Tensor,
                         window_size: float = 100.0) -> torch.Tensor:
        """
        Applique STDP avec une fenêtre temporelle.
        """
        delta_w = torch.zeros_like(self.weights)
        
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                # Trouve les paires dans la fenêtre
                for t_pre in pre_times[i]:
                    for t_post in post_times[j]:
                        delta_t = t_post - t_pre
                        
                        if abs(delta_t) < window_size:
                            delta_w[i, j] += self.exponential_stdp_kernel(delta_t)
        
        return delta_w


class TripletSTDP(STDPLearning):
    """
    STDP à triplets (Pfister & Gerstner, 2006).
    Prend en compte les interactions à trois spikes.
    """
    
    def __init__(self,
                 n_neurons: int,
                 A_plus_2: float = 0.001,
                 A_minus_2: float = 0.001,
                 tau_x: float = 100.0,
                 tau_y: float = 100.0,
                 **kwargs):
        
        super().__init__(n_neurons, **kwargs)
        
        # Paramètres additionnels pour les triplets
        self.A_plus_2 = A_plus_2
        self.A_minus_2 = A_minus_2
        self.tau_x = tau_x
        self.tau_y = tau_y
        
        # Traces additionnelles
        self.x_trace_2 = torch.zeros(n_neurons, device=self.device)
        self.y_trace_2 = torch.zeros(n_neurons, device=self.device)
    
    def triplet_update(self,
                      pre_spikes: torch.Tensor,
                      post_spikes: torch.Tensor,
                      current_time: float) -> torch.Tensor:
        """
        Mise à jour par triplets.
        """
        dt = 1.0
        
        # Met à jour les traces simples
        self.x_trace = self.x_trace * math.exp(-dt / self.tau_plus) + pre_spikes
        self.y_trace = self.y_trace * math.exp(-dt / self.tau_minus) + post_spikes
        
        # Met à jour les traces de triplets
        self.x_trace_2 = self.x_trace_2 * math.exp(-dt / self.tau_x) + pre_spikes
        self.y_trace_2 = self.y_trace_2 * math.exp(-dt / self.tau_y) + post_spikes
        
        # Calcul des changements
        delta_w = torch.zeros_like(self.weights)
        
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                # Terme paire
                if pre_spikes[i] > 0:
                    delta_w[i, j] += self.A_plus * self.y_trace[j]
                
                if post_spikes[j] > 0:
                    delta_w[i, j] -= self.A_minus * self.x_trace[i]
                
                # Terme triplet
                if pre_spikes[i] > 0:
                    delta_w[i, j] += self.A_plus_2 * self.y_trace_2[j] * self.x_trace[i]
                
                if post_spikes[j] > 0:
                    delta_w[i, j] -= self.A_minus_2 * self.x_trace_2[i] * self.y_trace[j]
        
        # Applique les changements
        self.weights += delta_w
        
        # Contraint les poids
        self.weights = torch.clamp(self.weights, self.w_min, self.w_max)
        
        return self.weights
