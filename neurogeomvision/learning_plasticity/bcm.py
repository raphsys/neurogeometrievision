"""
Module bcm.py - Règle BCM (Bienenstock-Cooper-Munro)
Plasticité dépendante du seuil modulable
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
import math


class BCMLearning:
    """
    Règle BCM : Δw = η * y * (y - θ) * x
    
    θ est un seuil modulable qui dépend de l'activité moyenne.
    """
    
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 learning_rate: float = 0.01,
                 theta_init: float = 1.0,
                 tau_theta: float = 100.0,
                 device: str = 'cpu'):
        
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.tau_theta = tau_theta
        self.device = device
        
        # Poids
        self.weights = torch.randn(output_size, input_size, device=device) * 0.1
        
        # Seuil modulable θ
        self.theta = torch.full((output_size,), theta_init, device=device)
        
        # Historique
        self.theta_history = []
        self.weight_history = []
    
    def bcm_update(self,
                  inputs: torch.Tensor,
                  outputs: torch.Tensor) -> torch.Tensor:
        """
        Mise à jour BCM.
        
        Args:
            inputs: x (batch_size, input_size)
            outputs: y (batch_size, output_size)
            
        Returns:
            Nouveaux poids
        """
        batch_size = inputs.shape[0]
        
        # Calcule le seuil moyen pour chaque neurone
        y_mean = outputs.mean(dim=0)  # (output_size,)
        y_mean_sq = (outputs ** 2).mean(dim=0)  # (output_size,)
        
        # Met à jour le seuil θ (moyenne glissante)
        self.theta = (1 - 1/self.tau_theta) * self.theta + (1/self.tau_theta) * y_mean_sq
        
        # Calcule les changements de poids
        for k in range(self.output_size):
            # Δw_k = η * y_k * (y_k - θ_k) * x
            y_k = outputs[:, k:k+1]  # (batch_size, 1)
            theta_k = self.theta[k]
            
            factor = y_k * (y_k - theta_k)  # (batch_size, 1)
            delta_w_k = self.learning_rate * torch.mm(factor.t(), inputs) / batch_size  # (1, input_size)
            
            self.weights[k:k+1, :] += delta_w_k
        
        # Normalisation
        self._normalize_weights()
        
        # Sauvegarde
        self.theta_history.append(self.theta.clone())
        self.weight_history.append(self.weights.clone())
        
        return self.weights
    
    def _normalize_weights(self):
        """Normalise les poids."""
        norms = torch.norm(self.weights, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)
        self.weights = self.weights / norms
    
    def compute_stability_measure(self, data: torch.Tensor) -> Dict:
        """
        Calcule des mesures de stabilité.
        """
        outputs = torch.mm(data, self.weights.t())
        
        # Homéostasie
        y_mean = outputs.mean(dim=0)
        y_var = outputs.var(dim=0)
        
        # Sélectivité
        selectivity = y_var / (y_mean + 1e-8)
        
        return {
            'mean_activity': y_mean,
            'variance': y_var,
            'selectivity': selectivity,
            'theta': self.theta
        }
    
    def learn_selectivity(self,
                         data: torch.Tensor,
                         n_epochs: int = 200) -> Dict:
        """
        Apprentissage pour développer la sélectivité.
        """
        stats = {
            'theta_history': [],
            'weight_norms': [],
            'selectivity': []
        }
        
        for epoch in range(n_epochs):
            # Forward pass
            outputs = torch.mm(data, self.weights.t())
            
            # Mise à jour BCM
            self.bcm_update(data, outputs)
            
            # Statistiques
            stats['theta_history'].append(self.theta.mean().item())
            stats['weight_norms'].append(torch.norm(self.weights).item())
            
            # Sélectivité
            current_stats = self.compute_stability_measure(data)
            stats['selectivity'].append(current_stats['selectivity'].mean().item())
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, "
                      f"θ moyen: {stats['theta_history'][-1]:.4f}, "
                      f"Sélectivité: {stats['selectivity'][-1]:.4f}")
        
        return stats


class DynamicBCM(BCMLearning):
    """
    BCM dynamique avec adaptation du seuil.
    """
    
    def __init__(self, *args, theta_adaptation_rate: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.theta_adaptation_rate = theta_adaptation_rate
    
    def dynamic_theta_update(self,
                           outputs: torch.Tensor,
                           target_activity: float = 0.1):
        """
        Adaptation dynamique du seuil.
        """
        y_mean = outputs.mean(dim=0)
        
        # Ajuste θ pour maintenir l'activité cible
        delta_theta = self.theta_adaptation_rate * (y_mean - target_activity)
        self.theta += delta_theta
        
        # Contraint
        self.theta = torch.clamp(self.theta, 0.0, 10.0)
    
    def homeostasis_update(self,
                          inputs: torch.Tensor,
                          outputs: torch.Tensor,
                          target_activity: float = 0.1):
        """
        Mise à jour avec homéostasie.
        """
        # Mise à jour BCM standard
        self.bcm_update(inputs, outputs)
        
        # Adaptation homéostatique
        self.dynamic_theta_update(outputs, target_activity)
        
        # Ajustement additionnel des poids pour homéostasie
        y_mean = outputs.mean(dim=0)
        weight_scaling = target_activity / (y_mean + 1e-8)
        weight_scaling = torch.clamp(weight_scaling, 0.5, 2.0)
        
        for k in range(self.output_size):
            self.weights[k, :] *= weight_scaling[k]
