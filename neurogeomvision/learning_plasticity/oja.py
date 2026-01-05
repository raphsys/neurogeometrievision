"""
Module oja.py - Règles d'Oja et Sanger
Apprentissage de composantes principales
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
import math


class OjaLearning:
    """
    Règle d'Oja : Δw = η * y * (x - y * w)
    
    Version normalisée de Hebb qui converge vers la 1ère composante principale.
    """
    
    def __init__(self,
                 input_size: int,
                 output_size: int = 1,
                 learning_rate: float = 0.01,
                 device: str = 'cpu'):
        
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.device = device
        
        # Poids
        self.weights = torch.randn(output_size, input_size, device=device) * 0.1
        
        # Normalisation initiale
        self._normalize_weights()
    
    def _normalize_weights(self):
        """Normalise les poids."""
        norms = torch.norm(self.weights, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)
        self.weights = self.weights / norms
    
    def oja_update(self,
                  inputs: torch.Tensor) -> torch.Tensor:
        """
        Mise à jour d'Oja.
        
        Args:
            inputs: x (batch_size, input_size)
            
        Returns:
            Nouveaux poids
        """
        batch_size = inputs.shape[0]
        
        # Calcule les sorties
        outputs = torch.mm(inputs, self.weights.t())  # (batch_size, output_size)
        
        # Règle d'Oja : Δw = η * y * (x - y * w)
        for k in range(self.output_size):
            y_k = outputs[:, k:k+1]  # (batch_size, 1)
            w_k = self.weights[k:k+1, :]  # (1, input_size)
            
            # (x - y * w) pour chaque échantillon
            reconstruction = y_k * w_k  # (batch_size, input_size)
            error = inputs - reconstruction  # (batch_size, input_size)
            
            # Δw_k = η * moyenne(y_k * error)
            delta_w_k = self.learning_rate * torch.mm(y_k.t(), error) / batch_size  # (1, input_size)
            
            self.weights[k:k+1, :] += delta_w_k
        
        # Normalisation
        self._normalize_weights()
        
        return self.weights
    
    def extract_first_pc(self,
                        data: torch.Tensor,
                        n_epochs: int = 100) -> torch.Tensor:
        """
        Extrait la première composante principale.
        """
        for epoch in range(n_epochs):
            # Mélange les données
            indices = torch.randperm(data.shape[0])
            
            for i in indices:
                sample = data[i:i+1]  # (1, input_size)
                self.oja_update(sample)
            
            # Affiche la progression
            if (epoch + 1) % 10 == 0:
                variance = self.explained_variance(data)
                print(f"Epoch {epoch + 1}/{n_epochs}, "
                      f"Variance expliquée: {variance:.4f}")
        
        return self.weights
    
    def explained_variance(self, data: torch.Tensor) -> float:
        """
        Calcule la variance expliquée.
        """
        outputs = torch.mm(data, self.weights.t())
        reconstructed = torch.mm(outputs, self.weights)
        
        ss_total = torch.sum((data - data.mean(dim=0)) ** 2)
        ss_residual = torch.sum((data - reconstructed) ** 2)
        
        return 1.0 - ss_residual / ss_total


class SangerLearning(OjaLearning):
    """
    Règle de Sanger (Generalized Hebbian Algorithm).
    Extrait plusieurs composantes principales.
    """
    
    def __init__(self,
                 input_size: int,
                 n_components: int,
                 learning_rate: float = 0.01,
                 device: str = 'cpu'):
        
        super().__init__(input_size, n_components, learning_rate, device)
        
    def sanger_update(self,
                     inputs: torch.Tensor) -> torch.Tensor:
        """
        Mise à jour de Sanger.
        """
        batch_size = inputs.shape[0]
        
        # Calcule toutes les sorties
        outputs = torch.mm(inputs, self.weights.t())  # (batch_size, n_components)
        
        # Met à jour chaque composante
        for k in range(self.output_size):
            y_k = outputs[:, k:k+1]  # (batch_size, 1)
            w_k = self.weights[k:k+1, :]  # (1, input_size)
            
            # Reconstruction avec les k premières composantes
            if k > 0:
                W_prev = self.weights[:k, :]  # (k, input_size)
                Y_prev = outputs[:, :k]  # (batch_size, k)
                reconstruction = torch.mm(Y_prev, W_prev)  # (batch_size, input_size)
                residual = inputs - reconstruction
            else:
                residual = inputs
            
            # Règle de Sanger
            delta_w_k = self.learning_rate * torch.mm(y_k.t(), residual) / batch_size
            self.weights[k:k+1, :] += delta_w_k
        
        # Orthonormalisation de Gram-Schmidt
        self._gram_schmidt()
        
        return self.weights
    
    def _gram_schmidt(self):
        """Orthonormalisation de Gram-Schmidt."""
        for k in range(self.output_size):
            # Sous-espace des composantes précédentes
            if k > 0:
                W_prev = self.weights[:k, :]  # (k, input_size)
                
                # Projection sur le complément orthogonal
                for j in range(k):
                    proj = torch.dot(self.weights[k, :], W_prev[j, :])
                    self.weights[k, :] -= proj * W_prev[j, :]
            
            # Normalisation
            norm = torch.norm(self.weights[k, :])
            if norm > 1e-8:
                self.weights[k, :] = self.weights[k, :] / norm
    
    def extract_pcs(self,
                   data: torch.Tensor,
                   n_epochs: int = 200) -> torch.Tensor:
        """
        Extrait plusieurs composantes principales.
        """
        variance_history = []
        
        for epoch in range(n_epochs):
            # Mélange
            indices = torch.randperm(data.shape[0])
            batch = data[indices]
            
            # Mise à jour
            self.sanger_update(batch)
            
            # Calcule la variance expliquée
            outputs = torch.mm(data, self.weights.t())
            reconstructed = torch.mm(outputs, self.weights)
            
            ss_total = torch.sum((data - data.mean(dim=0)) ** 2)
            ss_residual = torch.sum((data - reconstructed) ** 2)
            var_explained = 1.0 - ss_residual / ss_total
            
            variance_history.append(var_explained.item())
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, "
                      f"Variance expliquée: {var_explained:.4f}")
        
        return self.weights
