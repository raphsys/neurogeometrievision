"""
Module hebbian.py - Règles d'apprentissage hebbiennes
Implémente les règles de Hebb classiques et variations
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional, Callable
import math
import matplotlib.pyplot as plt


class HebbianLearning:
    """
    Apprentissage hebbien classique : Δw_ij = η * x_i * y_j
    
    "Neurons that fire together, wire together"
    """
    
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 learning_rate: float = 0.01,
                 device: str = 'cpu'):
        """
        Args:
            input_size: Dimension de l'entrée
            output_size: Dimension de la sortie
            learning_rate: Taux d'apprentissage η
            device: 'cpu' ou 'cuda'
        """
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.device = device
        
        # Initialise les poids aléatoirement
        self.weights = torch.randn(output_size, input_size, device=device) * 0.1
        
        # Historique
        self.weight_history = []
        self.activity_history = []
        
    def hebbian_update(self,
                       inputs: torch.Tensor,
                       outputs: torch.Tensor) -> torch.Tensor:
        """
        Mise à jour hebbienne classique : ΔW = η * y * x^T
        
        Args:
            inputs: Vecteur d'entrée x (batch_size, input_size)
            outputs: Vecteur de sortie y (batch_size, output_size)
            
        Returns:
            Nouveaux poids
        """
        batch_size = inputs.shape[0]
        
        # Règle de Hebb : ΔW = η * y^T * x
        delta_w = self.learning_rate * torch.mm(outputs.t(), inputs) / batch_size
        
        # Mise à jour
        self.weights += delta_w
        
        # Normalisation (pour éviter l'explosion)
        self.weights = self._normalize_weights(self.weights)
        
        # Sauvegarde
        self.weight_history.append(self.weights.clone())
        
        return self.weights
    
    def _normalize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Normalise les poids par colonne."""
        norms = torch.norm(weights, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)
        return weights / norms
    
    def compute_outputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Calcule les sorties : y = W·x."""
        return torch.mm(inputs, self.weights.t())
    
    def learn_from_data(self,
                       data: torch.Tensor,
                       n_epochs: int = 100,
                       batch_size: int = 32) -> Dict:
        """
        Apprend à partir d'un jeu de données.
        
        Args:
            data: Données d'apprentissage (n_samples, input_size)
            n_epochs: Nombre d'époques
            batch_size: Taille des batchs
            
        Returns:
            Statistiques d'apprentissage
        """
        n_samples = data.shape[0]
        stats = {
            'weight_norms': [],
            'weight_changes': [],
            'output_variance': []
        }
        
        for epoch in range(n_epochs):
            # Mélange les données
            indices = torch.randperm(n_samples)
            
            for batch_start in range(0, n_samples, batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_data = data[batch_indices]
                
                # Forward pass
                outputs = self.compute_outputs(batch_data)
                
                # Mise à jour hebbienne
                self.hebbian_update(batch_data, outputs)
            
            # Statistiques
            stats['weight_norms'].append(torch.norm(self.weights).item())
            
            if len(self.weight_history) > 1:
                change = torch.norm(self.weight_history[-1] - self.weight_history[-2]).item()
                stats['weight_changes'].append(change)
            
            # Variance des sorties
            all_outputs = self.compute_outputs(data)
            stats['output_variance'].append(all_outputs.var().item())
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, "
                      f"Norme poids: {stats['weight_norms'][-1]:.4f}")
        
        return stats
    
    def extract_features(self, n_features: int = 10) -> torch.Tensor:
        """
        Extrait les caractéristiques apprises.
        
        Returns:
            Filtres appris (n_features, input_size)
        """
        # Les poids représentent les caractéristiques apprises
        norms = torch.norm(self.weights, dim=1)
        _, indices = torch.topk(norms, n_features)
        return self.weights[indices]
    
    def visualize_learning(self, stats: Dict, save_path: str = None):
        """Visualise le processus d'apprentissage."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. Norme des poids
        axes[0, 0].plot(stats['weight_norms'])
        axes[0, 0].set_title("Norme des poids")
        axes[0, 0].set_xlabel("Époque")
        axes[0, 0].set_ylabel("Norme")
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Changement des poids
        if stats['weight_changes']:
            axes[0, 1].plot(stats['weight_changes'])
            axes[0, 1].set_title("Changement des poids")
            axes[0, 1].set_xlabel("Époque")
            axes[0, 1].set_ylabel("ΔW")
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Variance des sorties
        axes[1, 0].plot(stats['output_variance'])
        axes[1, 0].set_title("Variance des sorties")
        axes[1, 0].set_xlabel("Époque")
        axes[1, 0].set_ylabel("Variance")
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Matrice de poids
        im = axes[1, 1].imshow(self.weights.cpu().numpy(), 
                              cmap='RdBu_r', aspect='auto')
        axes[1, 1].set_title("Matrice de poids appris")
        axes[1, 1].set_xlabel("Entrées")
        axes[1, 1].set_ylabel("Sorties")
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.suptitle("Apprentissage Hebbien", fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
        
        return fig


class CovarianceLearning:
    """
    Règle de covariance : Δw_ij = η * (x_i - μ_i) * (y_j - μ_j)
    
    Extension de Hebb avec soustraction des moyennes.
    """
    
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 learning_rate: float = 0.01,
                 device: str = 'cpu'):
        
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.device = device
        
        # Poids
        self.weights = torch.randn(output_size, input_size, device=device) * 0.1
        
        # Moyennes (estimées en ligne)
        self.input_mean = torch.zeros(input_size, device=device)
        self.output_mean = torch.zeros(output_size, device=device)
        
        # Facteur d'oubli pour les moyennes
        self.forgetting_factor = 0.01
        
    def covariance_update(self,
                         inputs: torch.Tensor,
                         outputs: torch.Tensor) -> torch.Tensor:
        """
        Mise à jour par covariance.
        """
        batch_size = inputs.shape[0]
        
        # Met à jour les moyennes
        self.input_mean = (1 - self.forgetting_factor) * self.input_mean + \
                         self.forgetting_factor * inputs.mean(dim=0)
        self.output_mean = (1 - self.forgetting_factor) * self.output_mean + \
                          self.forgetting_factor * outputs.mean(dim=0)
        
        # Centrage
        inputs_centered = inputs - self.input_mean
        outputs_centered = outputs - self.output_mean
        
        # Règle de covariance
        delta_w = self.learning_rate * torch.mm(outputs_centered.t(), inputs_centered) / batch_size
        
        # Mise à jour
        self.weights += delta_w
        
        # Normalisation
        norms = torch.norm(self.weights, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)
        self.weights = self.weights / norms
        
        return self.weights
    
    def compute_covariance_matrix(self, data: torch.Tensor) -> torch.Tensor:
        """Calcule la matrice de covariance des données."""
        data_centered = data - data.mean(dim=0)
        covariance = torch.mm(data_centered.t(), data_centered) / (data.shape[0] - 1)
        return covariance
    
    def extract_principal_components(self, 
                                    data: torch.Tensor,
                                    n_components: int = 10) -> torch.Tensor:
        """
        Extrait les composantes principales via apprentissage hebbien.
        """
        # Implémentation d'Oja (simplifiée)
        for _ in range(100):
            for sample in data:
                x = sample.unsqueeze(0)  # (1, input_size)
                y = torch.mm(x, self.weights.t())  # (1, output_size)
                self.covariance_update(x, y)
        
        return self.weights[:n_components]
