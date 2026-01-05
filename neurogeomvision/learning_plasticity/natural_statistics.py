"""
Module natural_statistics.py - Statistiques naturelles des images
Apprentissage des filtres à partir d'images naturelles
VERSION CORRIGÉE
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
import math
import matplotlib.pyplot as plt


class NaturalStatistics:
    """
    Analyse des statistiques naturelles des images.
    VERSION CORRIGÉE avec calcul FFT fixé.
    """
    
    def __init__(self,
                 patch_size: int = 16,
                 device: str = 'cpu'):
        
        self.patch_size = patch_size
        self.device = device
        
    def extract_patches(self,
                       image: torch.Tensor,
                       n_patches: int = 1000) -> torch.Tensor:
        """
        Extrait des patches aléatoires d'une image.
        """
        h, w = image.shape
        
        patches = []
        for _ in range(n_patches):
            y = torch.randint(0, h - self.patch_size, (1,)).item()
            x = torch.randint(0, w - self.patch_size, (1,)).item()
            
            patch = image[y:y+self.patch_size, x:x+self.patch_size]
            patches.append(patch.flatten())
        
        return torch.stack(patches)  # (n_patches, patch_size*patch_size)
    
    def compute_patch_statistics(self,
                                patches: torch.Tensor) -> Dict:
        """
        Calcule les statistiques des patches.
        """
        # Moyenne et covariance
        mean = patches.mean(dim=0)
        patches_centered = patches - mean
        covariance = torch.mm(patches_centered.t(), patches_centered) / (patches.shape[0] - 1)
        
        # Spectre de puissance - CORRECTION: utilise torch.linalg.eig
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            # eigenvalues sont déjà réels pour une matrice symétrique
        except:
            # Fallback pour compatibilité
            eigenvalues, eigenvectors = torch.symeig(covariance, eigenvectors=True)
        
        # Sort eigenvalues in descending order
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Kurtosis (non-gaussianité)
        patches_whitened = torch.mm(patches_centered, eigenvectors)
        kurtosis = torch.mean(patches_whitened ** 4, dim=0) - 3
        
        return {
            'mean': mean,
            'covariance': covariance,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'kurtosis': kurtosis,
            'patch_size': self.patch_size
        }
    
    def analyze_natural_image(self,
                             image: torch.Tensor,
                             n_patches: int = 5000) -> Dict:
        """
        Analyse complète d'une image naturelle.
        VERSION CORRIGÉE avec calcul spectral simplifié.
        """
        # Extrait les patches
        patches = self.extract_patches(image, n_patches)
        
        # Calcule les statistiques
        stats = self.compute_patch_statistics(patches)
        
        # Spectre 1/f - VERSION SIMPLIFIÉE et CORRIGÉE
        h, w = image.shape
        
        # FFT 2D
        fft = torch.fft.fft2(image)
        fft_shifted = torch.fft.fftshift(fft)
        power_spectrum = torch.abs(fft_shifted) ** 2
        
        # Profile radial - version simplifiée
        center_y, center_x = h // 2, w // 2
        
        # Crée un masque de distance
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, device=self.device),
            torch.arange(w, device=self.device),
            indexing='ij'
        )
        
        distances = torch.sqrt((x_coords.float() - center_x) ** 2 + 
                              (y_coords.float() - center_y) ** 2)
        
        # Bins pour le profil radial
        max_radius = min(center_y, center_x)
        radial_bins = torch.arange(0, max_radius, 1.0, device=self.device)
        radial_profile = []
        
        for r in range(len(radial_bins) - 1):
            r_min = radial_bins[r]
            r_max = radial_bins[r + 1]
            
            mask = (distances >= r_min) & (distances < r_max)
            if mask.sum() > 0:
                avg_power = power_spectrum[mask].mean().item()
                radial_profile.append(avg_power)
        
        stats['power_spectrum'] = power_spectrum.cpu()
        stats['radial_profile'] = radial_profile
        stats['image_shape'] = image.shape
        
        return stats
    
    def create_1f_noise(self, size: int = 128) -> torch.Tensor:
        """
        Crée une texture avec spectre 1/f.
        VERSION CORRIGÉE.
        """
        # Bruit blanc
        noise = torch.randn(size, size, device=self.device)
        
        # FFT
        fft = torch.fft.fft2(noise)
        fft_shifted = torch.fft.fftshift(fft)
        
        # Crée un filtre 1/f
        y_coords, x_coords = torch.meshgrid(
            torch.arange(size, device=self.device),
            torch.arange(size, device=self.device),
            indexing='ij'
        )
        
        center = size // 2
        distances = torch.sqrt((x_coords.float() - center) ** 2 + 
                              (y_coords.float() - center) ** 2)
        
        # Évite division par zéro
        distances = torch.clamp(distances, min=1.0)
        
        # Filtre 1/f
        filter_1f = 1.0 / distances
        
        # Normalise le filtre
        filter_1f = filter_1f / filter_1f.max()
        
        # Applique le filtre
        fft_filtered = fft_shifted * filter_1f
        
        # IFFT
        fft_unshifted = torch.fft.ifftshift(fft_filtered)
        image_1f = torch.fft.ifft2(fft_unshifted).real
        
        # Normalise
        image_1f = (image_1f - image_1f.mean()) / image_1f.std()
        
        return image_1f
    
    def visualize_statistics(self,
                            stats: Dict,
                            save_path: str = None):
        """Visualise les statistiques."""
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        # 1. Valeurs propres
        if len(stats['eigenvalues']) > 0:
            n_show = min(50, len(stats['eigenvalues']))
            axes[0, 0].plot(stats['eigenvalues'][:n_show].cpu().numpy(), 'o-', markersize=3)
            axes[0, 0].set_title(f"Valeurs propres ({n_show} premières)")
            axes[0, 0].set_xlabel("Composante")
            axes[0, 0].set_ylabel("Valeur propre")
            axes[0, 0].set_yscale('log')
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, "Pas de valeurs propres", 
                           ha='center', va='center')
            axes[0, 0].axis('off')
        
        # 2. Kurtosis
        if len(stats['kurtosis']) > 0:
            n_show_k = min(100, len(stats['kurtosis']))
            axes[0, 1].hist(stats['kurtosis'][:n_show_k].cpu().numpy(), bins=30, alpha=0.7)
            axes[0, 1].set_title(f"Distribution de kurtosis")
            axes[0, 1].set_xlabel("Kurtosis")
            axes[0, 1].set_ylabel("Fréquence")
            axes[0, 1].axvline(0, color='r', linestyle='--', alpha=0.5, label='Gaussien')
            axes[0, 1].legend()
        else:
            axes[0, 1].text(0.5, 0.5, "Pas de données kurtosis", 
                           ha='center', va='center')
            axes[0, 1].axis('off')
        
        # 3. Profile spectral
        if 'radial_profile' in stats and stats['radial_profile']:
            axes[0, 2].plot(stats['radial_profile'], 'o-', markersize=3)
            axes[0, 2].set_title("Spectre de puissance radial")
            axes[0, 2].set_xlabel("Fréquence spatiale")
            axes[0, 2].set_ylabel("Puissance")
            axes[0, 2].set_yscale('log')
            axes[0, 2].grid(True, alpha=0.3)
        else:
            axes[0, 2].text(0.5, 0.5, "Pas de profil spectral", 
                           ha='center', va='center')
            axes[0, 2].axis('off')
        
        # 4-6. Premières composantes
        patch_dim = int(math.sqrt(stats.get('patch_size', 16)))
        
        for i in range(3):
            if i < stats['eigenvectors'].shape[1]:
                eigenvector = stats['eigenvectors'][:, i].cpu().numpy()
                if len(eigenvector) == patch_dim * patch_dim:
                    eigenvector = eigenvector.reshape(patch_dim, patch_dim)
                    
                    im = axes[1, i].imshow(eigenvector, cmap='RdBu_r')
                    axes[1, i].set_title(f"Composante {i+1}")
                    axes[1, i].axis('off')
                    plt.colorbar(im, ax=axes[1, i], fraction=0.046)
                else:
                    axes[1, i].text(0.5, 0.5, f"Shape mismatch\n{eigenvector.shape}", 
                                   ha='center', va='center')
                    axes[1, i].axis('off')
            else:
                axes[1, i].text(0.5, 0.5, f"Composante {i+1}\nnon disponible", 
                               ha='center', va='center')
                axes[1, i].axis('off')
        
        plt.suptitle("Statistiques des images naturelles", fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
        
        return fig


class ICA_Learning:
    """
    Apprentissage par ICA (Independent Component Analysis).
    VERSION CORRIGÉE.
    """
    
    def __init__(self,
                 input_dim: int,
                 n_components: int,
                 learning_rate: float = 0.01,
                 device: str = 'cpu'):
        
        self.input_dim = input_dim
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.device = device
        
        # Matrice de séparation
        self.W = torch.randn(n_components, input_dim, device=device) * 0.1
        
    def ica_update_simple(self,
                         inputs: torch.Tensor) -> torch.Tensor:
        """
        Mise à jour ICA simplifiée.
        """
        batch_size = inputs.shape[0]
        
        # Sources estimées
        sources = torch.mm(inputs, self.W.t())  # (batch_size, n_components)
        
        # Fonction non-linéaire (tanh) pour la super-gaussianité
        g = torch.tanh(sources)
        g_prime = 1 - g ** 2
        
        # Règle d'apprentissage simplifiée
        delta_W = self.learning_rate * torch.mm(g.t(), inputs) / batch_size
        
        # Met à jour
        self.W += delta_W
        
        # Orthogonalisation approximative
        if self.n_components > 1:
            self.W = self._orthogonalize(self.W)
        
        return sources
    
    def _orthogonalize(self, W: torch.Tensor) -> torch.Tensor:
        """Orthogonalisation simple."""
        # QR decomposition pour orthogonalisation
        try:
            Q, R = torch.linalg.qr(W.t())
            return Q.t()[:self.n_components, :]
        except:
            # Fallback: normalization seulement
            norms = torch.norm(W, dim=1, keepdim=True)
            return W / torch.clamp(norms, min=1e-8)
    
    def learn_gabor_filters_simple(self,
                                  natural_patches: torch.Tensor,
                                  n_epochs: int = 100) -> torch.Tensor:
        """
        Apprend des filtres de type Gabor à partir de patches naturels.
        VERSION SIMPLIFIÉE et ROBUSTE.
        """
        n_patches, patch_dim = natural_patches.shape
        
        # Blanchiment simple (ZCA-like)
        mean = natural_patches.mean(dim=0)
        patches_centered = natural_patches - mean
        
        # Calcul de la covariance
        cov = torch.mm(patches_centered.t(), patches_centered) / (n_patches - 1)
        
        # Décomposition en valeurs propres
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            idx = torch.argsort(eigenvalues, descending=True)
            eigenvectors = eigenvectors[:, idx]
            eigenvalues = eigenvalues[idx]
        except:
            # Fallback: pas de whitening
            eigenvectors = torch.eye(patch_dim, device=self.device)
            eigenvalues = torch.ones(patch_dim, device=self.device)
        
        # Whitening
        epsilon = 1e-8
        whitening_matrix = torch.mm(eigenvectors, 
                                   torch.diag(1.0 / torch.sqrt(eigenvalues + epsilon)))
        
        patches_whitened = torch.mm(patches_centered, whitening_matrix)
        
        # Apprentissage ICA simplifié
        print(f"ICA Learning: {n_patches} patches, {patch_dim} dimensions")
        
        for epoch in range(n_epochs):
            # Mélange
            indices = torch.randperm(n_patches)
            
            # Mini-batch
            batch_size = min(32, n_patches)
            total_loss = 0
            
            for i in range(0, n_patches, batch_size):
                batch_idx = indices[i:i+batch_size]
                if len(batch_idx) == 0:
                    continue
                    
                batch = patches_whitened[batch_idx]
                
                # Mise à jour
                sources = self.ica_update_simple(batch)
                
                # Perte (nég-entropie)
                loss = -torch.mean(torch.log(torch.cosh(sources)))
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / (n_patches // batch_size + 1)
                print(f"  Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")
        
        # Reconstruction des filtres
        filters = torch.mm(self.W, whitening_matrix.t())
        
        # Normalisation
        norms = torch.norm(filters, dim=1, keepdim=True)
        filters = filters / torch.clamp(norms, min=1e-8)
        
        print(f"✓ ICA terminé: {filters.shape[0]} filtres appris")
        
        return filters


class SparseCoding:
    """
    Codage parcimonieux (sparse coding).
    VERSION SIMPLIFIÉE.
    """
    
    def __init__(self,
                 input_dim: int,
                 n_basis: int,
                 sparsity_weight: float = 0.1,
                 learning_rate: float = 0.01,
                 device: str = 'cpu'):
        
        self.input_dim = input_dim
        self.n_basis = n_basis
        self.sparsity_weight = sparsity_weight
        self.learning_rate = learning_rate
        self.device = device
        
        # Dictionnaire de bases
        self.basis = torch.randn(n_basis, input_dim, device=device) * 0.1
        self.basis = self.basis / torch.norm(self.basis, dim=1, keepdim=True)
        
    def sparse_encode_simple(self,
                           inputs: torch.Tensor,
                           n_iterations: int = 20) -> torch.Tensor:
        """
        Encode parcimonieusement les entrées - version simplifiée.
        """
        batch_size = inputs.shape[0]
        coefficients = torch.zeros(batch_size, self.n_basis, device=self.device)
        
        for it in range(n_iterations):
            # Reconstruction
            reconstruction = torch.mm(coefficients, self.basis)
            error = inputs - reconstruction
            
            # Gradient par rapport aux coefficients
            grad_coeff = -torch.mm(error, self.basis.t()) / batch_size
            
            # Terme de parcimonie (L1)
            grad_coeff += self.sparsity_weight * torch.sign(coefficients)
            
            # Descente de gradient
            coefficients -= self.learning_rate * grad_coeff
            
            # Seuillage (soft thresholding)
            threshold = 0.01
            coefficients = torch.sign(coefficients) * torch.relu(torch.abs(coefficients) - threshold)
        
        return coefficients
    
    def learn_dictionary_simple(self,
                               data: torch.Tensor,
                               n_epochs: int = 50) -> torch.Tensor:
        """
        Apprend le dictionnaire de bases - version simplifiée.
        """
        n_samples = data.shape[0]
        
        for epoch in range(n_epochs):
            # Mélange
            indices = torch.randperm(n_samples)
            
            total_error = 0
            n_batches = 0
            
            for i in range(0, n_samples, 32):
                batch_idx = indices[i:i+32]
                if len(batch_idx) == 0:
                    continue
                    
                batch = data[batch_idx]
                
                # Étape E: Encode
                coefficients = self.sparse_encode_simple(batch, n_iterations=10)
                
                # Étape M: Met à jour les bases
                reconstruction = torch.mm(coefficients, self.basis)
                error = batch - reconstruction
                
                grad_basis = -torch.mm(coefficients.t(), error) / batch.shape[0]
                self.basis -= self.learning_rate * grad_basis
                
                # Normalise les bases
                norms = torch.norm(self.basis, dim=1, keepdim=True)
                self.basis = self.basis / torch.clamp(norms, min=1e-8)
                
                # Statistiques
                total_error += torch.mean(error ** 2).item()
                n_batches += 1
            
            if (epoch + 1) % 5 == 0:
                avg_error = total_error / max(n_batches, 1)
                sparsity = torch.mean(torch.abs(coefficients)).item() if 'coefficients' in locals() else 0
                
                print(f"  Epoch {epoch + 1}/{n_epochs}, "
                      f"Erreur: {avg_error:.4f}, "
                      f"Sparsité: {sparsity:.4f}")
        
        return self.basis
