"""
Module turing_patterns.py - Patterns de Turing pour la morphogénèse corticale
"""

import torch
import numpy as np
from typing import Tuple, List, Optional
import math
from scipy import ndimage


class TuringPatterns:
    """
    Patterns de Turing par réaction-diffusion.
    
    Modèle à deux composants (activateur/inhibiteur):
    ∂u/∂t = f(u,v) + D_u ∇²u
    ∂v/∂t = g(u,v) + D_v ∇²v
    """
    
    def __init__(self,
                 spatial_shape: Tuple[int, int],
                 device: str = 'cpu'):
        """
        Args:
            spatial_shape: (height, width)
            device: 'cpu' ou 'cuda'
        """
        self.spatial_shape = spatial_shape
        self.height, self.width = spatial_shape
        self.device = device
        
        # Paramètres du modèle (FitzHugh-Nagumo modifié)
        self.a = 0.1    # Taux d'activation
        self.b = 0.9    # Taux d'inhibition  
        self.c = 0.075  # Seuil d'activation
        
        # Coefficients de diffusion
        self.D_u = 0.2  # Diffusion de l'activateur
        self.D_v = 1.0  # Diffusion de l'inhibiteur (plus rapide)
        
        # Discrétisation
        self.dx = 1.0   # Pas spatial
        self.dt = 0.5   # Pas temporel
        
        # État
        self.u = None  # Activateur
        self.v = None  # Inhibiteur
        
        # Initialise
        self.initialize_state()
    
    def initialize_state(self,
                        noise_level: float = 0.1,
                        pattern: str = 'random'):
        """Initialise les concentrations u et v."""
        if pattern == 'random':
            self.u = torch.rand(self.height, self.width, device=self.device) * noise_level
            self.v = torch.rand(self.height, self.width, device=self.device) * noise_level
            
        elif pattern == 'spot':
            # Tache centrale
            self.u = torch.zeros(self.height, self.width, device=self.device)
            self.v = torch.zeros(self.height, self.width, device=self.device)
            
            center_y, center_x = self.height // 2, self.width // 2
            for y in range(self.height):
                for x in range(self.width):
                    dist = math.sqrt((y - center_y)**2 + (x - center_x)**2)
                    if dist < 5:
                        self.u[y, x] = 0.8
                        self.v[y, x] = 0.4
    
    def reaction_terms(self, u: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Termes de réaction f(u,v) et g(u,v)."""
        # Modèle FitzHugh-Nagumo modifié
        f = u - u**3 - v + self.a
        g = (u - self.b * v + self.c) / self.b
        
        return f, g
    
    def laplacian(self, field: torch.Tensor) -> torch.Tensor:
        """Calcule le Laplacien discret."""
        # Noyau Laplacien (5-point stencil)
        kernel = torch.tensor([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]], device=self.device).float()
        
        field_2d = field.unsqueeze(0).unsqueeze(0)
        kernel_2d = kernel.unsqueeze(0).unsqueeze(0)
        
        # Convolution avec padding
        laplacian = torch.nn.functional.conv2d(
            torch.nn.functional.pad(field_2d, (1, 1, 1, 1), mode='reflect'),
            kernel_2d
        ).squeeze()
        
        return laplacian / (self.dx ** 2)
    
    def step(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Un pas d'intégration."""
        # Termes de réaction
        f, g = self.reaction_terms(self.u, self.v)
        
        # Laplaciens
        laplacian_u = self.laplacian(self.u)
        laplacian_v = self.laplacian(self.v)
        
        # Équations de réaction-diffusion
        du_dt = f + self.D_u * laplacian_u
        dv_dt = g + self.D_v * laplacian_v
        
        # Intégration d'Euler
        u_new = self.u + du_dt * self.dt
        v_new = self.v + dv_dt * self.dt
        
        # Contraintes
        u_new = torch.clamp(u_new, -1, 1)
        v_new = torch.clamp(v_new, -1, 1)
        
        self.u = u_new
        self.v = v_new
        
        return u_new, v_new
    
    def simulate(self,
                n_steps: int = 500,
                save_every: int = 50) -> List[torch.Tensor]:
        """Simulation complète."""
        history = [self.u.clone()]
        
        for step in range(n_steps):
            u_new, v_new = self.step()
            
            if (step + 1) % save_every == 0:
                history.append(u_new.clone())
        
        return history
    
    def generate_pattern(self,
                        pattern_type: str = 'spots') -> torch.Tensor:
        """
        Génère un pattern de Turing spécifique.
        
        Args:
            pattern_type: 'spots', 'stripes', 'labyrinth', 'hexagons'
            
        Returns:
            Pattern final
        """
        # Ajuste les paramètres pour différents patterns
        if pattern_type == 'spots':
            # Points/taches
            self.D_u = 0.2
            self.D_v = 1.0
            self.a = 0.1
            
        elif pattern_type == 'stripes':
            # Rayures
            self.D_u = 0.1
            self.D_v = 0.5
            self.a = 0.05
            
        elif pattern_type == 'labyrinth':
            # Labyrinthe
            self.D_u = 0.16
            self.D_v = 0.8
            self.a = 0.0
            
        elif pattern_type == 'hexagons':
            # Hexagones
            self.D_u = 0.12
            self.D_v = 0.6
            self.a = 0.03
        
        # Réinitialise avec du bruit
        self.initialize_state(noise_level=0.2, pattern='random')
        
        # Simulation
        history = self.simulate(n_steps=300, save_every=100)
        
        return history[-1]
    
    def compute_dispersion_relation(self,
                                   k_min: float = 0.1,
                                   k_max: float = 2.0,
                                   n_points: int = 100) -> dict:
        """
        Calcule la relation de dispersion pour l'instabilité de Turing.
        
        Returns:
            Taux de croissance λ(k) pour différentes longueurs d'onde
        """
        k_values = np.linspace(k_min, k_max, n_points)
        growth_rates = []
        
        # Matrice jacobienne au point fixe (u0, v0)
        u0, v0 = 0.0, 0.0
        
        # Dérivées partielles
        f_u = 1 - 3 * u0**2
        f_v = -1
        g_u = 1 / self.b
        g_v = -1
        
        for k in k_values:
            k2 = k**2
            
            # Matrice M(k) = J - Dk²
            M = np.array([[f_u - self.D_u * k2, f_v],
                         [g_u, g_v - self.D_v * k2]])
            
            # Valeur propre avec plus grande partie réelle
            eigenvalues = np.linalg.eigvals(M)
            max_real = max(eig.real for eig in eigenvalues)
            
            growth_rates.append(max_real)
        
        return {
            'k': k_values,
            'lambda': growth_rates,
            'critical_k': k_values[np.argmax(growth_rates)] if growth_rates else 0
        }
    
    def visualize_pattern_evolution(self) -> dict:
        """Visualise l'évolution des patterns."""
        history = self.simulate(n_steps=400, save_every=80)
        
        n_frames = len(history)
        cols = min(4, n_frames)
        rows = (n_frames + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        for idx, (ax, pattern) in enumerate(zip(axes, history)):
            im = ax.imshow(pattern.cpu().numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title(f"Step {idx*80}")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        for idx in range(len(history), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        return {
            'history': history,
            'figure': fig,
            'final_pattern': history[-1]
        }
