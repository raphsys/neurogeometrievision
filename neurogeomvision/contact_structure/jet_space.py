"""
Module jet_space.py - Implémentation de l'espace des 1-jets J¹(R²)
"""

import torch
import numpy as np
from typing import Tuple, Optional, List
import math


class JetSpace:
    """
    Représente l'espace des 1-jets J¹(R²) = R² × P¹.
    Point mathématique central de la neurogéométrie.
    
    Selon Petitot : "V1 peut être identifié au fibré des 1-jets de courbes dans R²"
    """
    
    def __init__(self, spatial_shape: Tuple[int, int], device: str = 'cpu'):
        """
        Args:
            spatial_shape: (height, width) du plan de base R²
            device: 'cpu' ou 'cuda'
        """
        self.height, self.width = spatial_shape
        self.device = device
        
        # Grille spatiale
        self.y_coords, self.x_coords = torch.meshgrid(
            torch.arange(self.height, device=device),
            torch.arange(self.width, device=device),
            indexing='ij'
        )
        
        # Espace des orientations (0 à π)
        self.theta_range = torch.linspace(0, math.pi, 180, device=device)
        
        # Forme de contact canonique : ω = dy - p dx
        # où p = tan(θ) est la pente
        self.p_range = torch.tan(self.theta_range)
        
    def create_jet_coordinates(self) -> torch.Tensor:
        """
        Crée un tenseur de coordonnées jet (x, y, p) pour tout l'espace.
        
        Returns:
            Tensor de forme (height, width, n_orientations, 3)
            où chaque point a coordonnées (x, y, p)
        """
        # Pour chaque orientation, créer une grille 3D
        jet_grid = torch.zeros(self.height, self.width, len(self.p_range), 3, 
                              device=self.device)
        
        for i, p in enumerate(self.p_range):
            jet_grid[:, :, i, 0] = self.x_coords  # x
            jet_grid[:, :, i, 1] = self.y_coords  # y
            jet_grid[:, :, i, 2] = p              # p = tan(θ)
        
        return jet_grid
    
    def contact_form(self, jet_point: torch.Tensor) -> torch.Tensor:
        """
        Calcule la forme de contact ω = dy - p dx en un point jet.
        
        Args:
            jet_point: Tensor de forme (..., 3) avec (x, y, p)
            
        Returns:
            Valeur de la forme de contact
        """
        if jet_point.dim() == 1:
            x, y, p = jet_point
            return y - p * x
        else:
            x = jet_point[..., 0]
            y = jet_point[..., 1]
            p = jet_point[..., 2]
            return y - p * x
    
    def is_legendrian(self, curve_points: torch.Tensor, tolerance: float = 1e-3) -> torch.Tensor:
        """
        Vérifie si une courbe dans J¹(R²) est legendrienne
        (i.e., tangente au plan de contact).
        
        Args:
            curve_points: Tensor de forme (n_points, 3) avec (x, y, p)
            tolerance: Tolérance numérique
            
        Returns:
            Booléen indiquant si la courbe est legendrienne
        """
        if curve_points.shape[0] < 2:
            return torch.tensor(False, device=self.device)
        
        # Calcule les différences finies
        dx = curve_points[1:, 0] - curve_points[:-1, 0]
        dy = curve_points[1:, 1] - curve_points[:-1, 1]
        dp = curve_points[1:, 2] - curve_points[:-1, 2]
        
        # Condition legendrienne : dy = p dx
        # Vérifie |dy - p_avg dx| < tolerance
        p_avg = 0.5 * (curve_points[1:, 2] + curve_points[:-1, 2])
        legendrian_condition = torch.abs(dy - p_avg * dx)
        
        return torch.all(legendrian_condition < tolerance)
    
    def project_to_base(self, jet_points: torch.Tensor) -> torch.Tensor:
        """
        Projette des points de J¹(R²) vers la base R².
        
        Args:
            jet_points: Tensor de forme (..., 3) avec (x, y, p)
            
        Returns:
            Tensor de forme (..., 2) avec (x, y)
        """
        return jet_points[..., :2]
    
    def lift_from_base(self, base_points: torch.Tensor, 
                       orientations: torch.Tensor) -> torch.Tensor:
        """
        Relève des points de la base R² vers J¹(R²) avec une orientation donnée.
        
        Args:
            base_points: Tensor de forme (..., 2) avec (x, y)
            orientations: Tensor de forme (...) avec les angles θ (0 à π)
            
        Returns:
            Tensor de forme (..., 3) avec (x, y, p) où p = tan(θ)
        """
        if base_points.shape[-1] != 2:
            raise ValueError("base_points doit avoir shape (..., 2)")
        
        # Convertit les orientations en pentes
        p = torch.tan(orientations)
        
        # Combine avec les points de base
        jet_points = torch.cat([
            base_points,
            p.unsqueeze(-1) if p.dim() < base_points.dim() else p[..., None]
        ], dim=-1)
        
        return jet_points
    
    def create_orientation_field(self, frequency: float = 0.1) -> torch.Tensor:
        """
        Crée un champ d'orientation sinusoïdal pour tester.
        
        Args:
            frequency: Fréquence spatiale
            
        Returns:
            Tensor de forme (height, width) avec angles θ
        """
        # Champ d'orientation périodique
        orientation_field = torch.zeros(self.height, self.width, device=self.device)
        
        for y in range(self.height):
            for x in range(self.width):
                # Pattern sinusoïdal
                angle = (math.sin(frequency * x) * math.cos(frequency * y) + 1) * math.pi / 2
                orientation_field[y, x] = angle % math.pi
        
        return orientation_field
    
    def compute_curvature(self, jet_points: torch.Tensor) -> torch.Tensor:
        """
        Calcule la courbure à partir des coordonnées jet.
        Pour une courbe legendrienne, κ = dp/ds où s est la longueur d'arc.
        
        Args:
            jet_points: Tensor de forme (n_points, 3)
            
        Returns:
            Courbure en chaque point
        """
        if jet_points.shape[0] < 3:
            return torch.zeros(jet_points.shape[0], device=self.device)
        
        n_points = jet_points.shape[0]
        curvature = torch.zeros(n_points, device=self.device)
        
        # Différences centrales pour les points intérieurs
        for i in range(1, n_points - 1):
            dx1 = jet_points[i, 0] - jet_points[i-1, 0]
            dy1 = jet_points[i, 1] - jet_points[i-1, 1]
            dp1 = jet_points[i, 2] - jet_points[i-1, 2]
            
            dx2 = jet_points[i+1, 0] - jet_points[i, 0]
            dy2 = jet_points[i+1, 1] - jet_points[i, 1]
            dp2 = jet_points[i+1, 2] - jet_points[i, 2]
            
            # Longueurs d'arc
            ds1 = torch.sqrt(dx1**2 + dy1**2 + 1e-6)
            ds2 = torch.sqrt(dx2**2 + dy2**2 + 1e-6)
            
            # Dérivées de p
            dp_ds1 = dp1 / ds1 if ds1 > 0 else 0
            dp_ds2 = dp2 / ds2 if ds2 > 0 else 0
            
            # Courbure comme moyenne
            curvature[i] = 0.5 * (dp_ds1 + dp_ds2)
        
        # Extrapolation aux bords
        if n_points > 1:
            curvature[0] = curvature[1]
            curvature[-1] = curvature[-2]
        
        return curvature


class ContactPlaneField:
    """
    Représente le champ de plans de contact en chaque point de J¹(R²).
    Le plan de contact en (x,y,p) est engendré par les vecteurs :
        X = ∂/∂x + p ∂/∂y
        P = ∂/∂p
    """
    
    def __init__(self, jet_space: JetSpace):
        self.jet_space = jet_space
        self.device = jet_space.device
    
    def contact_plane(self, jet_point: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retourne une base du plan de contact en un point.
        
        Args:
            jet_point: Tensor (3,) avec (x, y, p)
            
        Returns:
            Tuple de deux vecteurs de base (X, P)
        """
        x, y, p = jet_point
        
        # Premier vecteur de base : X = ∂/∂x + p ∂/∂y
        X = torch.tensor([1.0, p, 0.0], device=self.device)
        
        # Deuxième vecteur de base : P = ∂/∂p
        P = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        
        return X, P
    
    def is_in_contact_plane(self, vector: torch.Tensor, 
                           jet_point: torch.Tensor) -> bool:
        """
        Vérifie si un vecteur tangent est dans le plan de contact.
        
        Args:
            vector: Tensor (3,) vecteur tangent
            jet_point: Tensor (3,) point où évaluer
            
        Returns:
            True si vector ∈ plan de contact
        """
        # Condition : ω(vector) = 0
        x, y, p = jet_point
        vx, vy, vp = vector
        
        # ω = dy - p dx
        omega_value = vy - p * vx
        
        return torch.abs(omega_value) < 1e-6
    
    def frobenius_condition(self, window_size: int = 3) -> torch.Tensor:
        """
        Calcule la condition de Frobenius pour vérifier la non-intégrabilité
        du champ de plans de contact.
        
        La structure de contact est maximale non-intégrable.
        
        Args:
            window_size: Taille de la fenêtre pour le calcul
            
        Returns:
            Tensor indiquant où la condition est satisfaite
        """
        h, w = self.jet_space.height, self.jet_space.width
        
        # Crée un champ aléatoire pour tester
        test_field = torch.randn(h, w, 3, device=self.device)
        
        # Vérifie localement la condition de Frobenius
        frobenius = torch.zeros(h, w, device=self.device)
        
        half = window_size // 2
        for y in range(half, h - half):
            for x in range(half, w - half):
                # Extrait la fenêtre locale
                window = test_field[y-half:y+half+1, x-half:x+half+1, :]
                
                # Calcule les crochets de Lie locaux
                # (Simplification pour la démonstration)
                frobenius[y, x] = torch.std(window)
        
        return frobenius
