"""
Module heisenberg_group.py - Groupe de Heisenberg comme modèle local
"""

import torch
import numpy as np
from typing import Tuple, Optional
import math


class HeisenbergGroup:
    """
    Groupe de Heisenberg H³ comme approximation locale de la structure de contact.
    
    Le groupe de Heisenberg est le groupe nilpotent de pas 2:
    (x, y, p) · (x', y', p') = (x+x', y+y', p+p' + 1/2(xy' - x'y))
    
    C'est le "cône tangent" à l'origine de la structure de contact.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def group_law(self, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        """
        Loi de groupe du groupe de Heisenberg.
        
        Args:
            g1, g2: Éléments du groupe (x, y, p)
            
        Returns:
            Produit g1 · g2
        """
        x1, y1, p1 = g1
        x2, y2, p2 = g2
        
        x = x1 + x2
        y = y1 + y2
        p = p1 + p2 + 0.5 * (x1 * y2 - y1 * x2)
        
        return torch.tensor([x, y, p], device=self.device)
    
    def inverse(self, g: torch.Tensor) -> torch.Tensor:
        """
        Inverse d'un élément du groupe.
        
        Args:
            g: (x, y, p)
            
        Returns:
            g^{-1}
        """
        x, y, p = g
        return torch.tensor([-x, -y, -p], device=self.device)
    
    def left_invariant_vector_fields(self, point: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Champs de vecteurs invariants à gauche.
        
        Args:
            point: Point où évaluer
            
        Returns:
            Base (X, Y, P) des champs invariants à gauche
        """
        # En coordonnées canoniques:
        X = torch.tensor([1.0, 0.0, -0.5 * point[1]], device=self.device)
        Y = torch.tensor([0.0, 1.0, 0.5 * point[0]], device=self.device)
        P = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        
        return X, Y, P
    
    def exponential_map(self, tangent_vector: torch.Tensor) -> torch.Tensor:
        """
        Application exponentielle du groupe de Heisenberg.
        
        Args:
            tangent_vector: (vx, vy, vp) dans l'algèbre de Lie
            
        Returns:
            Point du groupe
        """
        vx, vy, vp = tangent_vector
        
        if abs(vx) < 1e-6 and abs(vy) < 1e-6:
            # Cas dégénéré
            return torch.tensor([0.0, 0.0, vp], device=self.device)
        
        # Formule exacte pour l'exponentielle
        norm = torch.sqrt(vx**2 + vy**2)
        sin_norm = torch.sin(norm / 2)
        cos_norm = torch.cos(norm / 2)
        
        x = (sin_norm / (norm / 2)) * vx if norm > 0 else vx
        y = (sin_norm / (norm / 2)) * vy if norm > 0 else vy
        p = vp + (1 - cos_norm) / norm**2 * (vx * vy) if norm > 0 else vp
        
        return torch.tensor([x, y, p], device=self.device)
    
    def logarithm_map(self, group_element: torch.Tensor) -> torch.Tensor:
        """
        Application logarithme (inverse de l'exponentielle).
        
        Args:
            group_element: (x, y, p) dans le groupe
            
        Returns:
            Vecteur tangent
        """
        x, y, p = group_element
        
        if abs(x) < 1e-6 and abs(y) < 1e-6:
            return torch.tensor([0.0, 0.0, p], device=self.device)
        
        norm = torch.sqrt(x**2 + y**2)
        theta = 2 * torch.asin(norm / 2)
        
        vx = (theta / torch.sin(theta / 2)) * x / 2 if torch.sin(theta / 2) > 0 else x
        vy = (theta / torch.sin(theta / 2)) * y / 2 if torch.sin(theta / 2) > 0 else y
        vp = p - (1 - torch.cos(theta / 2)) / (theta**2 / 4) * (x * y) if theta > 0 else p
        
        return torch.tensor([vx, vy, vp], device=self.device)
    
    def left_invariant_metric(self, point: torch.Tensor,
                             vector1: torch.Tensor,
                             vector2: torch.Tensor) -> float:
        """
        Métrique invariante à gauche standard.
        
        Args:
            point: Point où évaluer
            vector1, vector2: Vecteurs tangents
            
        Returns:
            Produit scalaire
        """
        # Exprimé dans la base des champs invariants à gauche
        X, Y, P = self.left_invariant_vector_fields(point)
        
        # Coefficients dans cette base
        # On résout: vector = aX + bY + cP
        # C'est plus simple au niveau de l'algèbre de Lie
        
        # Pour la métrique standard: produit scalaire euclidien sur les coefficients
        # On projette sur l'espace horizontal (X, Y)
        vx1, vy1, vp1 = vector1
        vx2, vy2, vp2 = vector2
        
        # Métrique standard: seulement les composantes horizontales comptent
        return (vx1 * vx2 + vy1 * vy2).item()
    
    def geodesic_equation(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Équations des géodésiques pour la métrique invariante à gauche.
        
        Args:
            t: Paramètre
            state: [x, y, p, vx, vy, vp]
            
        Returns:
            Dérivées
        """
        x, y, p, vx, vy, vp = state
        
        # Pour la métrique standard invariante à gauche
        # Les géodésiques sont des droites dans l'algèbre de Lie
        dx_dt = vx
        dy_dt = vy
        dp_dt = vp + 0.5 * (x * vy - y * vx)
        
        # Vitesses constantes (géodésiques)
        dvx_dt = 0
        dvy_dt = 0
        dvp_dt = 0
        
        return np.array([dx_dt, dy_dt, dp_dt, dvx_dt, dvy_dt, dvp_dt])
    
    def compute_heisenberg_geodesic(self, start: torch.Tensor,
                                   direction: torch.Tensor,
                                   duration: float = 1.0,
                                   n_steps: int = 100) -> torch.Tensor:
        """
        Calcule une géodésique dans le groupe de Heisenberg.
        
        Args:
            start: Point de départ
            direction: Vecteur tangent initial
            duration: Durée
            n_steps: Nombre de pas
            
        Returns:
            Géodésique
        """
        t = torch.linspace(0, duration, n_steps, device=self.device)
        
        # Solution exacte pour les géodésiques
        # Pour la métrique standard: droites dans l'algèbre de Lie
        vx, vy, vp = direction
        
        geodesic = torch.zeros(n_steps, 3, device=self.device)
        
        for i, ti in enumerate(t):
            # Transport parallèle
            x = start[0] + vx * ti
            y = start[1] + vy * ti
            p = start[2] + vp * ti + 0.5 * (start[0] * vy * ti - start[1] * vx * ti)
            
            geodesic[i] = torch.tensor([x, y, p], device=self.device)
        
        return geodesic
    
    def heisenberg_distance(self, g1: torch.Tensor, g2: torch.Tensor) -> float:
        """
        Distance dans le groupe de Heisenberg (distance de Carnot-Carathéodory).
        
        Args:
            g1, g2: Points du groupe
            
        Returns:
            Distance
        """
        # Inverse de g1
        g1_inv = self.inverse(g1)
        
        # g = g1^{-1} · g2
        g = self.group_law(g1_inv, g2)
        
        # Logarithme
        v = self.logarithm_map(g)
        
        # Norme du vecteur tangent
        norm = torch.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        
        return norm.item()
    
    def heisenberg_sphere(self, center: torch.Tensor,
                         radius: float,
                         n_points: int = 1000) -> torch.Tensor:
        """
        Sphère dans le groupe de Heisenberg.
        
        Args:
            center: Centre
            radius: Rayon
            n_points: Nombre de points
            
        Returns:
            Points sur la sphère
        """
        points = []
        
        # Génère des directions uniformes
        for _ in range(n_points):
            # Direction aléatoire dans l'algèbre de Lie
            vx = torch.randn(1, device=self.device).item()
            vy = torch.randn(1, device=self.device).item()
            vp = torch.randn(1, device=self.device).item()
            
            # Normalise
            norm = math.sqrt(vx**2 + vy**2 + vp**2)
            vx, vy, vp = vx/norm, vy/norm, vp/norm
            
            direction = torch.tensor([vx, vy, vp], device=self.device)
            
            # Exponentielle
            point = self.exponential_map(radius * direction)
            
            # Translate au centre
            point = self.group_law(center, point)
            
            points.append(point)
        
        return torch.stack(points)
