"""
Module legendrian_lifts.py - Relevées legendriennes et intégration
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
import math
from .jet_space import JetSpace


class LegendrianLifts:
    """
    Gestion des relevées legendriennes de courbes dans l'espace de contact.
    
    Une courbe legendrienne est tangente aux plans de contact.
    Ces courbes modélisent les contours intégrés par V1.
    """
    
    def __init__(self, jet_space: JetSpace):
        self.jet_space = jet_space
        self.device = jet_space.device
    
    def lift_straight_line(self, start: Tuple[float, float],
                          end: Tuple[float, float],
                          n_points: int = 50) -> torch.Tensor:
        """
        Relève une ligne droite vers une courbe legendrienne.
        
        Args:
            start: Point de départ (x, y)
            end: Point d'arrivée (x, y)
            n_points: Nombre de points
            
        Returns:
            Courbe legendrienne dans J¹(R²)
        """
        # Interpolation linéaire dans la base
        t = torch.linspace(0, 1, n_points, device=self.device)
        start_tensor = torch.tensor(start, device=self.device)
        end_tensor = torch.tensor(end, device=self.device)
        
        base_curve = start_tensor + t.unsqueeze(1) * (end_tensor - start_tensor)
        
        # Orientation constante (pente de la ligne)
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        if abs(dx) > 1e-6:
            p = dy / dx
        else:
            p = float('inf') if dy > 0 else float('-inf')
        
        # Crée la courbe jet
        jet_curve = torch.zeros(n_points, 3, device=self.device)
        jet_curve[:, :2] = base_curve
        jet_curve[:, 2] = p
        
        return jet_curve
    
    def lift_circular_arc(self, center: Tuple[float, float],
                         radius: float,
                         start_angle: float,
                         end_angle: float,
                         n_points: int = 50) -> torch.Tensor:
        """
        Relève un arc de cercle vers une courbe legendrienne.
        
        Args:
            center: Centre du cercle (cx, cy)
            radius: Rayon
            start_angle: Angle de départ (radians)
            end_angle: Angle d'arrivée (radians)
            n_points: Nombre de points
            
        Returns:
            Courbe legendrienne dans J¹(R²)
        """
        angles = torch.linspace(start_angle, end_angle, n_points, device=self.device)
        
        # Points de base
        cx, cy = center
        base_curve = torch.zeros(n_points, 2, device=self.device)
        base_curve[:, 0] = cx + radius * torch.cos(angles)
        base_curve[:, 1] = cy + radius * torch.sin(angles)
        
        # Orientation = tangente au cercle
        # Pour un cercle (x-cx)² + (y-cy)² = r²
        # dy/dx = -(x-cx)/(y-cy)
        x_rel = base_curve[:, 0] - cx
        y_rel = base_curve[:, 1] - cy
        
        # Évite la division par zéro
        p_values = torch.zeros(n_points, device=self.device)
        for i in range(n_points):
            if abs(y_rel[i]) > 1e-6:
                p_values[i] = -x_rel[i] / y_rel[i]
            else:
                # Tangente verticale
                p_values[i] = float('inf') if x_rel[i] < 0 else float('-inf')
        
        # Combine
        jet_curve = torch.cat([base_curve, p_values.unsqueeze(1)], dim=1)
        
        return jet_curve
    
    def euler_integration(self, start_point: torch.Tensor,
                         vector_field: callable,
                         steps: int = 100,
                         step_size: float = 0.1) -> torch.Tensor:
        """
        Intégration d'Euler d'un champ de vecteurs dans l'espace de contact.
        
        Args:
            start_point: Point de départ (x, y, p)
            vector_field: Fonction qui retourne un vecteur tangent
            steps: Nombre de pas
            step_size: Taille du pas
            
        Returns:
            Courbe intégrée
        """
        curve = torch.zeros(steps + 1, 3, device=self.device)
        curve[0] = start_point
        
        for i in range(steps):
            current = curve[i]
            vector = vector_field(current)
            
            # Pas d'Euler
            next_point = current + step_size * vector
            
            # Projette sur la structure de contact si nécessaire
            if not self.jet_space.is_legendrian(torch.stack([current, next_point])):
                # Ajuste pour rester legendrien
                x, y, p = current
                dx, dy, dp = vector
                
                # Condition legendrienne : dy = p dx
                # On ajuste dy ou p
                p_new = dy / dx if abs(dx) > 1e-6 else p
                next_point = torch.tensor([
                    x + dx,
                    y + dy,
                    p + dp
                ], device=self.device)
            
            curve[i + 1] = next_point
        
        return curve
    
    def generate_test_contours(self, contour_type: str = 'simple') -> List[torch.Tensor]:
        """
        Génère des contours de test pour la validation.
        
        Args:
            contour_type: 'simple', 'circle', 'spiral', 'random'
            
        Returns:
            Liste de contours
        """
        contours = []
        
        if contour_type == 'simple':
            # Lignes droites
            lines = [
                [(10, 10), (100, 10)],   # Horizontal
                [(10, 10), (10, 100)],   # Vertical
                [(10, 10), (100, 100)],  # Diagonale
                [(100, 10), (10, 100)],  # Anti-diagonale
            ]
            
            for start, end in lines:
                contour = self.lift_straight_line(start, end)
                contours.append(contour)
        
        elif contour_type == 'circle':
            # Arcs de cercle
            arcs = [
                [(50, 50), 30, 0, math.pi],      # Demi-cercle supérieur
                [(50, 50), 30, math.pi, 2*math.pi],  # Demi-cercle inférieur
            ]
            
            for center, radius, start, end in arcs:
                contour = self.lift_circular_arc(center, radius, start, end)
                contours.append(contour)
        
        elif contour_type == 'spiral':
            # Spirale logarithmique
            n_points = 100
            t = torch.linspace(0, 2*math.pi, n_points, device=self.device)
            
            # Paramètres de la spirale
            a = 0.1
            b = 0.3
            
            x = 50 + a * torch.exp(b * t) * torch.cos(t)
            y = 50 + a * torch.exp(b * t) * torch.sin(t)
            
            base_curve = torch.stack([x, y], dim=1)
            
            # Orientation = angle de la tangente
            dx = torch.gradient(x)[0]
            dy = torch.gradient(y)[0]
            
            # Évite la division par zéro
            p = torch.zeros(n_points, device=self.device)
            for i in range(n_points):
                if abs(dx[i]) > 1e-6:
                    p[i] = dy[i] / dx[i]
                else:
                    p[i] = float('inf') if dy[i] > 0 else float('-inf')
            
            contour = torch.cat([base_curve, p.unsqueeze(1)], dim=1)
            contours.append(contour)
        
        return contours
    
    def compute_legendrian_energy(self, contour: torch.Tensor) -> float:
        """
        Calcule l'énergie legendrienne d'un contour.
        Mesure l'écart à la condition legendrienne.
        
        Args:
            contour: Tensor (n_points, 3)
            
        Returns:
            Énergie legendrienne
        """
        if contour.shape[0] < 2:
            return 0.0
        
        energy = 0.0
        
        for i in range(contour.shape[0] - 1):
            p1 = contour[i]
            p2 = contour[i + 1]
            
            # Différences
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # p moyen
            p_avg = 0.5 * (p1[2] + p2[2])
            
            # Condition legendrienne : dy = p_avg * dx
            error = dy - p_avg * dx
            energy += error ** 2
        
        return energy.item()
    
    def smooth_contour(self, contour: torch.Tensor,
                      window_size: int = 3) -> torch.Tensor:
        """
        Lisse un contour en utilisant un filtre moyenneur.
        
        Args:
            contour: Tensor (n_points, 3)
            window_size: Taille de la fenêtre de lissage
            
        Returns:
            Contour lissé
        """
        if contour.shape[0] < window_size:
            return contour
        
        smoothed = contour.clone()
        half = window_size // 2
        
        # Applique un filtre moyenneur
        for i in range(half, contour.shape[0] - half):
            window = contour[i-half:i+half+1, :]
            smoothed[i] = window.mean(dim=0)
        
        return smoothed
