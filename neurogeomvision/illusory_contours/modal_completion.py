"""
Module modal_completion.py - Complétion modale générique
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
import math
from scipy import interpolate


class ModalCompletion:
    """
    Complétion modale générique pour les contours fragmentés.
    
    La complétion modale se produit lorsque nous percevons un contour
    qui n'est pas physiquement présent mais est induit par l'alignement
    des fragments.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def complete_contour(self,
                        fragments: List[Tuple[float, float, float]],
                        method: str = 'bezier') -> torch.Tensor:
        """
        Complète un contour à partir de fragments alignés.
        
        Args:
            fragments: Liste de (x, y, orientation) pour chaque fragment
            method: 'linear', 'bezier', ou 'spline'
            
        Returns:
            Carte du contour complété
        """
        if len(fragments) < 2:
            raise ValueError("Au moins 2 fragments nécessaires")
        
        # Extrait positions et orientations
        positions = [(f[0], f[1]) for f in fragments]
        orientations = [f[2] for f in fragments]
        
        if method == 'linear':
            return self._linear_completion(positions, orientations)
        elif method == 'bezier':
            return self._bezier_completion(positions, orientations)
        elif method == 'spline':
            return self._spline_completion(positions, orientations)
        else:
            raise ValueError(f"Méthode inconnue: {method}")
    
    def _linear_completion(self,
                          positions: List[Tuple[float, float]],
                          orientations: List[float]) -> torch.Tensor:
        """Complétion linéaire simple."""
        # Crée une image
        size = 256
        contour = torch.zeros(size, size, device=self.device)
        
        # Pour chaque fragment, prolonge selon son orientation
        for (x, y), theta in zip(positions, orientations):
            # Convertit en coordonnées pixel
            px = int(x * size)
            py = int(y * size)
            
            # Direction
            dx = math.cos(theta)
            dy = math.sin(theta)
            
            # Prolonge dans les deux directions
            for direction in [-1, 1]:
                for t in range(0, 50):
                    tx = int(px + direction * t * dx)  # CONVERTIT EN INT
                    ty = int(py + direction * t * dy)  # CONVERTIT EN INT
                    
                    if 0 <= tx < size and 0 <= ty < size:
                        strength = 1.0 - (t / 50)
                        contour[ty, tx] = max(contour[ty, tx].item(), strength)
        
        return contour
    
    def _bezier_completion(self,
                          positions: List[Tuple[float, float]],
                          orientations: List[float]) -> torch.Tensor:
        """Complétion par courbe de Bézier."""
        size = 256
        contour = torch.zeros(size, size, device=self.device)
        
        if len(positions) < 2:
            return self._linear_completion(positions, orientations)
        
        # Points de contrôle pour la courbe de Bézier
        control_points = []
        
        for (x, y), theta in zip(positions, orientations):
            px = x * size
            py = y * size
            
            # Point de contrôle
            control_points.append([px, py])
            
            # Point tangent (pour la direction)
            if len(control_points) < 4:  # Limite à 4 points
                tx = px + 20 * math.cos(theta)
                ty = py + 20 * math.sin(theta)
                control_points.append([tx, ty])
        
        if len(control_points) < 4:
            return self._linear_completion(positions, orientations)
        
        # Prend seulement les 4 premiers points pour Bézier cubique
        points = np.array(control_points[:4], dtype=float)
        
        # Courbe de Bézier cubique
        t = np.linspace(0, 1, 100)
        
        try:
            # Bézier cubique
            curve = self._cubic_bezier(points[0], points[1], 
                                      points[2], points[3], t)
            
            # Dessine la courbe
            for i in range(len(curve) - 1):
                x1, y1 = curve[i]
                x2, y2 = curve[i + 1]
                
                # Interpolation linéaire entre les points
                steps = max(1, int(math.hypot(x2-x1, y2-y1)))
                for s in range(steps + 1):
                    sx = int(x1 + s * (x2-x1) / steps)
                    sy = int(y1 + s * (y2-y1) / steps)
                    
                    if 0 <= sx < size and 0 <= sy < size:
                        contour[sy, sx] = 1.0
        except Exception as e:
            print(f"  Bézier échouée: {e}")
            # Fallback linéaire
            contour = self._linear_completion(positions, orientations)
        
        return contour

    
    def _cubic_bezier(self, p0, p1, p2, p3, t):
        """Courbe de Bézier cubique."""
        t = np.array(t)
        one_minus_t = 1 - t
        
        # Convertit les points en arrays numpy
        p0 = np.array(p0, dtype=float)
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        p3 = np.array(p3, dtype=float)
        
        # Broadcasting correct
        curve = (one_minus_t[:, np.newaxis]**3 * p0 + 
                 3 * one_minus_t[:, np.newaxis]**2 * t[:, np.newaxis] * p1 +
                 3 * one_minus_t[:, np.newaxis] * t[:, np.newaxis]**2 * p2 +
                 t[:, np.newaxis]**3 * p3)
        
        return curve
        
    
    def _spline_completion(self,
                          positions: List[Tuple[float, float]],
                          orientations: List[float]) -> torch.Tensor:
        """Complétion par spline."""
        size = 256
        contour = torch.zeros(size, size, device=self.device)
        
        # Points pour la spline
        points = np.array([(x*size, y*size) for x, y in positions])
        
        if len(points) < 3:
            return self._linear_completion(positions, orientations)
        
        try:
            # Spline cubique
            t = np.arange(len(points))
            spline_x = interpolate.CubicSpline(t, points[:, 0])
            spline_y = interpolate.CubicSpline(t, points[:, 1])
            
            # Échantillonne la spline
            t_dense = np.linspace(0, len(points)-1, 200)
            curve_x = spline_x(t_dense)
            curve_y = spline_y(t_dense)
            
            # Dessine la courbe
            for i in range(len(curve_x) - 1):
                x1, y1 = curve_x[i], curve_y[i]
                x2, y2 = curve_x[i+1], curve_y[i+1]
                
                steps = int(math.hypot(x2-x1, y2-y1))
                if steps > 0:
                    for s in range(steps + 1):
                        sx = int(x1 + s * (x2-x1) / steps)
                        sy = int(y1 + s * (y2-y1) / steps)
                        
                        if 0 <= sx < size and 0 <= sy < size:
                            contour[sy, sx] = 1.0
        except:
            # Fallback
            contour = self._linear_completion(positions, orientations)
        
        return contour
    
    def generate_fragmented_line(self,
                                n_fragments: int = 5,
                                gap_size: float = 0.1) -> List[Tuple[float, float, float]]:
        """
        Génère une ligne fragmentée pour tester la complétion.
        
        Args:
            n_fragments: Nombre de fragments
            gap_size: Taille des espaces entre fragments
            
        Returns:
            Liste de fragments (x, y, orientation)
        """
        fragments = []
        
        # Ligne horizontale avec trous
        for i in range(n_fragments):
            # Position
            x = 0.2 + i * (0.6 / n_fragments)
            y = 0.5
            
            # Orientation horizontale
            theta = 0.0
            
            fragments.append((x, y, theta))
        
        return fragments
