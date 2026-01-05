"""
Module visual_illusions.py - Autres illusions visuelles célèbres
"""

import torch
import numpy as np
from typing import Tuple, List
import math


class EhrensteinIllusion:
    """
    Illusion d'Ehrenstein - Contours radiaux créent un carré/cercle illusoire.
    """
    
    def __init__(self, size: int = 256, device: str = 'cpu'):
        self.size = size
        self.device = device
    
    def generate_stimulus(self,
                         n_lines: int = 12,
                         line_length: float = 0.4,
                         gap_size: float = 0.15) -> torch.Tensor:
        """
        Génère un stimulus d'Ehrenstein.
        
        Args:
            n_lines: Nombre de lignes radiales
            line_length: Longueur des lignes
            gap_size: Taille du trou central
            
        Returns:
            Image du stimulus
        """
        image = torch.zeros(self.size, self.size, device=self.device)
        center = self.size // 2
        
        # Rayons
        inner_radius = gap_size * self.size / 2
        outer_radius = (gap_size + line_length) * self.size / 2
        
        # Angles pour les lignes
        angles = np.linspace(0, 2*math.pi, n_lines, endpoint=False)
        
        for angle in angles:
            # Points de début et fin de la ligne
            x1 = center + inner_radius * math.cos(angle)
            y1 = center + inner_radius * math.sin(angle)
            x2 = center + outer_radius * math.cos(angle)
            y2 = center + outer_radius * math.sin(angle)
            
            # Dessine la ligne
            self._draw_line(image, x1, y1, x2, y2, thickness=2)
        
        return image
    
    def _draw_line(self, image, x1, y1, x2, y2, thickness=2):
        """Dessine une ligne épaisse."""
        h, w = image.shape
        
        # Paramètre t de 0 à 1
        length = max(math.hypot(x2-x1, y2-y1), 1)
        
        for t in np.linspace(0, 1, int(length)):
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            # Dessine un disque à chaque point
            radius = thickness // 2
            for dy in range(-radius, radius+1):
                for dx in range(-radius, radius+1):
                    if dx*dx + dy*dy <= radius*radius:
                        px = int(x + dx)
                        py = int(y + dy)
                        
                        if 0 <= px < w and 0 <= py < h:
                            image[py, px] = 1.0
    
    def predict_contour(self, stimulus: torch.Tensor) -> torch.Tensor:
        """
        Prédit le contour illusoire (carré ou cercle).
        
        Returns:
            Carte du contour
        """
        from scipy import ndimage
        
        stim_np = stimulus.cpu().numpy()
        h, w = stim_np.shape
        center_y, center_x = h // 2, w // 2
        
        # Détecte les extrémités des lignes
        sobel_x = ndimage.sobel(stim_np, axis=1)
        sobel_y = ndimage.sobel(stim_np, axis=0)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Points aux extrémités
        edge_points = np.column_stack(np.where(edges > 0.1))
        
        # Contour illusoire (cercle ou carré)
        contour = np.zeros((h, w))
        
        # Pour un carré de Kanizsa
        square_size = 0.25 * min(h, w)
        
        # Dessine un carré illusoire
        for y in range(h):
            for x in range(w):
                # Distance au centre
                dx = abs(x - center_x)
                dy = abs(y - center_y)
                
                # Bords du carré
                if (abs(dx - square_size) < 3 and dy < square_size) or \
                   (abs(dy - square_size) < 3 and dx < square_size):
                    contour[y, x] = 1.0
        
        return torch.tensor(contour, device=self.device)


class PetterEffect:
    """
    Effet Petter - Complétion entre régions de luminosité différente.
    """
    
    def generate_stimulus(self, size: int = 256) -> torch.Tensor:
        """
        Génère un stimulus montrant l'effet Petter.
        
        Deux régions de luminosité différente avec des contours alignés
        créent un contour illusoire continu.
        """
        image = torch.zeros(size, size)
        
        # Deux rectangles avec luminosité différente
        rect1_y1, rect1_y2 = size//4, 3*size//4
        rect1_x1, rect1_x2 = size//4, size//2
        
        rect2_y1, rect2_y2 = size//4, 3*size//4
        rect2_x1, rect2_x2 = size//2, 3*size//4
        
        # Rectangle 1 (clair)
        image[rect1_y1:rect1_y2, rect1_x1:rect1_x2] = 0.7
        
        # Rectangle 2 (foncé)
        image[rect2_y1:rect2_y2, rect2_x1:rect2_x2] = 0.3
        
        # Contour commun (aligné)
        for y in range(rect1_y1, rect1_y2):
            image[y, size//2] = 0.0  # Ligne de séparation
        
        return image
