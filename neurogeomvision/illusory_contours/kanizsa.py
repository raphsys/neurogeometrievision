"""
Module kanizsa.py - Illusions de Kanizsa (triangles et carrés)
"""

import torch
import numpy as np
from typing import Tuple, List, Optional
import math
import matplotlib.pyplot as plt


class KanizsaTriangle:
    """
    Triangle de Kanizsa - Le contour illusoire triangulaire.
    Trois 'pac-men' orientés créent la perception d'un triangle blanc.
    """
    
    def __init__(self, size: int = 256, device: str = 'cpu'):
        self.size = size
        self.device = device
        
    def generate_stimulus(self, 
                         triangle_size: float = 0.3,
                         pacman_radius: float = 0.08,
                         gap_angle: float = 60) -> torch.Tensor:
        """
        Génère un stimulus de triangle de Kanizsa.
        
        Args:
            triangle_size: Taille relative du triangle (0-1)
            pacman_radius: Rayon des pac-men
            gap_angle: Angle d'ouverture des pac-men (degrés)
            
        Returns:
            Image du stimulus (size, size)
        """
        image = torch.zeros(self.size, self.size, device=self.device)
        
        # Centre de l'image
        center = self.size // 2
        
        # Sommets du triangle équilatéral
        radius = triangle_size * self.size / 2
        
        # Angles des sommets (120° d'écart)
        angles = [0, 120, 240]  # En degrés
        
        # Positions des pac-men aux sommets
        pacman_positions = []
        for angle in angles:
            rad = math.radians(angle)
            x = center + radius * math.cos(rad)
            y = center + radius * math.sin(rad)
            pacman_positions.append((x, y, angle))
        
        # Dessine les pac-men
        gap_rad = math.radians(gap_angle / 2)
        
        for px, py, angle in pacman_positions:
            # Orientation du pac-man (vers l'intérieur du triangle)
            pacman_angle = math.radians(angle + 180)  # Tourné vers le centre
            
            # Dessine un disque avec un secteur manquant
            for y in range(self.size):
                for x in range(self.size):
                    # Distance au centre du pac-man
                    dx = x - px
                    dy = y - py
                    distance = math.sqrt(dx**2 + dy**2)
                    
                    if distance <= pacman_radius * self.size:
                        # Angle par rapport au centre
                        point_angle = math.atan2(dy, dx)
                        
                        # Différence avec l'orientation du pac-man
                        angle_diff = (point_angle - pacman_angle + math.pi) % (2*math.pi) - math.pi
                        
                        # Ne dessine pas dans le secteur manquant
                        if abs(angle_diff) > gap_rad:
                            image[y, x] = 1.0
        
        return image
    
    def predict_illusory_contour(self,
                                stimulus: torch.Tensor,
                                from_association_field = None) -> torch.Tensor:
        """
        Prédit le contour illusoire à partir du stimulus.
        
        Args:
            stimulus: Image du stimulus
            from_association_field: Module de champ d'association optionnel
            
        Returns:
            Carte du contour illusoire
        """
        if from_association_field is None:
            # Méthode simple basée sur les alignements
            return self._simple_completion(stimulus)
        else:
            # Utilise le champ d'association
            return self._association_field_completion(stimulus, from_association_field)
    
    def _simple_completion(self, stimulus: torch.Tensor) -> torch.Tensor:
        """Complétion simple basée sur les prolongements linéaires."""
        from scipy import ndimage
        
        stimulus_np = stimulus.cpu().numpy()
        
        # Détection des bords
        edges = ndimage.sobel(stimulus_np)
        
        # Prolongement des lignes
        h, w = stimulus_np.shape
        
        # Centre pour le triangle
        center_y, center_x = h // 2, w // 2
        
        # Crée le contour illusoire
        contour = np.zeros((h, w))
        
        # Points de départ (bords des pac-men)
        edge_points = np.column_stack(np.where(edges > 0.1))
        
        for y, x in edge_points[:100]:  # Limite
            # Orientation locale (approximative)
            # Pour Kanizsa, les lignes pointent vers le centre
            
            # Vecteur vers le centre
            dx = center_x - x
            dy = center_y - y
            
            # Normalise
            norm = max(math.sqrt(dx*dx + dy*dy), 1)
            dx, dy = dx/norm, dy/norm
            
            # Prolonge la ligne
            for t in range(1, 30):
                tx = int(x + t * dx)
                ty = int(y + t * dy)
                
                if 0 <= tx < w and 0 <= ty < h:
                    # Force décroissante avec la distance
                    strength = 1.0 - (t / 30)
                    contour[ty, tx] = max(contour[ty, tx], strength)
        
        return torch.tensor(contour, device=self.device)
    
    def visualize_kanizsa(self,
                         show_prediction: bool = True) -> dict:
        """
        Génère et visualise un triangle de Kanizsa.
        
        Returns:
            Dict avec stimulus et prédiction
        """
        # Génère le stimulus
        stimulus = self.generate_stimulus()
        
        # Prédit le contour
        if show_prediction:
            prediction = self.predict_illusory_contour(stimulus)
        else:
            prediction = None
        
        # Visualisation
        fig, axes = plt.subplots(1, 2 if show_prediction else 1, 
                                figsize=(10, 5) if show_prediction else (5, 5))
        
        if show_prediction:
            ax1, ax2 = axes
        else:
            ax1 = axes
            ax2 = None
        
        # Stimulus
        im1 = ax1.imshow(stimulus.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        ax1.set_title("Stimulus de Kanizsa")
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        # Prédiction
        if ax2 is not None and prediction is not None:
            im2 = ax2.imshow(prediction.cpu().numpy(), cmap='hot')
            ax2.set_title("Contour illusoire prédit")
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        plt.tight_layout()
        
        return {
            'stimulus': stimulus,
            'prediction': prediction,
            'figure': fig
        }


class KanizsaSquare:
    """
    Carré de Kanizsa - Quatre quarts de cercle créent un carré illusoire.
    """
    
    def __init__(self, size: int = 256, device: str = 'cpu'):
        self.size = size
        self.device = device
    
    def generate_stimulus(self, 
                         square_size: float = 0.4,
                         corner_radius: float = 0.1) -> torch.Tensor:
        """
        Génère un stimulus de carré de Kanizsa.
        
        Args:
            square_size: Taille relative du carré
            corner_radius: Rayon des quarts de cercle
            
        Returns:
            Image du stimulus
        """
        image = torch.zeros(self.size, self.size, device=self.device)
        center = self.size // 2
        half_size = square_size * self.size / 2
        
        # Quatre coins
        corners = [
            (center - half_size, center - half_size),  # Haut gauche
            (center + half_size, center - half_size),  # Haut droit
            (center + half_size, center + half_size),  # Bas droit
            (center - half_size, center + half_size),  # Bas gauche
        ]
        
        # Angles d'ouverture pour chaque quart de cercle
        corner_angles = [
            (0, 90),     # Haut gauche: 0-90°
            (90, 180),   # Haut droit: 90-180°
            (180, 270),  # Bas droit: 180-270°
            (270, 360),  # Bas gauche: 270-360°
        ]
        
        radius = corner_radius * self.size
        
        for (cx, cy), (angle_start, angle_end) in zip(corners, corner_angles):
            # Dessine un quart de cercle
            for y in range(self.size):
                for x in range(self.size):
                    dx = x - cx
                    dy = y - cy
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    if distance <= radius:
                        angle = math.degrees(math.atan2(dy, dx)) % 360
                        
                        # Vérifie si dans le secteur angulaire
                        if angle_start <= angle < angle_end:
                            image[y, x] = 1.0
        
        return image
    
    def predict_contours(self, stimulus: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prédit les contours horizontaux et verticaux.
        
        Returns:
            (contours_horizontaux, contours_verticaux)
        """
        from scipy import ndimage
        
        stim_np = stimulus.cpu().numpy()
        
        # Détecte les bords horizontaux
        horizontal = ndimage.sobel(stim_np, axis=0)
        
        # Détecte les bords verticaux
        vertical = ndimage.sobel(stim_np, axis=1)
        
        # Centre pour le carré
        h, w = stim_np.shape
        center_y, center_x = h // 2, w // 2
        
        # Prolonge les lignes horizontales
        horiz_contour = np.zeros((h, w))
        vert_contour = np.zeros((h, w))
        
        # Points de bords horizontaux
        horiz_points = np.column_stack(np.where(abs(horizontal) > 0.1))
        
        for y, x in horiz_points[:50]:
            # Prolonge à gauche et droite
            for dx in range(-30, 31):
                tx = x + dx
                if 0 <= tx < w:
                    strength = 1.0 - abs(dx) / 30
                    horiz_contour[y, tx] = max(horiz_contour[y, tx], strength)
        
        # Points de bords verticaux
        vert_points = np.column_stack(np.where(abs(vertical) > 0.1))
        
        for y, x in vert_points[:50]:
            # Prolonge en haut et bas
            for dy in range(-30, 31):
                ty = y + dy
                if 0 <= ty < h:
                    strength = 1.0 - abs(dy) / 30
                    vert_contour[ty, x] = max(vert_contour[ty, x], strength)
        
        return (torch.tensor(horiz_contour, device=self.device),
                torch.tensor(vert_contour, device=self.device))
