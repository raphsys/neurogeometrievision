"""
Module contact_space.py - Structure de contact principale pour V1
"""

import torch
import numpy as np
from typing import Tuple, Optional, List, Dict
import math
from .jet_space import JetSpace, ContactPlaneField


class ContactStructureV1:
    """
    Implémente la structure de contact de V1 selon le modèle de Petitot.
    
    L'espace de contact V = R² × P¹ muni de la forme ω = dy - p dx
    est invariant sous l'action du groupe euclidien E(2).
    """
    
    def __init__(self, spatial_shape: Tuple[int, int], 
                 orientation_bins: int = 36,
                 device: str = 'cpu'):
        """
        Args:
            spatial_shape: (height, width) du champ visuel
            orientation_bins: Nombre de discrétisations d'orientation
            device: 'cpu' or 'cuda'
        """
        self.spatial_shape = spatial_shape
        self.orientation_bins = orientation_bins
        self.device = device
        
        # Espace des jets sous-jacent
        self.jet_space = JetSpace(spatial_shape, device)
        
        # Champ de plans de contact
        self.contact_plane_field = ContactPlaneField(self.jet_space)
        
        # Discrétisation des orientations
        self.theta_values = torch.linspace(0, math.pi, orientation_bins, device=device)
        self.p_values = torch.tan(self.theta_values)  # p = tan(θ)
        
        # Grille complète de l'espace de contact
        self.contact_grid = self._create_contact_grid()
        
        # Métrique sous-riemannienne (à initialiser plus tard)
        self.subriemannian_metric = None
        
    def _create_contact_grid(self) -> Dict[str, torch.Tensor]:
        """Crée une grille discrète de l'espace de contact V."""
        h, w = self.spatial_shape
        
        # Coordonnées spatiales
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, device=self.device),
            torch.arange(w, device=self.device),
            indexing='ij'
        )
        
        # Pour chaque orientation, créer une slice
        grid = {}
        for i, theta in enumerate(self.theta_values):
            p = self.p_values[i]
            
            # Crée une couche 2D pour cette orientation
            layer = torch.stack([x_coords, y_coords, 
                                torch.full((h, w), p, device=self.device)], dim=-1)
            grid[f"theta_{i:03d}"] = layer
        
        return grid
    
    def e2_invariance(self, translation: Tuple[float, float], 
                     rotation: float) -> bool:
        """
        Vérifie l'invariance sous l'action du groupe euclidien E(2).
        
        Args:
            translation: (dx, dy) translation
            rotation: angle de rotation (radians)
            
        Returns:
            True si la structure est invariante
        """
        # Test sur un point aléatoire
        test_point = torch.tensor([10.0, 10.0, 0.5], device=self.device)
        
        # Applique la transformation E(2)
        x, y, p = test_point
        dx, dy = translation
        
        # Rotation des coordonnées spatiales
        x_rot = x * math.cos(rotation) - y * math.sin(rotation)
        y_rot = x * math.sin(rotation) + y * math.cos(rotation)
        
        # Transformation de la pente
        # p' = (p * cosθ + sinθ) / (cosθ - p * sinθ)
        cosθ, sinθ = math.cos(rotation), math.sin(rotation)
        if cosθ - p * sinθ != 0:
            p_rot = (p * cosθ + sinθ) / (cosθ - p * sinθ)
        else:
            p_rot = float('inf')
        
        # Applique la translation
        x_trans = x_rot + dx
        y_trans = y_rot + dy
        
        transformed_point = torch.tensor([x_trans, y_trans, p_rot], device=self.device)
        
        # La forme de contact doit être préservée
        omega_orig = self.jet_space.contact_form(test_point)
        omega_trans = self.jet_space.contact_form(transformed_point)
        
        return torch.abs(omega_orig - omega_trans) < 1e-6
    
    def legendrian_lift(self, base_curve: torch.Tensor, 
                       orientation_field: torch.Tensor) -> torch.Tensor:
        """
        Relève une courbe de la base R² vers J¹(R²) selon un champ d'orientation.
        
        Args:
            base_curve: Tensor (n_points, 2) points (x, y)
            orientation_field: Tensor (height, width) angles θ
            
        Returns:
            Tensor (n_points, 3) points jet (x, y, p)
        """
        n_points = base_curve.shape[0]
        jet_curve = torch.zeros(n_points, 3, device=self.device)
        
        for i in range(n_points):
            x, y = base_curve[i]
            
            # Convertit en indices entiers pour l'interpolation
            xi, yi = int(x), int(y)
            
            # Clampe les indices
            xi = max(0, min(xi, self.spatial_shape[1] - 1))
            yi = max(0, min(yi, self.spatial_shape[0] - 1))
            
            # Orientation au point (interpolation simple)
            theta = orientation_field[yi, xi]
            p = torch.tan(theta)
            
            jet_curve[i] = torch.tensor([x, y, p], device=self.device)
        
        return jet_curve
    
    def compute_contact_energy(self, jet_curve: torch.Tensor) -> float:
        """
        Calcule l'énergie de contact d'une courbe.
        Mesure à quel point la courbe est legendrienne.
        
        Args:
            jet_curve: Tensor (n_points, 3)
            
        Returns:
            Énergie de contact (plus petite = plus legendrienne)
        """
        n_points = jet_curve.shape[0]
        if n_points < 2:
            return 0.0
        
        energy = 0.0
        
        for i in range(n_points - 1):
            p1 = jet_curve[i]
            p2 = jet_curve[i + 1]
            
            # Différences
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dp = p2[2] - p1[2]
            
            # Énergie basée sur la condition legendrienne
            p_avg = 0.5 * (p1[2] + p2[2])
            legendrian_error = dy - p_avg * dx
            
            energy += legendrian_error ** 2
        
        return energy / (n_points - 1)
    
    def parallel_transport(self, start_point: torch.Tensor,
                         direction: torch.Tensor,
                         steps: int = 10,
                         step_size: float = 1.0) -> torch.Tensor:
        """
        Transport parallèle dans la structure de contact.
        Simule les connexions horizontales de V1.
        
        Args:
            start_point: Tensor (3,) point de départ (x, y, p)
            direction: Tensor (2,) direction dans la base (dx, dy)
            steps: Nombre de pas
            step_size: Taille du pas
            
        Returns:
            Chemin transporté dans J¹(R²)
        """
        path = torch.zeros(steps + 1, 3, device=self.device)
        path[0] = start_point
        
        # Normalise la direction
        dir_norm = torch.norm(direction)
        if dir_norm > 0:
            direction = direction / dir_norm
        
        for i in range(steps):
            current = path[i]
            x, y, p = current
            
            # Mouvement dans la base
            dx = direction[0] * step_size
            dy = direction[1] * step_size
            
            # Pour rester dans le plan de contact, on doit ajuster p
            # Condition legendrienne : dy = p dx
            if abs(dx) > 1e-6:
                # Si dx ≠ 0, on peut calculer le nouveau p
                p_new = dy / dx
            else:
                # Si dx = 0, p peut être quelconque (vertical)
                p_new = p
            
            # Nouveau point
            new_point = torch.tensor([
                x + dx,
                y + dy,
                p_new
            ], device=self.device)
            
            path[i + 1] = new_point
        
        return path
    
    def create_association_field(self, 
                               reference_orientation: float,
                               spatial_range: int = 20) -> torch.Tensor:
        """
        Crée un champ d'association local pour une orientation de référence.
        Modélise les connexions horizontales de V1.
        
        Args:
            reference_orientation: Orientation de référence (radians)
            spatial_range: Rayon spatial
            
        Returns:
            Tensor (2*range+1, 2*range+1, 3) champ local
        """
        size = 2 * spatial_range + 1
        field = torch.zeros(size, size, 3, device=self.device)
        
        # Centre du champ
        center = spatial_range
        pref_p = torch.tan(torch.tensor(reference_orientation, device=self.device))
        
        for y in range(size):
            for x in range(size):
                # Coordonnées relatives au centre
                dx = x - center
                dy = y - center
                
                # Distance et angle
                dist = math.sqrt(dx**2 + dy**2)
                if dist == 0:
                    angle = 0
                else:
                    angle = math.atan2(dy, dx)
                
                # Règle d'association de Field, Hayes & Hess (1993) :
                # La force de connexion dépend de la cohérence d'orientation
                # et de l'alignement collinéaire
                
                # Orientation cible (idéalement alignée avec la référence)
                target_angle = reference_orientation
                
                # Ajuste p selon la position relative
                # Pour les connexions collinéaires, p change linéairement
                p = pref_p
                
                field[y, x, 0] = dx
                field[y, x, 1] = dy
                field[y, x, 2] = p
        
        return field
    
    def integrate_contour(self, seed_points: List[torch.Tensor],
                         orientation_map: torch.Tensor,
                         max_steps: int = 100,
                         threshold: float = 0.1) -> List[torch.Tensor]:
        """
        Intègre un contour à partir de points seeds en utilisant la structure de contact.
        
        Args:
            seed_points: Liste de points de départ (x, y, p)
            orientation_map: Carte d'orientation du champ visuel
            max_steps: Nombre maximum de pas d'intégration
            threshold: Seuil d'arrêt pour la cohérence
            
        Returns:
            Liste de contours intégrés
        """
        contours = []
        
        for seed in seed_points:
            contour = [seed.clone()]
            current = seed.clone()
            
            # Direction initiale basée sur l'orientation
            theta = torch.atan(torch.tensor(current[2].item(), device=self.device))
            direction = torch.tensor([math.cos(theta), math.sin(theta)], device=self.device)
            
            for step in range(max_steps):
                # Avance dans la direction courante
                next_point = current[:2] + direction
                
                # Clampe aux limites
                x, y = next_point
                x = max(0, min(x, self.spatial_shape[1] - 1))
                y = max(0, min(y, self.spatial_shape[0] - 1))
                
                # Obtient l'orientation locale
                xi, yi = int(x), int(y)
                local_theta = orientation_map[yi, xi]
                local_p = torch.tan(local_theta)
                
                # Nouveau point jet
                new_jet = torch.tensor([x, y, local_p], device=self.device)
                
                # Vérifie la cohérence
                if step > 0:
                    coherence = self._compute_coherence(contour[-1], new_jet)
                    if coherence < threshold:
                        break
                
                contour.append(new_jet)
                current = new_jet
                
                # Met à jour la direction
                theta = local_theta
                direction = torch.tensor([math.cos(theta), math.sin(theta)], device=self.device)
            
            # Intègre aussi dans l'autre direction
            # (même logique mais direction opposée)
            
            contours.append(torch.stack(contour))
        
        return contours
    
    def _compute_coherence(self, point1: torch.Tensor, 
                          point2: torch.Tensor) -> float:
        """Calcule la cohérence entre deux points jet."""
        # Différence d'orientation normalisée
        p1, p2 = point1[2], point2[2]
        theta1 = torch.atan(p1)
        theta2 = torch.atan(p2)
        
        angular_diff = torch.abs(theta1 - theta2)
        # Normalise entre 0 et π
        angular_diff = torch.min(angular_diff, math.pi - angular_diff)
        
        # Cohérence décroît avec la différence angulaire
        coherence = torch.exp(-angular_diff / (math.pi / 8))
        
        return coherence.item()
    
    def visualize_contact_space(self, plane: str = 'xy'):
        """
        Visualise une coupe de l'espace de contact.
        
        Args:
            plane: Plan de coupe ('xy', 'xp', ou 'yp')
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Prend un sous-échantillon pour la visualisation
        subsample = 4
        h, w = self.spatial_shape
        h_sub, w_sub = h // subsample, w // subsample
        
        fig = plt.figure(figsize=(12, 10))
        
        if plane == 'xy':
            # Coupe XY à p fixe
            ax = fig.add_subplot(111)
            
            # Prend plusieurs valeurs de p
            p_indices = [0, len(self.p_values)//2, -1]
            colors = ['r', 'g', 'b']
            
            for idx, color in zip(p_indices, colors):
                p = self.p_values[idx].item()
                
                # Points où ω = 0 (plan de contact)
                x_vals = np.arange(0, w_sub)
                y_vals = p * x_vals
                
                ax.plot(x_vals, y_vals, color=color, 
                       label=f'p = {p:.2f} (θ = {self.theta_values[idx]:.2f} rad)')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Coupes XY des plans de contact')
            ax.legend()
            ax.grid(True)
            
        elif plane in ['xp', 'yp']:
            # Visualisation 3D
            ax = fig.add_subplot(111, projection='3d')
            
            # Échantillonne des points
            sample_points = []
            for y in range(0, h, subsample * 2):
                for x in range(0, w, subsample * 2):
                    for theta_idx in range(0, len(self.theta_values), 4):
                        p = self.p_values[theta_idx].item()
                        sample_points.append([x, y, p])
            
            if sample_points:
                points = np.array(sample_points)
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          c=points[:, 2], cmap='hsv', alpha=0.6)
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('p = tan(θ)')
                ax.set_title('Espace de contact J¹(R²)')
        
        plt.tight_layout()
        return fig
