"""
Module geodesics.py - Géodésiques sous-riemanniennes pour l'intégration de contours
"""

import torch
import numpy as np
from typing import Tuple, List, Optional, Callable
import math
from scipy.integrate import solve_bvp
from scipy.optimize import differential_evolution

from .subriemannian_geometry import SubRiemannianGeometry


class SubRiemannianGeodesics:
    """
    Calcul des géodésiques sous-riemanniennes pour modéliser
    l'intégration de contours dans V1.
    """
    
    def __init__(self, spatial_shape: Tuple[int, int], device: str = 'cpu'):
        """
        Args:
            spatial_shape: (height, width) du champ visuel
            device: 'cpu' ou 'cuda'
        """
        self.spatial_shape = spatial_shape
        self.device = device
        
        # Géométrie sous-riemannienne sous-jacente
        self.geometry = SubRiemannianGeometry(spatial_shape, device)
        
        # Cache pour les géodésiques calculées
        self.geodesic_cache = {}
    
    def find_geodesic_between_points(self, 
                                    start_point: torch.Tensor,
                                    end_point: torch.Tensor,
                                    method: str = 'shooting') -> torch.Tensor:
        """
        Trouve la géodésique sous-riemannienne entre deux points.
        
        Args:
            start_point, end_point: Points (x, y, p)
            method: 'shooting', 'variational', ou 'graph'
            
        Returns:
            Géodésique (n_points, 3)
        """
        # Vérifie le cache
        cache_key = (tuple(start_point.tolist()), tuple(end_point.tolist()), method)
        if cache_key in self.geodesic_cache:
            return self.geodesic_cache[cache_key].clone()
        
        if method == 'shooting':
            geodesic = self._shooting_geodesic(start_point, end_point)
        elif method == 'variational':
            geodesic = self._variational_geodesic(start_point, end_point)
        elif method == 'graph':
            geodesic = self._graph_search_geodesic(start_point, end_point)
        else:
            raise ValueError(f"Méthode inconnue: {method}")
        
        # Met en cache
        self.geodesic_cache[cache_key] = geodesic.clone()
        
        return geodesic
    
    def _shooting_geodesic(self, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        """Méthode de shooting pour trouver les géodésiques."""
        try:
            # Convertit en numpy
            start_np = start.cpu().numpy()
            end_np = end.cpu().numpy()
            
            # Utilise la méthode de shooting de la géométrie
            trajectory = self.geometry.shooting_method(start_np, end_np, max_iter=100)
            
            # Convertit en torch
            geodesic = torch.tensor(trajectory[:, :3], device=self.device)
            
            return geodesic
            
        except Exception as e:
            print(f"Shooting method failed, using straight line: {e}")
            # Fallback: ligne droite dans l'espace de contact
            return self._straight_line_fallback(start, end)
    
    def _variational_geodesic(self, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        """Méthode variationnelle (minimisation d'énergie)."""
        n_points = 50
        
        # Initial guess: ligne droite
        t = torch.linspace(0, 1, n_points, device=self.device)
        initial_curve = start + t.unsqueeze(1) * (end - start)
        
        # Fonction coût: énergie sous-riemannienne
        def energy(curve_flat: np.ndarray) -> float:
            curve = torch.tensor(
                curve_flat.reshape(n_points, 3),
                device=self.device
            )
            
            # Fixe les points d'extrémité
            curve[0] = start
            curve[-1] = end
            
            # Énergie + pénalité de lissage
            energy_val = self.geometry.energy_functional(curve)
            
            # Pénalité pour la courbure
            curvature_penalty = self._curvature_penalty(curve)
            
            return energy_val + 0.1 * curvature_penalty
        
        # Optimisation
        initial_flat = initial_curve.cpu().numpy().flatten()
        bounds = [(None, None)] * len(initial_flat)
        
        result = differential_evolution(
            energy,
            bounds,
            maxiter=100,
            popsize=20,
            disp=False
        )
        
        if result.success:
            optimal_flat = result.x
            geodesic = torch.tensor(
                optimal_flat.reshape(n_points, 3),
                device=self.device
            )
            geodesic[0] = start
            geodesic[-1] = end
            
            return geodesic
        else:
            print("Variational method failed, using shooting")
            return self._shooting_geodesic(start, end)
    
    def _graph_search_geodesic(self, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        """Recherche de plus court chemin dans un graphe."""
        # Discrétisation de l'espace
        h, w = self.spatial_shape
        
        # Grille grossière pour la démonstration
        x_samples = torch.linspace(0, w-1, 20, device=self.device)
        y_samples = torch.linspace(0, h-1, 20, device=self.device)
        p_samples = torch.linspace(-10, 10, 10, device=self.device)  # p = tan(θ)
        
        # Construction du graphe (simplifié)
        # En pratique, il faudrait Dijkstra/A* sur la grille
        
        # Pour la démonstration, retourne une interpolation
        return self._straight_line_fallback(start, end)
    
    def _straight_line_fallback(self, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        """Ligne droite dans l'espace de contact (fallback)."""
        n_points = 50
        t = torch.linspace(0, 1, n_points, device=self.device)
        curve = start + t.unsqueeze(1) * (end - start)
        return curve
    
    def _curvature_penalty(self, curve: torch.Tensor) -> float:
        """Pénalité pour la courbure (pour lisser les géodésiques)."""
        if curve.shape[0] < 3:
            return 0.0
        
        penalty = 0.0
        
        for i in range(1, curve.shape[0] - 1):
            prev = curve[i-1]
            curr = curve[i]
            next_p = curve[i+1]
            
            # Courbure discrète
            curvature = torch.norm(next_p - 2*curr + prev)
            penalty += curvature.item()
        
        return penalty
    
    def integrate_contour_geodesically(self, 
                                     seed_points: List[torch.Tensor],
                                     orientation_map: torch.Tensor,
                                     search_radius: float = 20.0) -> List[torch.Tensor]:
        """
        Intègre un contour en trouvant des géodésiques entre points seeds.
        
        Args:
            seed_points: Points de départ (x, y, p)
            orientation_map: Carte d'orientation V1
            search_radius: Rayon de recherche pour connecter les points
            
        Returns:
            Liste de géodésiques formant le contour
        """
        if len(seed_points) < 2:
            return []
        
        # Trie les points par proximité
        sorted_points = self._sort_points_by_proximity(seed_points)
        
        geodesics = []
        
        # Connecte les points consécutifs par des géodésiques
        for i in range(len(sorted_points) - 1):
            start = sorted_points[i]
            end = sorted_points[i + 1]
            
            # Vérifie la distance
            dist = torch.norm(start[:2] - end[:2]).item()
            if dist > search_radius * 2:
                # Points trop éloignés, saute
                continue
            
            # Trouve la géodésique
            geodesic = self.find_geodesic_between_points(start, end, method='shooting')
            
            # Ajuste aux orientations locales
            adjusted_geodesic = self._adjust_to_orientation_map(geodesic, orientation_map)
            
            geodesics.append(adjusted_geodesic)
        
        # Connecte aussi le dernier au premier pour fermer le contour
        if len(sorted_points) > 2:
            start = sorted_points[-1]
            end = sorted_points[0]
            dist = torch.norm(start[:2] - end[:2]).item()
            
            if dist <= search_radius * 2:
                geodesic = self.find_geodesic_between_points(start, end, method='shooting')
                adjusted_geodesic = self._adjust_to_orientation_map(geodesic, orientation_map)
                geodesics.append(adjusted_geodesic)
        
        return geodesics
    
    def _sort_points_by_proximity(self, points: List[torch.Tensor]) -> List[torch.Tensor]:
        """Trie les points par proximité spatiale (voyageur de commerce simplifié)."""
        if not points:
            return []
        
        # Commence par le premier point
        sorted_points = [points[0].clone()]
        remaining = points[1:].copy()
        
        while remaining:
            last = sorted_points[-1]
            
            # Trouve le point le plus proche
            min_dist = float('inf')
            min_idx = 0
            
            for i, point in enumerate(remaining):
                dist = torch.norm(last[:2] - point[:2]).item()
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
            
            # Ajoute le point le plus proche
            sorted_points.append(remaining[min_idx].clone())
            remaining.pop(min_idx)
        
        return sorted_points
    
    def _adjust_to_orientation_map(self, geodesic: torch.Tensor,
                                  orientation_map: torch.Tensor) -> torch.Tensor:
        """
        Ajuste une géodésique pour qu'elle suive la carte d'orientation.
        
        Args:
            geodesic: Géodésique brute
            orientation_map: Carte d'orientation V1
            
        Returns:
            Géodésique ajustée
        """
        adjusted = geodesic.clone()
        
        for i in range(geodesic.shape[0]):
            x, y, p = geodesic[i]
            
            # Convertit en indices
            xi = int(torch.clamp(x, 0, self.spatial_shape[1] - 1))
            yi = int(torch.clamp(y, 0, self.spatial_shape[0] - 1))
            
            # Orientation locale
            local_theta = orientation_map[yi, xi]
            local_p = torch.tan(local_theta)
            
            # Ajuste p (mais garde la position)
            adjusted[i, 2] = local_p
        
        return adjusted
    
    def compute_geodesic_distance_matrix(self, points: List[torch.Tensor]) -> torch.Tensor:
        """
        Calcule la matrice des distances géodésiques entre points.
        
        Args:
            points: Liste de points (x, y, p)
            
        Returns:
            Matrice de distances (n x n)
        """
        n = len(points)
        distance_matrix = torch.zeros((n, n), device=self.device)
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.geometry.compute_carnot_caratheodory_distance(
                    points[i], points[j]
                )
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        return distance_matrix
    
    def find_geodesic_medial_axis(self, boundary_points: List[torch.Tensor],
                                 orientation_map: torch.Tensor) -> torch.Tensor:
        """
        Trouve l'axe médian géodésique (squelette) d'une forme.
        
        Args:
            boundary_points: Points du contour
            orientation_map: Carte d'orientation
            
        Returns:
            Axe médian géodésique
        """
        if len(boundary_points) < 3:
            return torch.stack(boundary_points) if boundary_points else torch.tensor([])
        
        # Matrice de distances
        dist_matrix = self.compute_geodesic_distance_matrix(boundary_points)
        
        # Points "centraux" (plus proches de tous les autres)
        total_distances = dist_matrix.sum(dim=1)
        center_idx = torch.argmin(total_distances)
        
        # Construit l'axe en connectant au centre
        medial_points = []
        
        # Prend un sous-ensemble de points du contour
        n_samples = min(8, len(boundary_points))
        step = len(boundary_points) // n_samples
        
        for i in range(0, len(boundary_points), step):
            if i == center_idx:
                continue
            
            # Géodésique du point du contour vers le centre
            geodesic = self.find_geodesic_between_points(
                boundary_points[i],
                boundary_points[center_idx]
            )
            
            # Ajoute le milieu de la géodésique
            mid_idx = geodesic.shape[0] // 2
            medial_points.append(geodesic[mid_idx])
        
        if medial_points:
            return torch.stack(medial_points)
        else:
            return torch.stack([boundary_points[center_idx]])
    
    def visualize_geodesic_field(self, reference_point: torch.Tensor,
                                n_geodesics: int = 12,
                                length: float = 20.0) -> dict:
        """
        Visualise un champ de géodésiques partant d'un point.
        
        Args:
            reference_point: Point de départ
            n_geodesics: Nombre de géodésiques
            length: Longueur des géodésiques
            
        Returns:
            Dict avec géodésiques et visualisations
        """
        geodesics = []
        
        # Directions radiales
        for i in range(n_geodesics):
            angle = 2 * math.pi * i / n_geodesics
            
            # Point final dans la direction
            dx = length * math.cos(angle)
            dy = length * math.sin(angle)
            
            # Devine une orientation finale
            end_p = math.tan(angle)  # Orientation dans la direction
            
            end_point = torch.tensor([
                reference_point[0] + dx,
                reference_point[1] + dy,
                end_p
            ], device=self.device)
            
            # Trouve la géodésique
            geodesic = self.find_geodesic_between_points(
                reference_point,
                end_point,
                method='shooting'
            )
            
            geodesics.append(geodesic)
        
        return {
            'geodesics': geodesics,
            'reference_point': reference_point,
            'n_geodesics': n_geodesics
        }
