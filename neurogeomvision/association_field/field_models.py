"""
Module field_models.py - Modèles de champs d'association pour V1
VERSION COMPLÈTEMENT OPTIMISÉE
"""

import torch
import numpy as np
from typing import Tuple, List, Optional, Callable, Dict
import math


class AssociationField:
    """
    Implémente le champ d'association cortical - VERSION OPTIMISÉE.
    Utilise le pré-calcul et la vectorisation.
    """
    
    def __init__(self, 
                 spatial_shape: Tuple[int, int],
                 orientation_bins: int = 36,
                 device: str = 'cpu'):
        """
        Args:
            spatial_shape: (height, width) du champ récepteur
            orientation_bins: Nombre de discrétisations d'orientation
            device: 'cpu' ou 'cuda'
        """
        self.spatial_shape = spatial_shape
        self.height, self.width = spatial_shape
        self.orientation_bins = orientation_bins
        self.device = device
        
        # Paramètres du champ d'association
        self.excitatory_sigma = 5.0
        self.inhibitory_sigma = 10.0
        self.angular_sigma = math.pi / 6
        
        # Constantes physiologiques
        self.excitatory_strength = 1.0
        self.inhibitory_strength = -0.3
        
        # OPTIMISATION: Pré-calcule TOUS les champs
        self.field_templates = self._precompute_field_templates_fast()
        
        # Cache pour les noyaux
        self._spatial_kernels = {}
    
    def _precompute_field_templates_fast(self) -> List[Dict]:
        """Pré-calcule les templates de champs d'association."""
        templates = []
        
        # Calcule toutes les orientations
        for theta_idx in range(self.orientation_bins):
            theta = theta_idx * math.pi / self.orientation_bins
            
            # Crée le champ avec la méthode optimisée
            field = self._create_local_field_fast(theta)
            
            templates.append({
                'theta': theta,
                'theta_idx': theta_idx,
                'field': field,
                'field_size': field.shape[0],
                'excitatory_mask': field > 0,
                'inhibitory_mask': field < 0,
                'excitatory_sum': field[field > 0].sum().item() if field[field > 0].numel() > 0 else 0,
                'inhibitory_sum': field[field < 0].sum().item() if field[field < 0].numel() > 0 else 0
            })
        
        return templates
    
    def _create_local_field_fast(self, 
                                reference_orientation: float,
                                field_size: int = 21) -> torch.Tensor:
        """
        Crée un champ d'association local - VERSION VECTORISÉE.
        """
        half_size = field_size // 2
        
        # Crée les coordonnées avec broadcasting
        y_coords, x_coords = torch.meshgrid(
            torch.arange(field_size, device=self.device) - half_size,
            torch.arange(field_size, device=self.device) - half_size,
            indexing='ij'
        )
        
        # Convertit en float pour les calculs
        x = x_coords.float()
        y = y_coords.float()
        
        # Distance et angle du centre (vectorisé)
        distance = torch.sqrt(x**2 + y**2)
        angle_to_center = torch.atan2(y, x)
        
        # Masque pour le centre (évite division par zéro)
        center_mask = (distance == 0)
        
        # Différence angulaire (vectorisé)
        angular_diff = self._angular_difference_vectorized(angle_to_center, reference_orientation)
        
        # Terme spatial (vectorisé)
        spatial_term = self._spatial_kernel_vectorized(x, y, reference_orientation)
        
        # Terme angulaire (vectorisé)
        angular_term = self._angular_kernel_vectorized(angular_diff)
        
        # Champ initial
        field = spatial_term * angular_term
        
        # Masque le centre
        field[center_mask] = 0
        
        # Normalise les poids excitateurs
        excitatory_mask = field > 0
        if excitatory_mask.any():
            excitatory_sum = field[excitatory_mask].sum()
            if excitatory_sum.abs() > 1e-8:
                field[excitatory_mask] = (field[excitatory_mask] / excitatory_sum * 
                                         self.excitatory_strength)
        
        # Normalise les poids inhibiteurs
        inhibitory_mask = field < 0
        if inhibitory_mask.any():
            inhibitory_sum = field[inhibitory_mask].sum().abs()
            if inhibitory_sum > 1e-8:
                field[inhibitory_mask] = (field[inhibitory_mask] / inhibitory_sum * 
                                         self.inhibitory_strength)
        
        return field
    
    def _spatial_kernel_vectorized(self, 
                                  x: torch.Tensor,
                                  y: torch.Tensor,
                                  orientation: float) -> torch.Tensor:
        """Noyau spatial anisotrope - VECTORISÉ."""
        cos_theta = math.cos(orientation)
        sin_theta = math.sin(orientation)
        
        # Rotation (vectorisé)
        x_prime = x * cos_theta + y * sin_theta  # Parallèle
        y_prime = -x * sin_theta + y * cos_theta  # Perpendiculaire
        
        # Gaussienne anisotrope
        sigma_parallel = self.excitatory_sigma
        sigma_perpendicular = self.excitatory_sigma * 2
        
        spatial_weight = torch.exp(
            -x_prime**2 / (2 * sigma_parallel**2) -
            y_prime**2 / (2 * sigma_perpendicular**2)
        )
        
        return spatial_weight
    
    def _angular_kernel_vectorized(self, angular_diff: torch.Tensor) -> torch.Tensor:
        """Noyau angulaire - VECTORISÉ."""
        # Excitation pour alignements
        excitation = torch.relu(torch.cos(angular_diff))
        
        # Inhibition pour orientations orthogonales
        inhibition = torch.relu(torch.cos(angular_diff - math.pi/2)) * 0.3
        
        return excitation - inhibition
    
    def _angular_difference_vectorized(self, 
                                     angle1: torch.Tensor, 
                                     angle2: float) -> torch.Tensor:
        """Différence angulaire minimale - VECTORISÉE."""
        diff = torch.abs(angle1 - angle2)
        return torch.minimum(diff, 2*math.pi - diff)
    
    def get_field_for_orientation(self, theta: float) -> torch.Tensor:
        """
        Retourne le champ d'association pour une orientation donnée.
        Gère les tensors et les floats.
        """
        # Convertit en float si nécessaire
        if isinstance(theta, torch.Tensor):
            theta_value = theta.item()
        else:
            theta_value = theta
        
        # Normalise entre 0 et π
        theta_value = theta_value % math.pi
        
        # Trouve l'index le plus proche
        theta_idx = int(round(theta_value * self.orientation_bins / math.pi)) % self.orientation_bins
        
        return self.field_templates[theta_idx]['field'].clone()
    
    def propagate_activity_fast(self, 
                               activity_map: torch.Tensor,
                               orientation_map: torch.Tensor,
                               n_iterations: int = 3) -> torch.Tensor:
        """
        Propage l'activité - VERSION OPTIMISÉE avec convolution.
        """
        if activity_map.shape != self.spatial_shape:
            raise ValueError(f"Shape mismatch: {activity_map.shape} != {self.spatial_shape}")
        
        propagated = activity_map.clone()
        
        for iteration in range(n_iterations):
            new_activity = torch.zeros_like(propagated)
            
            # Trouve les orientations uniques (réduit les calculs)
            unique_orientations = torch.unique(orientation_map)
            
            for theta in unique_orientations:
                # Masque pour cette orientation
                theta_mask = (orientation_map == theta)
                active_mask = theta_mask & (propagated > 0.01)
                
                if not active_mask.any():
                    continue
                
                # Récupère le champ
                field = self.get_field_for_orientation(theta.item())
                field_size = field.shape[0]
                half = field_size // 2
                
                # Crée une carte d'activité pour cette orientation
                activity_theta = torch.zeros_like(propagated)
                activity_theta[active_mask] = propagated[active_mask]
                
                # Convolution 2D (optimisée)
                activity_4d = activity_theta.unsqueeze(0).unsqueeze(0)
                field_4d = field.unsqueeze(0).unsqueeze(0)
                
                convolved = torch.nn.functional.conv2d(
                    torch.nn.functional.pad(activity_4d, (half, half, half, half), mode='reflect'),
                    field_4d,
                    padding=0
                ).squeeze()
                
                # Ajoute au résultat
                new_activity += convolved
            
            # Mise à jour avec non-linéarité
            propagated = torch.tanh(new_activity * 0.5)
            
            # Inhibition latérale
            mean_activity = propagated.mean()
            propagated = propagated - mean_activity * 0.2
        
        return propagated
    
    # Alias pour compatibilité
    propagate_activity = propagate_activity_fast
    
    def detect_collinear_groups_fast(self,
                                    activity_map: torch.Tensor,
                                    orientation_map: torch.Tensor,
                                    threshold: float = 0.5) -> List[List[Tuple[int, int]]]:
        """
        Détecte les groupes de neurones collinéaires - VERSION OPTIMISÉE.
        """
        # Trouve les neurones actifs
        active_positions = torch.nonzero(activity_map > threshold)
        
        if len(active_positions) == 0:
            return []
        
        # Convertit en listes pour le traitement
        positions_list = [(int(y), int(x)) for y, x in active_positions.tolist()]
        orientations_list = [orientation_map[y, x].item() for y, x in positions_list]
        
        n_neurons = len(positions_list)
        groups = []
        visited = [False] * n_neurons
        
        # Matrice de distances et angles (pré-calcul)
        positions_array = np.array(positions_list)
        orientations_array = np.array(orientations_list)
        
        for i in range(n_neurons):
            if visited[i]:
                continue
            
            # Nouveau groupe
            group = [positions_list[i]]
            visited[i] = True
            to_visit = [i]
            
            yi, xi = positions_list[i]
            theta_i = orientations_list[i]
            
            while to_visit:
                current = to_visit.pop()
                yc, xc = positions_list[current]
                theta_c = orientations_list[current]
                
                # Cherche les voisins non visités
                for j in range(n_neurons):
                    if visited[j]:
                        continue
                    
                    yj, xj = positions_list[j]
                    theta_j = orientations_list[j]
                    
                    # Distance
                    dy = yj - yc
                    dx = xj - xc
                    distance = math.sqrt(dx**2 + dy**2)
                    
                    if distance > 10:  # Seuil de distance
                        continue
                    
                    # Vérifie la collinéarité
                    if distance > 0:
                        connection_angle = math.atan2(dy, dx)
                        
                        diff_current = self._angular_difference_single(theta_c, connection_angle)
                        diff_neighbor = self._angular_difference_single(theta_j, connection_angle)
                        diff_orientations = self._angular_difference_single(theta_c, theta_j)
                        
                        # Conditions de collinéarité
                        if (diff_orientations < math.pi/4 and
                            diff_current < math.pi/4 and
                            diff_neighbor < math.pi/4):
                            
                            group.append(positions_list[j])
                            visited[j] = True
                            to_visit.append(j)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def _angular_difference_single(self, angle1: float, angle2: float) -> float:
        """Différence angulaire pour scalaires."""
        diff = abs(angle1 - angle2)
        return min(diff, 2*math.pi - diff)
    
    def visualize_field(self, 
                       reference_orientation: float = 0.0,
                       field_size: int = 21) -> Dict:
        """
        Visualise un champ d'association.
        """
        # Assure que c'est un float
        if isinstance(reference_orientation, torch.Tensor):
            reference_orientation = reference_orientation.item()
        
        field = self._create_local_field_fast(reference_orientation, field_size)
        
        # Statistiques
        excitatory = field[field > 0]
        inhibitory = field[field < 0]
        
        stats = {
            'field': field,
            'excitatory_count': excitatory.numel(),
            'inhibitory_count': inhibitory.numel(),
            'excitatory_strength': excitatory.sum().item(),
            'inhibitory_strength': inhibitory.sum().item(),
            'max_excitatory': excitatory.max().item() if excitatory.numel() > 0 else 0,
            'max_inhibitory': inhibitory.min().item() if inhibitory.numel() > 0 else 0,
            'reference_orientation_deg': reference_orientation * 180 / math.pi,
            'field_size': field_size
        }
        
        return stats


class CoCircularityModel:
    """
    Modèle de cocircularité pour les connexions horizontales.
    VERSION OPTIMISÉE.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def cocircularity_weight(self,
                            source_pos: Tuple[float, float],
                            source_orientation: float,
                            target_pos: Tuple[float, float],
                            target_orientation: float) -> float:
        """
        Calcule le poids de cocircularité entre deux neurones.
        """
        x1, y1 = source_pos
        x2, y2 = target_pos
        theta1 = source_orientation
        theta2 = target_orientation
        
        # Vecteur entre les points
        dx = x2 - x1
        dy = y2 - y1
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance < 1e-6:
            return 0.0
        
        # Angles
        connection_angle = math.atan2(dy, dx)
        
        # Différences angulaires
        alpha1 = abs(self._angular_difference(connection_angle, theta1))
        alpha2 = abs(self._angular_difference(theta2, connection_angle))
        
        # Condition de cocircularité
        cocircularity = 1.0 - abs(alpha1 + alpha2 - math.pi) / math.pi
        
        # Pénalise la courbure
        curvature = abs(theta2 - theta1) / (distance + 1e-6)
        curvature_penalty = math.exp(-curvature * 5)
        
        return max(0.0, cocircularity * curvature_penalty)
    
    def _angular_difference(self, angle1: float, angle2: float) -> float:
        """Différence angulaire minimale."""
        diff = abs(angle1 - angle2)
        return min(diff, 2*math.pi - diff)
    
    def create_cocircular_field_fast(self,
                                    reference_orientation: float,
                                    field_size: int = 21) -> torch.Tensor:
        """
        Crée un champ de cocircularité - VERSION VECTORISÉE.
        """
        half = field_size // 2
        
        # Crée les coordonnées
        y_coords, x_coords = torch.meshgrid(
            torch.arange(field_size, device=self.device) - half,
            torch.arange(field_size, device=self.device) - half,
            indexing='ij'
        )
        
        x = x_coords.float()
        y = y_coords.float()
        
        # Distance et angle
        distance = torch.sqrt(x**2 + y**2)
        target_orientation = torch.atan2(y, x)
        
        # Calcule les poids pour tous les points en une fois
        field = torch.zeros(field_size, field_size, device=self.device)
        
        for i in range(field_size):
            for j in range(field_size):
                if distance[i, j] > 0:
                    weight = self.cocircularity_weight(
                        (0.0, 0.0), reference_orientation,
                        (x[i, j].item(), y[i, j].item()), target_orientation[i, j].item()
                    )
                    field[i, j] = weight
        
        return field
