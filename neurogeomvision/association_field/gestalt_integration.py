"""
Module gestalt_integration.py - Intégration gestaltiste via la connectivité corticale
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
import math
from scipy import ndimage


class GestaltIntegration:
    """
    Implémente les principes gestaltistes via la connectivité corticale.
    
    Principes:
    1. Proximité
    2. Similarité (orientation)
    3. Bonne continuation
    4. Clôture
    5. Symétrie
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def proximity_grouping(self,
                         positions: List[Tuple[float, float]],
                         max_distance: float = 10.0) -> List[List[int]]:
        """
        Regroupement par proximité (loi de la proximité).
        
        Args:
            positions: Liste de positions
            max_distance: Distance maximale pour le regroupement
            
        Returns:
            Groupes d'indices
        """
        n = len(positions)
        if n == 0:
            return []
        
        # Matrice de distances
        groups = []
        visited = [False] * n
        
        for i in range(n):
            if visited[i]:
                continue
            
            # Nouveau groupe
            group = [i]
            visited[i] = True
            queue = [i]
            
            while queue:
                current = queue.pop(0)
                
                for j in range(n):
                    if not visited[j]:
                        # Distance
                        xi, yi = positions[i]
                        xj, yj = positions[j]
                        distance = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                        
                        if distance <= max_distance:
                            group.append(j)
                            visited[j] = True
                            queue.append(j)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def similarity_grouping(self,
                          orientations: List[float],
                          positions: List[Tuple[float, float]],
                          angular_threshold: float = math.pi/6) -> List[List[int]]:
        """
        Regroupement par similarité d'orientation.
        
        Args:
            orientations: Liste d'orientations
            positions: Liste de positions
            angular_threshold: Seuil angulaire
            
        Returns:
            Groupes d'indices
        """
        n = len(orientations)
        if n == 0:
            return []
        
        groups = []
        visited = [False] * n
        
        for i in range(n):
            if visited[i]:
                continue
            
            group = [i]
            visited[i] = True
            queue = [i]
            
            while queue:
                current = queue.pop(0)
                current_orientation = orientations[current]
                
                for j in range(n):
                    if not visited[j]:
                        # Similarité d'orientation
                        orientation_diff = abs(self._angular_difference(
                            current_orientation, orientations[j]
                        ))
                        
                        if orientation_diff <= angular_threshold:
                            # Vérifie aussi la proximité
                            xi, yi = positions[i]
                            xj, yj = positions[j]
                            distance = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                            
                            if distance <= 15.0:  # Proximité spatiale aussi
                                group.append(j)
                                visited[j] = True
                                queue.append(j)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def good_continuation_grouping(self,
                                 positions: List[Tuple[float, float]],
                                 orientations: List[float],
                                 curvature_threshold: float = 0.1) -> List[List[int]]:
        """
        Regroupement par bonne continuation (collinéarité/cocircularité).
        
        Args:
            positions: Liste de positions
            orientations: Liste d'orientations
            curvature_threshold: Seuil de courbure
            
        Returns:
            Groupes d'indices
        """
        from .field_models import CoCircularityModel
        
        cocircular = CoCircularityModel(self.device)
        n = len(positions)
        
        if n < 2:
            return []
        
        # Matrice de cocircularité
        cocircularity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                weight = cocircular.cocircularity_weight(
                    positions[i], orientations[i],
                    positions[j], orientations[j]
                )
                cocircularity_matrix[i, j] = weight
                cocircularity_matrix[j, i] = weight
        
        # Groupement basé sur la cocircularité
        groups = []
        visited = [False] * n
        
        for i in range(n):
            if visited[i]:
                continue
            
            group = [i]
            visited[i] = True
            
            # Trouve les voisins cocirculaires
            for j in range(n):
                if not visited[j] and cocircularity_matrix[i, j] > 0.7:
                    # Vérifie la courbure du groupe élargi
                    if self._check_group_curvature([i, j], positions, orientations, curvature_threshold):
                        group.append(j)
                        visited[j] = True
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def _check_group_curvature(self,
                             group_indices: List[int],
                             positions: List[Tuple[float, float]],
                             orientations: List[float],
                             max_curvature: float) -> bool:
        """Vérifie si un groupe a une courbure acceptable."""
        if len(group_indices) < 3:
            return True
        
        # Calcule la courbure moyenne
        curvatures = []
        
        for i in range(len(group_indices) - 2):
            idx1, idx2, idx3 = group_indices[i], group_indices[i+1], group_indices[i+2]
            
            p1 = positions[idx1]
            p2 = positions[idx2]
            p3 = positions[idx3]
            
            # Courbure discrète
            curvature = self._discrete_curvature(p1, p2, p3)
            curvatures.append(abs(curvature))
        
        if curvatures:
            avg_curvature = np.mean(curvatures)
            return avg_curvature <= max_curvature
        
        return True
    
    def _discrete_curvature(self,
                           p1: Tuple[float, float],
                           p2: Tuple[float, float],
                           p3: Tuple[float, float]) -> float:
        """Calcule la courbure discrète pour trois points."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        
        # Vecteurs
        v1 = np.array([x2 - x1, y2 - y1])
        v2 = np.array([x3 - x2, y3 - y2])
        
        # Normes
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        
        if n1 == 0 or n2 == 0:
            return 0.0
        
        # Angle entre les vecteurs
        cos_angle = np.dot(v1, v2) / (n1 * n2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = math.acos(cos_angle)
        
        # Courbure = angle / longueur moyenne
        curvature = angle / ((n1 + n2) / 2 + 1e-6)
        
        return curvature
    
    def closure_grouping(self,
                       positions: List[Tuple[float, float]],
                       orientations: List[float],
                       closure_threshold: float = 0.8) -> List[List[int]]:
        """
        Regroupement par clôture (tendance à compléter les formes fermées).
        OPTIMISÉ : Limité à 50 points maximum.
        """
        n = len(positions)
        if n < 3 or n > 50:  # LIMITE STRICTE
            return []
        
        groups = []
        
        # Échantillonnage si trop grand
        if n > 20:
            indices = np.random.choice(n, 20, replace=False)
            positions = [positions[i] for i in indices]
            orientations = [orientations[i] for i in indices]
            n = 20
        
        # Recherche de triangles qui pourraient fermer une forme
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    # Vérifie si les points forment un triangle
                    p1, p2, p3 = positions[i], positions[j], positions[k]
                    
                    # Distances
                    d12 = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                    d23 = math.sqrt((p2[0]-p3[0])**2 + (p2[1]-p3[1])**2)
                    d31 = math.sqrt((p3[0]-p1[0])**2 + (p3[1]-p1[1])**2)
                    
                    # Périmètre
                    perimeter = d12 + d23 + d31
                    
                    # Aire (formule de Héron) avec protection numérique
                    s = perimeter / 2
                    radicand = s * (s-d12) * (s-d23) * (s-d31)
                    
                    # Protection contre les erreurs numériques
                    if radicand <= 0:
                        continue
                    
                    area = math.sqrt(radicand)
                    
                    # Ratio d'isopérimétrie (mesure de circularité)
                    if perimeter > 0:
                        circularity = 4 * math.pi * area / (perimeter**2)
                        
                        if circularity > closure_threshold:
                            # Vérifie les orientations
                            o1, o2, o3 = orientations[i], orientations[j], orientations[k]
                            
                            # Les orientations devraient pointer vers l'intérieur
                            if self._orientations_point_inward([p1, p2, p3], [o1, o2, o3]):
                                groups.append([i, j, k])
        
        return groups
        
            
    def _orientations_point_inward(self,
                                 positions: List[Tuple[float, float]],
                                 orientations: List[float]) -> bool:
        """Vérifie si les orientations pointent vers l'intérieur du polygone."""
        if len(positions) < 3:
            return True
        
        # Centre de masse
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)
        
        for (x, y), theta in zip(positions, orientations):
            # Vecteur orientation
            ox = math.cos(theta)
            oy = math.sin(theta)
            
            # Vecteur vers le centre
            dx = cx - x
            dy = cy - y
            
            # Normalise
            norm = math.sqrt(dx**2 + dy**2)
            if norm > 0:
                dx /= norm
                dy /= norm
                
                # Produit scalaire (doit être positif pour pointer vers le centre)
                dot = ox*dx + oy*dy
                if dot < 0.5:  # Ne pointe pas suffisamment vers le centre
                    return False
        
        return True
    
    def symmetry_grouping(self,
                        positions: List[Tuple[float, float]],
                        orientations: List[float]) -> List[List[int]]:
        """
        Regroupement par symétrie.
        
        Args:
            positions: Positions
            orientations: Orientations
            
        Returns:
            Groupes symétriques
        """
        n = len(positions)
        if n < 2:
            return []
        
        groups = []
        visited = [False] * n
        
        # Pour chaque paire, cherche un axe de symétrie
        for i in range(n):
            if visited[i]:
                continue
            
            for j in range(i+1, n):
                if visited[j]:
                    continue
                
                # Axe de symétrie potentiel
                p1 = positions[i]
                p2 = positions[j]
                
                # Milieu
                mx = (p1[0] + p2[0]) / 2
                my = (p1[1] + p2[1]) / 2
                
                # Vecteur entre les points
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                
                # Axe perpendiculaire
                if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                    continue
                
                # Normalise
                norm = math.sqrt(dx**2 + dy**2)
                dx /= norm
                dy /= norm
                
                # Vecteur perpendiculaire (axe de symétrie)
                axis_x = -dy
                axis_y = dx
                
                # Vérifie les symétries
                symmetric_group = [i, j]
                
                for k in range(n):
                    if k == i or k == j or visited[k]:
                        continue
                    
                    p3 = positions[k]
                    
                    # Réflexion par rapport à l'axe
                    reflected = self._reflect_point(p3, (mx, my), (axis_x, axis_y))
                    
                    # Cherche le point le plus proche du reflet
                    min_dist = float('inf')
                    min_idx = -1
                    
                    for l in range(n):
                        if l in symmetric_group or visited[l]:
                            continue
                        
                        dist = math.sqrt(
                            (reflected[0] - positions[l][0])**2 +
                            (reflected[1] - positions[l][1])**2
                        )
                        
                        if dist < min_dist and dist < 5.0:  # Seuil de 5 pixels
                            min_dist = dist
                            min_idx = l
                    
                    if min_idx != -1:
                        symmetric_group.append(min_idx)
                
                if len(symmetric_group) > 2:
                    groups.append(symmetric_group)
                    for idx in symmetric_group:
                        visited[idx] = True
                    break
        
        return groups
    
    def _reflect_point(self,
                      point: Tuple[float, float],
                      line_point: Tuple[float, float],
                      line_dir: Tuple[float, float]) -> Tuple[float, float]:
        """Réfléchit un point par rapport à une ligne."""
        x, y = point
        x0, y0 = line_point
        dx, dy = line_dir
        
        # Projection sur la ligne
        t = ((x - x0)*dx + (y - y0)*dy) / (dx**2 + dy**2 + 1e-6)
        
        # Point projeté
        px = x0 + t*dx
        py = y0 + t*dy
        
        # Réflexion
        rx = 2*px - x
        ry = 2*py - y
        
        return (rx, ry)
    
    def integrate_gestalt_principles(self,
                                   positions: List[Tuple[float, float]],
                                   orientations: List[float],
                                   activities: Optional[List[float]] = None) -> Dict:
        """
        Intègre tous les principes gestaltistes.
        
        Args:
            positions: Positions
            orientations: Orientations
            activities: Activités (optionnel)
            
        Returns:
            Groupements selon tous les principes
        """
        if activities is None:
            activities = [1.0] * len(positions)
        
        # Applique tous les principes
        proximity_groups = self.proximity_grouping(positions)
        similarity_groups = self.similarity_grouping(orientations, positions)
        continuation_groups = self.good_continuation_grouping(positions, orientations)
        closure_groups = self.closure_grouping(positions, orientations)
        symmetry_groups = self.symmetry_grouping(positions, orientations)
        
        # Fusionne les groupes
        all_groups = []
        all_groups.extend(proximity_groups)
        all_groups.extend(similarity_groups)
        all_groups.extend(continuation_groups)
        all_groups.extend(closure_groups)
        all_groups.extend(symmetry_groups)
        
        # Nettoie les doublons
        cleaned_groups = self._clean_overlapping_groups(all_groups)
        
        # Scores des groupes
        group_scores = []
        for group in cleaned_groups:
            score = self._compute_group_score(group, positions, orientations, activities)
            group_scores.append(score)
        
        return {
            'proximity_groups': proximity_groups,
            'similarity_groups': similarity_groups,
            'continuation_groups': continuation_groups,
            'closure_groups': closure_groups,
            'symmetry_groups': symmetry_groups,
            'integrated_groups': cleaned_groups,
            'group_scores': group_scores
        }
    
    def _clean_overlapping_groups(self, groups: List[List[int]]) -> List[List[int]]:
        """Nettoie les groupes qui se chevauchent."""
        if not groups:
            return []
        
        # Trie par taille décroissante
        groups_sorted = sorted(groups, key=lambda g: len(g), reverse=True)
        
        cleaned = []
        used_indices = set()
        
        for group in groups_sorted:
            # Vérifie le chevauchement
            overlap = used_indices.intersection(set(group))
            
            if len(overlap) / len(group) < 0.5:  # Moins de 50% de chevauchement
                cleaned.append(group)
                used_indices.update(group)
        
        return cleaned
    
    def _compute_group_score(self,
                           group: List[int],
                           positions: List[Tuple[float, float]],
                           orientations: List[float],
                           activities: List[float]) -> float:
        """Calcule un score de qualité pour un groupe."""
        if len(group) < 2:
            return 0.0
        
        # Score basé sur:
        # 1. Taille du groupe
        size_score = min(len(group) / 10.0, 1.0)
        
        # 2. Activité moyenne
        activity_score = np.mean([activities[i] for i in group])
        
        # 3. Cohérence d'orientation
        group_orientations = [orientations[i] for i in group]
        orientation_variance = np.var(group_orientations)
        orientation_score = math.exp(-orientation_variance / (math.pi/4)**2)
        
        # 4. Compacité spatiale
        group_positions = [positions[i] for i in group]
        centroid = np.mean(group_positions, axis=0)
        distances = [math.sqrt((p[0]-centroid[0])**2 + (p[1]-centroid[1])**2) 
                    for p in group_positions]
        compactness_score = 1.0 / (np.mean(distances) + 1.0)
        
        # Score total
        total_score = (size_score * 0.2 +
                      activity_score * 0.3 +
                      orientation_score * 0.3 +
                      compactness_score * 0.2)
        
        return total_score
    
    def _angular_difference(self, angle1: float, angle2: float) -> float:
        """Différence angulaire minimale."""
        diff = abs(angle1 - angle2)
        return min(diff, 2*math.pi - diff)
