"""
Module cortical_connectivity.py - Connectivité corticale de V1
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
import math
import networkx as nx


class CorticalConnectivity:
    """
    Modèle complet de connectivité corticale pour V1.
    Combine champ d'association, inhibition latérale, et feedback.
    """
    
    def __init__(self, 
                 spatial_shape: Tuple[int, int],
                 orientation_bins: int = 36,
                 device: str = 'cpu'):
        """
        Args:
            spatial_shape: (height, width)
            orientation_bins: Nombre d'orientations discrétisées
            device: 'cpu' ou 'cuda'
        """
        self.spatial_shape = spatial_shape
        self.height, self.width = spatial_shape
        self.orientation_bins = orientation_bins
        self.device = device
        
        # Modules de connectivité
        from .field_models import AssociationField, CoCircularityModel
        
        self.association_field = AssociationField(
            spatial_shape, orientation_bins, device
        )
        self.cocircularity_model = CoCircularityModel(device)
        
        # Paramètres de connectivité
        self.connection_threshold = 0.1
        self.max_connection_distance = 20.0
        
        # Graphe de connectivité
        self.connectivity_graph = None
        
    def build_cortical_graph(self,
                            activity_map: torch.Tensor,
                            orientation_map: torch.Tensor,
                            threshold: float = 0.3,
                            max_neurons: int = 500) -> nx.Graph:
        """
        Construit un graphe de connectivité corticale OPTIMISÉ.
        
        Args:
            activity_map: Carte d'activité
            orientation_map: Carte d'orientation
            threshold: Seuil d'activité
            max_neurons: Nombre maximum de neurones à traiter
            
        Returns:
            Graphe NetworkX des connexions corticales
        """
        G = nx.Graph()
        
        # Trouve les neurones actifs
        active_positions = torch.nonzero(activity_map > threshold)
        
        # LIMITE le nombre de neurones
        if len(active_positions) > max_neurons:
            print(f"  Limitation: {len(active_positions)} → {max_neurons} neurones")
            # Sélection aléatoire uniforme
            indices = torch.randperm(len(active_positions))[:max_neurons]
            active_positions = active_positions[indices]
        
        n_neurons = len(active_positions)
        print(f"  Construction graphe avec {n_neurons} neurones...")
        
        # Pré-calcule les données
        node_data = []
        for pos in active_positions:
            y, x = pos.tolist()
            node_id = f"{y}_{x}"
            
            node_data.append({
                'id': node_id,
                'pos': (x, y),
                'orientation': orientation_map[y, x].item(),
                'activity': activity_map[y, x].item()
            })
        
        # Ajoute les nœuds
        for data in node_data:
            G.add_node(data['id'], **data)
        
        # OPTIMISATION: Distance maximum réduite
        spatial_limit = min(15, max(self.height, self.width) // 4)
        
        # Construit un kd-tree pour recherche spatiale rapide
        from scipy.spatial import KDTree
        positions_array = np.array([data['pos'] for data in node_data])
        tree = KDTree(positions_array)
        
        # Cherche les voisins dans un rayon
        for i, data_i in enumerate(node_data):
            # Recherche des voisins proches
            neighbors = tree.query_ball_point(positions_array[i], spatial_limit)
            
            for j in neighbors:
                if i == j:
                    continue
                
                data_j = node_data[j]
                
                # Distance déjà vérifiée par KDTree
                pos1 = data_i['pos']
                pos2 = data_j['pos']
                orientation1 = data_i['orientation']
                orientation2 = data_j['orientation']
                
                # Poids de connexion
                weight = self.compute_connection_weight(
                    pos1, orientation1,
                    pos2, orientation2
                )
                
                if weight > self.connection_threshold:
                    G.add_edge(data_i['id'], data_j['id'], 
                              weight=weight,
                              distance=np.linalg.norm(np.array(pos1) - np.array(pos2)))
        
        print(f"  Graphe construit: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")
        return G
        
            
    def compute_connection_weight(self,
                                 pos1: Tuple[float, float],
                                 orientation1: float,
                                 pos2: Tuple[float, float],
                                 orientation2: float) -> float:
        """
        Calcule le poids d'une connexion entre deux neurones (OPTIMISÉ).
        
        Args:
            pos1, pos2: Positions (x, y)
            orientation1, orientation2: Orientations en radians
            
        Returns:
            Poids de connexion (0 à 1)
        """
        x1, y1 = pos1
        x2, y2 = pos2
        
        # Distance rapide
        dx = x2 - x1
        dy = y2 - y1
        distance_sq = dx*dx + dy*dy
        
        # Distance maximum rapide
        if distance_sq > 400:  # 20²
            return 0.0
        
        distance = math.sqrt(distance_sq)
        
        # 1. Similarité d'orientation (la plus rapide)
        orientation_similarity = 0.5 * (math.cos(2 * (orientation1 - orientation2)) + 1)
        
        # 2. Alignement avec la connexion
        if distance > 1e-6:
            connection_angle = math.atan2(dy, dx)
            
            # Différence avec les orientations
            diff1 = self._angular_difference(connection_angle, orientation1)
            diff2 = self._angular_difference(orientation2, connection_angle)
            
            alignment = (math.cos(diff1) + math.cos(diff2)) / 2
        else:
            alignment = 0.0
        
        # 3. Décroissance spatiale
        spatial_decay = math.exp(-distance_sq / (2 * 50))  # sigma² = 50
        
        # Moyenne pondérée (poids réduits pour l'orientation)
        total_weight = (
            0.4 * orientation_similarity +
            0.4 * max(0, alignment) +
            0.2 * spatial_decay
        )
        
        return total_weight
        
            
    def _field_weight(self,
                     pos1: Tuple[int, int],
                     orientation1: float,
                     pos2: Tuple[int, int],
                     orientation2: float) -> float:
        """Poids basé sur le champ d'association classique."""
        y1, x1 = pos1
        y2, x2 = pos2
        
        # Vecteur de connexion
        dx = x2 - x1
        dy = y2 - y1
        connection_angle = math.atan2(dy, dx)
        
        # Différence avec l'orientation source
        diff_source = abs(self._angular_difference(connection_angle, orientation1))
        
        # Pour une connexion collinéaire, la cible devrait être alignée
        # avec le vecteur de connexion
        diff_target = abs(self._angular_difference(orientation2, connection_angle))
        
        # Poids basé sur l'alignement
        weight = (math.cos(diff_source) + math.cos(diff_target)) / 2
        
        return max(0, weight)
    
    def _angular_difference(self, angle1: float, angle2: float) -> float:
        """Différence angulaire minimale."""
        diff = abs(angle1 - angle2)
        return min(diff, 2*math.pi - diff)
    
    def propagate_activity_dynamically(self,
                                      initial_activity: torch.Tensor,
                                      orientation_map: torch.Tensor,
                                      time_steps: int = 10,
                                      dt: float = 0.1) -> torch.Tensor:
        """
        Propage dynamiquement l'activité à travers le réseau cortical.
        
        Args:
            initial_activity: Activité initiale
            orientation_map: Carte d'orientation
            time_steps: Nombre de pas de temps
            dt: Pas de temps
            
        Returns:
            Évolution temporelle de l'activité
        """
        activity = initial_activity.clone()
        activity_history = [activity.clone()]
        
        # Constantes de temps
        tau_excitation = 10.0  # ms
        tau_inhibition = 20.0  # ms
        
        alpha_exc = math.exp(-dt / tau_excitation)
        alpha_inh = math.exp(-dt / tau_inhibition)
        
        for t in range(time_steps):
            # Convolution avec le champ d'association
            convolved = torch.zeros_like(activity)
            
            for y in range(self.height):
                for x in range(self.width):
                    if activity[y, x] > 0:
                        theta = orientation_map[y, x]
                        field = self.association_field.get_field_for_orientation(theta)
                        
                        field_half = field.shape[0] // 2
                        
                        for fy in range(field.shape[0]):
                            for fx in range(field.shape[1]):
                                ty = y + fy - field_half
                                tx = x + fx - field_half
                                
                                if (0 <= ty < self.height and 
                                    0 <= tx < self.width):
                                    
                                    weight = field[fy, fx]
                                    convolved[ty, tx] += activity[y, x] * weight
            
            # Équation différentielle
            # dA/dt = -A/tau + excitation - inhibition
            
            excitation = convolved
            inhibition = torch.nn.functional.conv2d(
                activity.unsqueeze(0).unsqueeze(0),
                torch.ones(1, 1, 5, 5, device=self.device) / 25,
                padding=2
            ).squeeze()
            
            # Mise à jour
            activity = (alpha_exc * activity + 
                       (1 - alpha_exc) * excitation -
                       (1 - alpha_inh) * inhibition * 0.5)
            
            # Non-linéarité (seuil)
            activity = torch.sigmoid(activity) * 2 - 1
            
            activity_history.append(activity.clone())
        
        return torch.stack(activity_history)
    
    def find_contours_via_connectivity(self,
                                      activity_map: torch.Tensor,
                                      orientation_map: torch.Tensor,
                                      min_contour_length: int = 5) -> List[List[Tuple[int, int]]]:
        """
        Trouve les contours en utilisant la connectivité corticale.
        
        Args:
            activity_map: Carte d'activité
            orientation_map: Carte d'orientation
            min_contour_length: Longueur minimale des contours
            
        Returns:
            Liste des contours (listes de positions)
        """
        # Construit le graphe
        G = self.build_cortical_graph(activity_map, orientation_map)
        
        if G.number_of_nodes() == 0:
            return []
        
        # Trouve les composantes connexes
        contours = []
        
        for component in nx.connected_components(G):
            if len(component) < min_contour_length:
                continue
            
            # Extrait les positions
            positions = []
            for node_id in component:
                node_data = G.nodes[node_id]
                positions.append((node_data['pos'][1], node_data['pos'][0]))  # (y, x)
            
            # Trie par position pour avoir un ordre
            positions.sort(key=lambda p: (p[0], p[1]))
            
            contours.append(positions)
        
        return contours
    
    def compute_connectivity_statistics(self,
                                       activity_map: torch.Tensor,
                                       orientation_map: torch.Tensor) -> Dict:
        """
        Calcule des statistiques sur la connectivité.
        
        Args:
            activity_map: Carte d'activité
            orientation_map: Carte d'orientation
            
        Returns:
            Statistiques de connectivité
        """
        G = self.build_cortical_graph(activity_map, orientation_map)
        
        stats = {
            'n_neurons': G.number_of_nodes(),
            'n_connections': G.number_of_edges(),
            'connection_density': G.number_of_edges() / max(1, G.number_of_nodes()),
            'average_degree': sum(dict(G.degree()).values()) / max(1, G.number_of_nodes()),
            'clustering_coefficient': nx.average_clustering(G) if G.number_of_nodes() > 0 else 0,
            'connected_components': nx.number_connected_components(G),
            'largest_component_size': max([len(c) for c in nx.connected_components(G)], default=0)
        }
        
        # Statistiques sur les poids
        if G.number_of_edges() > 0:
            weights = [data['weight'] for _, _, data in G.edges(data=True)]
            stats.update({
                'avg_weight': np.mean(weights),
                'std_weight': np.std(weights),
                'max_weight': np.max(weights),
                'min_weight': np.min(weights)
            })
        
        return stats
    
    def visualize_connectivity(self,
                              activity_map: torch.Tensor,
                              orientation_map: torch.Tensor,
                              threshold: float = 0.3) -> Dict:
        """
        Prépare les données pour la visualisation de la connectivité.
        
        Args:
            activity_map: Carte d'activité
            orientation_map: Carte d'orientation
            threshold: Seuil d'activité
            
        Returns:
            Données de visualisation
        """
        G = self.build_cortical_graph(activity_map, orientation_map, threshold)
        
        # Extrait les données pour la visualisation
        nodes_data = []
        edges_data = []
        
        for node_id, node_data in G.nodes(data=True):
            nodes_data.append({
                'id': node_id,
                'x': node_data['pos'][0],
                'y': node_data['pos'][1],
                'orientation': node_data['orientation'],
                'activity': node_data['activity']
            })
        
        for node1_id, node2_id, edge_data in G.edges(data=True):
            node1_data = G.nodes[node1_id]
            node2_data = G.nodes[node2_id]
            
            edges_data.append({
                'source': node1_id,
                'target': node2_id,
                'weight': edge_data['weight'],
                'distance': edge_data['distance'],
                'source_pos': node1_data['pos'],
                'target_pos': node2_data['pos']
            })
        
        return {
            'nodes': nodes_data,
            'edges': edges_data,
            'graph': G
        }
