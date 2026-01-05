"""
Module ganglion_cells.py - Cellules ganglionnaires de la rétine
Transformation des signaux rétiniens en potentiels d'action
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import math


class GanglionCell(nn.Module):
    """
    Classe de base pour les cellules ganglionnaires.
    Transforme les signaux analogiques en trains de spikes.
    """
    
    def __init__(self,
                 cell_type: str = 'parasol',  # 'parasol', 'midget', 'bistratified'
                 receptive_field_size: int = 10,
                 temporal_filter_tau: float = 20.0,
                 spike_threshold: float = 0.5,
                 refractory_period: float = 5.0,
                 adaptation_strength: float = 0.1,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.cell_type = cell_type
        self.receptive_field_size = receptive_field_size
        self.temporal_filter_tau = temporal_filter_tau
        self.spike_threshold = spike_threshold
        self.refractory_period = refractory_period
        self.adaptation_strength = adaptation_strength
        self.device = device
        
        # Filtre spatial (centre-surround amélioré)
        self.spatial_filter = self._create_spatial_filter()
        
        # Filtre temporel
        self.register_buffer('temporal_state', torch.tensor(0.0, device=device))
        
        # Adaptation et période réfractaire
        self.register_buffer('adaptation_state', torch.tensor(0.0, device=device))
        self.register_buffer('refractory_counter', torch.tensor(0.0, device=device))
        
        # Dernier spike
        self.register_buffer('last_spike_time', torch.tensor(-1000.0, device=device))
        
        # Caractéristiques par type de cellule
        self._setup_cell_type()
    
    def _setup_cell_type(self):
        """Configure les paramètres selon le type de cellule."""
        type_params = {
            'parasol': {  # Cellules M (magnocellulaires)
                'receptive_field': 15,
                'temporal_tau': 15.0,
                'spike_threshold': 0.4,
                'contrast_gain': 1.2
            },
            'midget': {  # Cellules P (parvocellulaires)
                'receptive_field': 8,
                'temporal_tau': 30.0,
                'spike_threshold': 0.6,
                'contrast_gain': 0.8
            },
            'bistratified': {  # Cellules bistratifiées (S-cone)
                'receptive_field': 12,
                'temporal_tau': 25.0,
                'spike_threshold': 0.5,
                'contrast_gain': 1.0
            }
        }
        
        params = type_params.get(self.cell_type, type_params['parasol'])
        self.receptive_field_size = params['receptive_field']
        self.temporal_filter_tau = params['temporal_tau']
        self.spike_threshold = params['spike_threshold']
        self.contrast_gain = params['contrast_gain']
        
        # Recréer le filtre spatial avec les nouveaux paramètres
        self.spatial_filter = self._create_spatial_filter()
    
    def _create_spatial_filter(self) -> torch.Tensor:
        """Crée un filtre spatial DoG (Difference of Gaussians)."""
        size = self.receptive_field_size
        center = size // 2
        
        # Grille
        y, x = torch.meshgrid(
            torch.arange(size, device=self.device) - center,
            torch.arange(size, device=self.device) - center,
            indexing='ij'
        )
        
        r = torch.sqrt(x**2 + y**2)
        
        # Centre et surround
        sigma_center = self.receptive_field_size / 4.0
        sigma_surround = sigma_center * 2.5
        
        center_gauss = torch.exp(-r**2 / (2 * sigma_center**2))
        surround_gauss = torch.exp(-r**2 / (2 * sigma_surround**2))
        
        # Normaliser
        center_gauss = center_gauss / center_gauss.sum()
        surround_gauss = surround_gauss / surround_gauss.sum()
        
        # DoG avec rapport centre/surround spécifique au type
        if self.cell_type == 'parasol':
            surround_strength = 0.8  # Fort surround pour mouvement
        elif self.cell_type == 'midget':
            surround_strength = 0.5  # Surround modéré pour couleur
        else:
            surround_strength = 0.6
        
        dog_filter = center_gauss - surround_strength * surround_gauss
        
        return dog_filter.unsqueeze(0).unsqueeze(0)
    
    def reset_state(self):
        """Réinitialise l'état."""
        self.temporal_state = torch.tensor(0.0, device=self.device)
        self.adaptation_state = torch.tensor(0.0, device=self.device)
        self.refractory_counter = torch.tensor(0.0, device=self.device)
        self.last_spike_time = torch.tensor(-1000.0, device=self.device)
    
    def temporal_filter(self, input_signal: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """Filtre temporel."""
        alpha = math.exp(-dt / self.temporal_filter_tau)
        self.temporal_state = alpha * self.temporal_state + (1 - alpha) * input_signal
        return self.temporal_state
    
    def spatial_filtering(self, input_map: torch.Tensor) -> torch.Tensor:
        """Filtrage spatial."""
        if len(input_map.shape) == 2:
            input_map = input_map.unsqueeze(0)  # (1, H, W)
        
        height, width = input_map.shape[-2:]
        
        if height >= self.receptive_field_size and width >= self.receptive_field_size:
            filtered = F.conv2d(
                input_map.unsqueeze(1),  # (B, 1, H, W)
                self.spatial_filter,
                padding=self.receptive_field_size // 2
            ).squeeze(1)
        else:
            filtered = input_map
        
        return filtered
    
    def spike_generation(self, membrane_potential: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """
        Génère des spikes à partir du potentiel membranaire.
        
        Args:
            membrane_potential: Potentiel d'entrée
            dt: Pas de temps
            
        Returns:
            Tensor de spikes binaires
        """
        # Adaptation
        self.adaptation_state *= math.exp(-dt / 100.0)  # Décroissance lente
        
        # Période réfractaire
        if self.refractory_counter > 0:
            self.refractory_counter -= dt
            spikes = torch.zeros_like(membrane_potential)
        else:
            # Potentiel avec adaptation
            effective_potential = membrane_potential - self.adaptation_state
            
            # Génération de spikes
            spikes = (effective_potential > self.spike_threshold).float()
            
            # Mise à jour de l'adaptation
            spike_count = spikes.sum().item()
            if spike_count > 0:
                self.adaptation_state += self.adaptation_strength * spike_count
                self.refractory_counter = self.refractory_period
                self.last_spike_time = torch.tensor(0.0, device=self.device)  # Réinitialiser
        
        return spikes
    
    def forward(self,
                bipolar_input: torch.Tensor,
                dt: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Traitement complet de la cellule ganglionnaire.
    
        Args:
            bipolar_input: Entrée des cellules bipolaires (B, H, W) ou (H, W)
            dt: Pas de temps
        
        Returns:
            spikes: Spikes de sortie
            membrane_potential: Potentiel membranaire
        """
        # S'assurer que c'est 2D (B, H, W) ou (H, W)
        if len(bipolar_input.shape) == 2:
            bipolar_input = bipolar_input.unsqueeze(0)  # (1, H, W)
        elif len(bipolar_input.shape) == 4:
            # (B, 1, H, W) -> (B, H, W)
            bipolar_input = bipolar_input.squeeze(1)
        
        batch_size, input_height, input_width = bipolar_input.shape
    
        # Filtrage spatial
        if input_height >= self.receptive_field_size and input_width >= self.receptive_field_size:
            # Préparer pour convolution
            bipolar_input_4d = bipolar_input.unsqueeze(1)  # (B, 1, H, W)
            spatial_response = F.conv2d(
                bipolar_input_4d,
                self.spatial_filter,
                padding=self.receptive_field_size // 2
            ).squeeze(1)  # (B, H, W)
        else:
            spatial_response = bipolar_input
        
        # Gain de contraste
        spatial_response = spatial_response * self.contrast_gain
        
        # Filtrage temporel
        membrane_potential = self.temporal_filter(spatial_response, dt)
        
        # Non-linéarité (rectification)
        membrane_potential = torch.relu(membrane_potential)
        
        # Génération de spikes
        spikes = self.spike_generation(membrane_potential, dt)
        
        return spikes, membrane_potential


class ONGanglionCell(GanglionCell):
    """
    Cellule ganglionnaire ON (répond aux augmentations de lumière).
    """
    
    def __init__(self, **kwargs):
        kwargs['cell_type'] = kwargs.get('cell_type', 'midget')
        super().__init__(**kwargs)
        
        # ON cells: réponse positive à la lumière
        self.on_gain = 1.0
    
    def forward(self, bipolar_input: torch.Tensor, dt: float = 1.0):
        """Cellule ON : réponse aux augmentations."""
        # Les cellules ON reçoivent des entrées ON-bipolaires
        response = super().forward(bipolar_input, dt)
        return response


class OFFGanglionCell(GanglionCell):
    """
    Cellule ganglionnaire OFF (répond aux diminutions de lumière).
    """
    
    def __init__(self, **kwargs):
        kwargs['cell_type'] = kwargs.get('cell_type', 'parasol')
        super().__init__(**kwargs)
        
        # OFF cells: réponse négative à la lumière
        self.off_gain = -1.0
    
    def forward(self, bipolar_input: torch.Tensor, dt: float = 1.0):
        """Cellule OFF : réponse aux diminutions."""
        # Les cellules OFF reçoivent des entrées OFF-bipolaires
        # Mais nous inversons le signe pour une réponse positive aux diminutions
        inverted_input = -bipolar_input
        response = super().forward(inverted_input, dt)
        return response


class GanglionCellLayer(nn.Module):
    """
    Couche de cellules ganglionnaires.
    Simule une population de cellules ganglionnaires de différents types.
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int],
                 n_on_cells: int = 3,
                 n_off_cells: int = 3,
                 cell_distribution: Dict[str, float] = None,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.input_shape = input_shape
        self.device = device
        
        # Distribution par défaut des types de cellules
        self.cell_distribution = cell_distribution or {
            'parasol': 0.3,   # 30% M-cells
            'midget': 0.6,    # 60% P-cells
            'bistratified': 0.1  # 10% bistratifiées
        }
        
        # Créer les cellules ganglionnaires
        self.on_cells = self._create_cell_population(n_on_cells, on_off='ON')
        self.off_cells = self._create_cell_population(n_off_cells, on_off='OFF')
        
        # Total de cellules
        self.n_cells = n_on_cells + n_off_cells
        
        # Positions réceptives (pour diversité spatiale)
        self.receptive_positions = self._create_receptive_positions()
    
    def _create_cell_population(self, n_cells: int, on_off: str = 'ON') -> nn.ModuleList:
        """Crée une population de cellules ganglionnaires."""
        cells = nn.ModuleList()
        
        for i in range(n_cells):
            # Échantillonner le type selon la distribution
            cell_type = np.random.choice(
                list(self.cell_distribution.keys()),
                p=list(self.cell_distribution.values())
            )
            
            # Créer la cellule
            if on_off == 'ON':
                cell = ONGanglionCell(cell_type=cell_type, device=self.device)
            else:
                cell = OFFGanglionCell(cell_type=cell_type, device=self.device)
            
            cells.append(cell)
        
        return cells
    
    def _create_receptive_positions(self) -> Dict:
        """Crée des positions réceptives variées."""
        height, width = self.input_shape
        positions = {}
        
        # Pour chaque cellule, assigner une position réceptive
        cell_idx = 0
        
        # Positions pour les cellules ON
        for i, cell in enumerate(self.on_cells):
            # Position aléatoire dans le champ visuel
            pos_y = torch.rand(1) * height
            pos_x = torch.rand(1) * width
            
            positions[f'on_{i}'] = {
                'position': (pos_y.item(), pos_x.item()),
                'size': cell.receptive_field_size,
                'type': cell.cell_type
            }
            cell_idx += 1
        
        # Positions pour les cellules OFF
        for i, cell in enumerate(self.off_cells):
            pos_y = torch.rand(1) * height
            pos_x = torch.rand(1) * width
            
            positions[f'off_{i}'] = {
                'position': (pos_y.item(), pos_x.item()),
                'size': cell.receptive_field_size,
                'type': cell.cell_type
            }
            cell_idx += 1
        
        return positions
    
    def reset_state(self):
        """Réinitialise toutes les cellules."""
        for cell in self.on_cells:
            cell.reset_state()
        for cell in self.off_cells:
            cell.reset_state()
    
    def _extract_receptive_field(self,
                                input_map: torch.Tensor,
                                position: Tuple[float, float],
                                field_size: int) -> torch.Tensor:
        """Extrait le champ réceptif à une position donnée."""
        # Gestion des dimensions
        original_shape = input_map.shape
        
        if len(input_map.shape) == 2:
            # (H, W) -> (1, 1, H, W)
            input_map = input_map.unsqueeze(0).unsqueeze(0)
        elif len(input_map.shape) == 3:
            # (C, H, W) ou (B, H, W)
            if input_map.shape[0] <= 3:  # Probablement des canaux (C, H, W)
                # Prendre la moyenne des canaux et ajouter batch
                if input_map.shape[0] == 3:  # RGB
                    input_map = input_map.mean(dim=0, keepdim=True)  # (1, H, W)
                input_map = input_map.unsqueeze(0)  # (1, 1, H, W)
            else:
                # (B, H, W) -> (B, 1, H, W)
                input_map = input_map.unsqueeze(1)
        elif len(input_map.shape) == 4:
            # (B, C, H, W)
            if input_map.shape[1] > 1:
                # Prendre la moyenne des canaux
                input_map = input_map.mean(dim=1, keepdim=True)
            # Sinon déjà (B, 1, H, W)
        else:
            raise ValueError(f"Shape d'entrée non supportée: {original_shape}")
        
        # Maintenant input_map est (B, 1, H, W)
        batch_size, channels, height, width = input_map.shape
        
        # Coordonnées de la position
        center_y, center_x = position
        
        # Bornes du champ réceptif
        y_start = max(0, int(center_y - field_size // 2))
        y_end = min(height, int(center_y + field_size // 2))
        x_start = max(0, int(center_x - field_size // 2))
        x_end = min(width, int(center_x + field_size // 2))
        
        # Extraire la région
        if y_end > y_start and x_end > x_start:
            receptive_field = input_map[:, :, y_start:y_end, x_start:x_end]
            
            # Redimensionner si nécessaire
            if receptive_field.shape[-2:] != (field_size, field_size):
                receptive_field = F.interpolate(
                    receptive_field,
                    size=(field_size, field_size),
                    mode='bilinear'
                )
        else:
            # Si hors limites, retourner zéro
            receptive_field = torch.zeros(batch_size, 1, field_size, field_size, device=self.device)
        
        # Retourner (B, field_size, field_size) pour GanglionCell.forward
        return receptive_field.squeeze(1)

    def forward(self,
                bipolar_on_input: torch.Tensor,
                bipolar_off_input: torch.Tensor,
                dt: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Traitement par la couche ganglionnaire.
        
        Args:
            bipolar_on_input: Entrée ON des cellules bipolaires (H, W) ou (B, H, W) ou (C, H, W)
            bipolar_off_input: Entrée OFF des cellules bipolaires
            dt: Pas de temps
            
        Returns:
            Dictionnaire avec les réponses
        """
        # Vérifier et normaliser les dimensions
        def normalize_input(input_tensor):
            original_shape = input_tensor.shape
            
            if len(input_tensor.shape) == 2:
                # (H, W) -> (1, 1, H, W)
                return input_tensor.unsqueeze(0).unsqueeze(0)
            elif len(input_tensor.shape) == 3:
                # (C, H, W) ou (B, H, W)
                if input_tensor.shape[0] <= 3:  # Probablement des canaux
                    # RGB ou similar, prendre la moyenne
                    if input_tensor.shape[0] == 3:
                        input_tensor = input_tensor.mean(dim=0, keepdim=True)  # (1, H, W)
                    return input_tensor.unsqueeze(0)  # (1, 1, H, W)
                else:
                    # (B, H, W) -> (B, 1, H, W)
                    return input_tensor.unsqueeze(1)
            elif len(input_tensor.shape) == 4:
                # (B, C, H, W)
                if input_tensor.shape[1] > 1:
                    # Prendre la moyenne des canaux
                    input_tensor = input_tensor.mean(dim=1, keepdim=True)
                return input_tensor
            else:
                raise ValueError(f"Shape d'entrée non supportée: {original_shape}")
        
        on_input = normalize_input(bipolar_on_input)
        off_input = normalize_input(bipolar_off_input)
        
        batch_size = on_input.shape[0]
        
        # Réponses des cellules ON
        on_spikes_list = []
        on_potentials_list = []
        
        for i, cell in enumerate(self.on_cells):
            if f'on_{i}' in self.receptive_positions:
                pos_info = self.receptive_positions[f'on_{i}']
                position = pos_info['position']
                field_size = pos_info['size']
                
                # Extraire le champ réceptif
                receptive_field = self._extract_receptive_field(
                    on_input, position, field_size
                )
                
                # Traitement par la cellule
                spikes, potential = cell(receptive_field, dt)
                
                on_spikes_list.append(spikes)
                on_potentials_list.append(potential)
        
        # Réponses des cellules OFF
        off_spikes_list = []
        off_potentials_list = []
        
        for i, cell in enumerate(self.off_cells):
            if f'off_{i}' in self.receptive_positions:
                pos_info = self.receptive_positions[f'off_{i}']
                position = pos_info['position']
                field_size = pos_info['size']
                
                # Extraire le champ réceptif
                receptive_field = self._extract_receptive_field(
                    off_input, position, field_size
                )
                
                # Traitement par la cellule
                spikes, potential = cell(receptive_field, dt)
                
                off_spikes_list.append(spikes)
                off_potentials_list.append(potential)
        
        # Convertir en tenseurs (gestion des tailles variables en gardant une liste ou en padant)
        # Ici on utilise une liste de tensors pour éviter les erreurs de stack avec tailles différentes
        # Ou on redimensionne tout à 1x1 si on veut juste l'activité
        
        # Option: Stack si tailles identiques, sinon liste
        try:
            on_spikes = torch.stack(on_spikes_list, dim=1) if on_spikes_list else torch.tensor([])
            on_potentials = torch.stack(on_potentials_list, dim=1) if on_potentials_list else torch.tensor([])
            off_spikes = torch.stack(off_spikes_list, dim=1) if off_spikes_list else torch.tensor([])
            off_potentials = torch.stack(off_potentials_list, dim=1) if off_potentials_list else torch.tensor([])
        except RuntimeError:
            # Tailles différentes, on prend la moyenne spatiale pour chaque cellule (rate code)
            on_spikes = torch.stack([s.mean(dim=(-1,-2)) for s in on_spikes_list], dim=1) if on_spikes_list else torch.tensor([])
            on_potentials = torch.stack([p.mean(dim=(-1,-2)) for p in on_potentials_list], dim=1) if on_potentials_list else torch.tensor([])
            off_spikes = torch.stack([s.mean(dim=(-1,-2)) for s in off_spikes_list], dim=1) if off_spikes_list else torch.tensor([])
            off_potentials = torch.stack([p.mean(dim=(-1,-2)) for p in off_potentials_list], dim=1) if off_potentials_list else torch.tensor([])
        
        return {
            'on_spikes': on_spikes,
            'on_potentials': on_potentials,
            'off_spikes': off_spikes,
            'off_potentials': off_potentials,
            'positions': self.receptive_positions,
            'n_cells': self.n_cells
        }    


def create_ganglion_population(input_shape: Tuple[int, int],
                              n_total_cells: int = 100,
                              on_off_ratio: float = 0.5,
                              device: str = 'cpu') -> GanglionCellLayer:
    """
    Crée une population standard de cellules ganglionnaires.
    
    Args:
        input_shape: Forme d'entrée (H, W)
        n_total_cells: Nombre total de cellules
        on_off_ratio: Ratio ON/OFF (0.5 = égal)
        device: Device
        
    Returns:
        Couche ganglionnaire
    """
    n_on_cells = int(n_total_cells * on_off_ratio)
    n_off_cells = n_total_cells - n_on_cells
    
    return GanglionCellLayer(
        input_shape=input_shape,
        n_on_cells=n_on_cells,
        n_off_cells=n_off_cells,
        device=device
    )
