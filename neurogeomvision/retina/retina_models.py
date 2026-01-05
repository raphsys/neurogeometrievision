"""
Module retina_models.py - Modèles complets de rétine
Intégration de tous les composants rétiniens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import math

# Import des autres modules du package retina
from .photoreceptors import PhotoreceptorLayer, create_retinal_mosaic
from .retinal_cells import RetinalNetwork, create_retinal_circuit
from .ganglion_cells import GanglionCellLayer, create_ganglion_population
from .retinal_maps import RetinotopicMap, create_retinotopic_mapping
from .retinal_processing import create_retinal_processing_pipeline


class SimpleRetinaModel(nn.Module):
    """
    Modèle simplifié de rétine.
    Intègre photorécepteurs, cellules bipolaires et ganglionnaires.
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int],
                 n_ganglion_cells: int = 100,
                 use_color: bool = True,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.input_shape = input_shape
        self.n_ganglion_cells = n_ganglion_cells
        self.use_color = use_color
        self.device = device
        
        # 1. Couche de photorécepteurs
        self.photoreceptors = create_retinal_mosaic(
            shape=input_shape,
            receptor_types=['S', 'M', 'L', 'rod'] if use_color else ['rod'],
            device=device
        )
        
        # 2. Circuit rétinien (cellules horizontales, bipolaires, amacrines)
        n_channels = 3 if use_color else 1
        self.retinal_circuit = create_retinal_circuit(
            input_shape=input_shape,
            n_channels=n_channels,
            device=device
        )
        
        # 3. Couche de cellules ganglionnaires
        self.ganglion_layer = create_ganglion_population(
            input_shape=input_shape,
            n_total_cells=n_ganglion_cells,
            on_off_ratio=0.5,  # 50% ON, 50% OFF
            device=device
        )
        
        # 4. Pipeline de traitement optionnel
        self.processing_pipeline = create_retinal_processing_pipeline(
            steps=['filtering', 'center_surround', 'normalization'],
            device=device
        )
        
        # Paramètres
        self.dt = 1.0  # ms
        self.current_time = 0.0
    
    def reset_state(self):
        """Réinitialise tous les états."""
        self.photoreceptors.reset_state()
        self.retinal_circuit.reset_state()
        self.ganglion_layer.reset_state()
        self.current_time = 0.0
    
    def forward(self,
                image: torch.Tensor,
                wavelength: Optional[float] = None,
                light_level: float = 1.0,
                return_intermediate: bool = False) -> Dict[str, torch.Tensor]:
        """
        Traitement complet d'une image par le modèle de rétine.
        
        Args:
            image: Image d'entrée (H, W) ou (C, H, W) ou (B, C, H, W)
            wavelength: Longueur d'onde (pour monochrome)
            light_level: Niveau de lumière (0-1)
            return_intermediate: Retourner les résultats intermédiaires
            
        Returns:
            Résultats du traitement
        """
        # Gestion des dimensions d'entrée
        batch_mode = len(image.shape) == 4
        if len(image.shape) == 2:
            image = image.unsqueeze(0)  # (1, H, W)
        if len(image.shape) == 3 and self.use_color and image.shape[0] != 3:
            image = image.unsqueeze(0)  # (B, 1, H, W)
        
        if image.dtype != torch.float32:
            image = image.float()
        
        # Normaliser l'image
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Résultats intermédiaires
        intermediate = {}
        
        # 1. Photorécepteurs
        photoreceptor_response = self.photoreceptors(image, wavelength=wavelength, dt=self.dt)
        if return_intermediate:
            intermediate['photoreceptor_response'] = photoreceptor_response
            intermediate['receptor_types'] = self.photoreceptors.get_receptor_types()
        
        # 2. Pipeline de traitement optionnel
        processed = self.processing_pipeline(photoreceptor_response, light_level=light_level)
        if return_intermediate:
            intermediate.update(processed)
        
        # 3. Circuit rétinien
        circuit_input = processed.get('output', photoreceptor_response)
        circuit_results = self.retinal_circuit(circuit_input, dt=self.dt)
        if return_intermediate:
            intermediate.update(circuit_results)
        
        # 4. Cellules ganglionnaires
        # Séparer les entrées ON et OFF pour les cellules ganglionnaires
        on_input = circuit_results.get('on_bipolar', circuit_input)
        off_input = circuit_results.get('off_bipolar', circuit_input)
        
        # Si pas de sortie bipolaire spécifique, utiliser le traitement centre-surround
        if on_input.numel() == 0 or off_input.numel() == 0:
            cs_results = center_surround_processing(circuit_input, device=self.device)
            on_input = cs_results['on_response']
            off_input = cs_results['off_response']
        
        ganglion_results = self.ganglion_layer(on_input, off_input, dt=self.dt)
        
        # 5. Agréger les résultats
        results = {
            'ganglion_spikes': {
                'on': ganglion_results['on_spikes'],
                'off': ganglion_results['off_spikes']
            },
            'ganglion_potentials': {
                'on': ganglion_results['on_potentials'],
                'off': ganglion_results['off_potentials']
            },
            'n_ganglion_cells': self.ganglion_layer.n_cells,
            'ganglion_positions': ganglion_results['positions'],
            'processing_time': self.current_time,
            'input_shape': image.shape
        }
        
        # Mettre à jour le temps
        self.current_time += self.dt
        
        if return_intermediate:
            results['intermediate'] = intermediate
        
        return results
    
    def simulate_sequence(self,
                         image_sequence: List[torch.Tensor],
                         light_levels: Optional[List[float]] = None,
                         reset_between_frames: bool = False) -> Dict[str, List]:
        """
        Simule la réponse à une séquence d'images.
        
        Args:
            image_sequence: Liste d'images
            light_levels: Niveaux de lumière pour chaque image
            reset_between_frames: Réinitialiser l'état entre les images
            
        Returns:
            Réponses temporelles
        """
        if light_levels is None:
            light_levels = [1.0] * len(image_sequence)
        
        all_spikes = []
        all_potentials = []
        
        for i, (image, light_level) in enumerate(zip(image_sequence, light_levels)):
            if reset_between_frames and i > 0:
                self.reset_state()
            
            results = self(image, light_level=light_level, return_intermediate=False)
            
            # Agréger les spikes
            frame_spikes = {
                'on': results['ganglion_spikes']['on'],
                'off': results['ganglion_spikes']['off']
            }
            
            # Agréger les potentiels
            frame_potentials = {
                'on': results['ganglion_potentials']['on'],
                'off': results['ganglion_potentials']['off']
            }
            
            all_spikes.append(frame_spikes)
            all_potentials.append(frame_potentials)
        
        return {
            'spike_sequence': all_spikes,
            'potential_sequence': all_potentials,
            'n_frames': len(image_sequence),
            'frame_times': [i * self.dt for i in range(len(image_sequence))]
        }


class BioInspiredRetina(nn.Module):
    """
    Rétine bio-inspirée avec cartographie rétinotopique.
    Simule le traitement complet de la rétine au cortex.
    """
    
    def __init__(self,
                 retinal_shape: Tuple[int, int],
                 cortical_shape: Tuple[int, int],
                 n_ganglion_cells: int = 200,
                 include_retinotopic_mapping: bool = True,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.retinal_shape = retinal_shape
        self.cortical_shape = cortical_shape
        self.n_ganglion_cells = n_ganglion_cells
        self.include_retinotopic_mapping = include_retinotopic_mapping
        self.device = device
        
        # 1. Modèle de rétine
        self.retina = SimpleRetinaModel(
            input_shape=retinal_shape,
            n_ganglion_cells=n_ganglion_cells,
            use_color=True,
            device=device
        )
        
        # 2. Carte rétinotopique (optionnelle)
        if include_retinotopic_mapping:
            self.retinotopic_map = create_retinotopic_mapping(
                retinal_resolution=retinal_shape,
                cortical_resolution=cortical_shape,
                magnification=15.0,
                device=device
            )
        
        # 3. Couche corticale simple (optionnelle)
        self.cortical_processing = self._create_cortical_processing()
        
        # Paramètres temporels
        self.dt = 1.0
        self.time = 0.0
    
    def _create_cortical_processing(self) -> nn.Module:
        """Crée une couche de traitement cortical simple."""
        return nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),  # ON et OFF
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(self.cortical_shape)
        )
    
    def reset_state(self):
        """Réinitialise tous les états."""
        self.retina.reset_state()
        self.time = 0.0
    
    def forward(self,
                image: torch.Tensor,
                return_cortical: bool = False) -> Dict[str, torch.Tensor]:
        """
        Traitement complet : rétine -> carte rétinotopique -> cortex.
        
        Args:
            image: Image d'entrée
            return_cortical: Retourner la représentation corticale
            
        Returns:
            Résultats du traitement
        """
        # 1. Traitement rétinien
        retinal_results = self.retina(image, return_intermediate=False)
        
        # Extraire les spikes ganglionnaires
        on_spikes = retinal_results['ganglion_spikes']['on']
        off_spikes = retinal_results['ganglion_spikes']['off']
        
        # 2. Organiser les spikes en carte spatiale
        if on_spikes.numel() > 0 and off_spikes.numel() > 0:
            # Créer des cartes de spikes ON et OFF
            spike_maps = self._create_spike_maps(
                retinal_results['ganglion_positions'],
                on_spikes,
                off_spikes
            )
        else:
            # Fallback: utiliser les potentiels
            on_potentials = retinal_results['ganglion_potentials']['on']
            off_potentials = retinal_results['ganglion_potentials']['off']
            spike_maps = {
                'on': on_potentials.mean(dim=1, keepdim=True),
                'off': off_potentials.mean(dim=1, keepdim=True)
            }
        
        # 3. Application de la carte rétinotopique (si activée)
        if self.include_retinotopic_mapping:
            cortical_on = self.retinotopic_map(spike_maps['on'], mode='retina_to_cortex')
            cortical_off = self.retinotopic_map(spike_maps['off'], mode='retina_to_cortex')
        else:
            cortical_on = spike_maps['on']
            cortical_off = spike_maps['off']
        
        # 4. Traitement cortical (optionnel)
        cortical_representation = None
        if return_cortical and cortical_on is not None and cortical_off is not None:
            # Combiner ON et OFF
            cortical_input = torch.cat([cortical_on, cortical_off], dim=1)
            cortical_representation = self.cortical_processing(cortical_input)
        
        # 5. Résultats
        results = {
            'retinal_results': retinal_results,
            'spike_maps': spike_maps,
            'cortical_on': cortical_on,
            'cortical_off': cortical_off,
            'cortical_representation': cortical_representation,
            'processing_time': self.time,
            'n_ganglion_cells': self.n_ganglion_cells
        }
        
        self.time += self.dt
        return results
    
    def _create_spike_maps(self,
                          ganglion_positions: Dict,
                          on_spikes: torch.Tensor,
                          off_spikes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Crée des cartes spatiales à partir des spikes ganglionnaires.
        
        Args:
            ganglion_positions: Positions des cellules ganglionnaires
            on_spikes: Spikes ON
            off_spikes: Spikes OFF
            
        Returns:
            Cartes de spikes
        """
        height, width = self.retinal_shape
        batch_size = on_spikes.shape[0] if len(on_spikes.shape) > 2 else 1
        
        on_map = torch.zeros(batch_size, 1, height, width, device=self.device)
        off_map = torch.zeros(batch_size, 1, height, width, device=self.device)
        
        # Remplir les cartes avec les spikes
        cell_idx = 0
        
        # Cellules ON
        for i in range(len(self.retina.ganglion_layer.on_cells)):
            pos_key = f'on_{i}'
            if pos_key in ganglion_positions:
                pos = ganglion_positions[pos_key]['position']
                y, x = int(pos[0]), int(pos[1])
                
                # Vérifier les limites
                if 0 <= y < height and 0 <= x < width:
                    if batch_size == 1:
                        if len(on_spikes.shape) == 2:  # (n_cells, receptive_field_area)
                            if i < on_spikes.shape[0]:
                                spike_value = on_spikes[i].mean()
                            else:
                                spike_value = 0.0
                        else:  # (1, n_cells, ...)
                            if i < on_spikes.shape[1]:
                                spike_value = on_spikes[0, i].mean()
                            else:
                                spike_value = 0.0
                    else:
                        if i < on_spikes.shape[1]:
                            spike_value = on_spikes[:, i].mean()
                        else:
                            spike_value = 0.0
                    
                    on_map[:, 0, y, x] = spike_value
            
            cell_idx += 1
        
        # Cellules OFF
        for i in range(len(self.retina.ganglion_layer.off_cells)):
            pos_key = f'off_{i}'
            if pos_key in ganglion_positions:
                pos = ganglion_positions[pos_key]['position']
                y, x = int(pos[0]), int(pos[1])
                
                if 0 <= y < height and 0 <= x < width:
                    if batch_size == 1:
                        if len(off_spikes.shape) == 2:
                            if i < off_spikes.shape[0]:
                                spike_value = off_spikes[i].mean()
                            else:
                                spike_value = 0.0
                        else:
                            if i < off_spikes.shape[1]:
                                spike_value = off_spikes[0, i].mean()
                            else:
                                spike_value = 0.0
                    else:
                        if i < off_spikes.shape[1]:
                            spike_value = off_spikes[:, i].mean()
                        else:
                            spike_value = 0.0
                    
                    off_map[:, 0, y, x] = spike_value
            
            cell_idx += 1
        
        # Lissage gaussien
        if height > 10 and width > 10:
            gaussian = _create_gaussian_filter(5, 1.0, self.device)
            on_map = F.conv2d(on_map, gaussian, padding=2)
            off_map = F.conv2d(off_map, gaussian, padding=2)
        
        return {'on': on_map, 'off': off_map}


def simulate_retinal_response(image: torch.Tensor,
                             model_type: str = 'simple',
                             **kwargs) -> Dict[str, torch.Tensor]:
    """
    Fonction utilitaire pour simuler la réponse rétinienne.
    
    Args:
        image: Image d'entrée
        model_type: Type de modèle ('simple' ou 'bioinspired')
        **kwargs: Paramètres du modèle
        
    Returns:
        Réponse rétinienne
    """
    device = kwargs.get('device', 'cpu')
    
    if model_type == 'simple':
        input_shape = image.shape[-2:]
        model = SimpleRetinaModel(
            input_shape=input_shape,
            n_ganglion_cells=kwargs.get('n_ganglion_cells', 100),
            use_color=kwargs.get('use_color', True),
            device=device
        )
    elif model_type == 'bioinspired':
        retinal_shape = image.shape[-2:]
        cortical_shape = kwargs.get('cortical_shape', (200, 200))
        model = BioInspiredRetina(
            retinal_shape=retinal_shape,
            cortical_shape=cortical_shape,
            n_ganglion_cells=kwargs.get('n_ganglion_cells', 200),
            include_retinotopic_mapping=kwargs.get('include_mapping', True),
            device=device
        )
    else:
        raise ValueError(f"Type de modèle inconnu: {model_type}")
    
    # Réinitialiser l'état
    model.reset_state()
    
    # Simulation
    results = model(image, return_cortical=kwargs.get('return_cortical', False))
    
    return results


def process_visual_scene(scene_images: List[torch.Tensor],
                        model: nn.Module,
                        reset_between_frames: bool = False) -> Dict[str, List]:
    """
    Traite une scène visuelle (séquence d'images).
    
    Args:
        scene_images: Liste d'images de la scène
        model: Modèle de rétine
        reset_between_frames: Réinitialiser entre les images
        
    Returns:
        Réponses temporelles
    """
    if not hasattr(model, 'simulate_sequence'):
        # Si le modèle n'a pas de méthode simulate_sequence, utiliser forward
        all_results = []
        
        for i, image in enumerate(scene_images):
            if reset_between_frames and i > 0:
                model.reset_state()
            
            results = model(image)
            all_results.append(results)
        
        return {'frame_responses': all_results, 'n_frames': len(scene_images)}
    else:
        # Utiliser la méthode simulate_sequence du modèle
        return model.simulate_sequence(scene_images, reset_between_frames=reset_between_frames)


# Fonctions d'aide (définies dans retinal_processing.py)
def _create_gaussian_filter(size: int, sigma: float, device: str) -> torch.Tensor:
    """Crée un filtre gaussien (dupliqué pour éviter les imports circulaires)."""
    center = size // 2
    
    y, x = torch.meshgrid(
        torch.arange(size, device=device) - center,
        torch.arange(size, device=device) - center,
        indexing='ij'
    )
    
    r = torch.sqrt(x**2 + y**2)
    gaussian = torch.exp(-r**2 / (2 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    
    return gaussian.unsqueeze(0).unsqueeze(0)


def center_surround_processing(image: torch.Tensor, **kwargs):
    """Wrapper pour éviter les imports circulaires."""
    from .retinal_processing import center_surround_processing as cs_processing
    return cs_processing(image, **kwargs)
