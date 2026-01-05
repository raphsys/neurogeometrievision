"""
Module photoreceptors.py - Modélisation des photorécepteurs rétiniens
Cônes (S, M, L) et bâtonnets, organisation en mosaïque
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import math


class Photoreceptor(nn.Module):
    """
    Classe de base pour les photorécepteurs individuels.
    """
    
    def __init__(self,
                 spectral_sensitivity: List[float] = None,
                 response_gain: float = 1.0,
                 adaptation_rate: float = 0.1,
                 noise_level: float = 0.01,
                 tau_response: float = 20.0,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.spectral_sensitivity = spectral_sensitivity or [1.0]
        self.response_gain = response_gain
        self.adaptation_rate = adaptation_rate
        self.noise_level = noise_level
        self.tau_response = tau_response
        self.device = device
        
        # État d'adaptation
        self.register_buffer('adaptation_state', torch.tensor(1.0, device=device))
        self.register_buffer('current_response', torch.tensor(0.0, device=device))
        self.register_buffer('filtered_response', torch.tensor(0.0, device=device))
    
    def reset_state(self):
        """Réinitialise l'état."""
        self.adaptation_state = torch.tensor(1.0, device=self.device)
        self.current_response = torch.tensor(0.0, device=self.device)
        self.filtered_response = torch.tensor(0.0, device=self.device)
    
    def spectral_response(self, wavelength: float) -> float:
        """
        Réponse spectrale (simplifiée).
        
        Args:
            wavelength: Longueur d'onde en nm
            
        Returns:
            Sensibilité normalisée
        """
        # Pour simplifier, sensibilité gaussienne
        peak_wavelength = 550.0  # nm (vert)
        bandwidth = 100.0  # nm
        
        response = math.exp(-((wavelength - peak_wavelength) ** 2) / (2 * bandwidth ** 2))
        return response
    
    def temporal_filter(self, light_input: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """
        Filtre temporel de la réponse.
        
        Args:
            light_input: Intensité lumineuse
            dt: Pas de temps
            
        Returns:
            Réponse filtrée
        """
        alpha = math.exp(-dt / self.tau_response)
        self.filtered_response = alpha * self.filtered_response + (1 - alpha) * light_input
        return self.filtered_response
    
    def adapt(self, light_level: float, dt: float = 1.0):
        """
        Adaptation à la lumière.
        
        Args:
            light_level: Niveau de lumière
            dt: Pas de temps
        """
        # Adaptation lente
        target_adaptation = 1.0 / (1.0 + light_level)
        self.adaptation_state += self.adaptation_rate * (target_adaptation - self.adaptation_state) * dt
        self.adaptation_state = torch.clamp(self.adaptation_state, 0.1, 10.0)
    
    def forward(self,
                light_intensity: torch.Tensor,
                wavelength: Optional[float] = None,
                dt: float = 1.0) -> torch.Tensor:
        """
        Réponse du photorécepteur.
        
        Args:
            light_intensity: Intensité lumineuse
            wavelength: Longueur d'onde (optionnel)
            dt: Pas de temps
            
        Returns:
            Réponse normalisée
        """
        # Bruit
        noise = torch.randn_like(light_intensity) * self.noise_level
        light_with_noise = torch.clamp(light_intensity + noise, 0.0, None)
        
        # Sensibilité spectrale
        if wavelength is not None:
            spectral_factor = self.spectral_response(wavelength)
            light_with_noise = light_with_noise * spectral_factor
        
        # Adaptation
        self.adapt(light_with_noise.mean().item(), dt)
        
        # Filtre temporel
        filtered = self.temporal_filter(light_with_noise, dt)
        
        # Réponse non-linéaire (logarithmique approximative)
        response = self.response_gain * torch.log(1.0 + filtered * self.adaptation_state)
        
        # Normalisation
        response = torch.tanh(response)
        
        self.current_response = response
        return response


class Cone(Photoreceptor):
    """
    Cône photorécepteur.
    Types : S (bleu), M (vert), L (rouge)
    """
    
    def __init__(self,
                 cone_type: str = 'M',  # 'S', 'M', 'L'
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.cone_type = cone_type
        
        # Sensibilités spectrales par type
        self.peak_wavelengths = {
            'S': 420.0,  # Bleu
            'M': 534.0,  # Vert
            'L': 564.0   # Rouge
        }
        
        self.bandwidths = {
            'S': 50.0,
            'M': 60.0,
            'L': 70.0
        }
        
        # Gains par type
        gains = {'S': 0.8, 'M': 1.0, 'L': 0.9}
        self.response_gain = gains.get(cone_type, 1.0)
    
    def spectral_response(self, wavelength: float) -> float:
        """Réponse spectrale spécifique au type de cône."""
        peak = self.peak_wavelengths.get(self.cone_type, 550.0)
        bandwidth = self.bandwidths.get(self.cone_type, 60.0)
        
        response = math.exp(-((wavelength - peak) ** 2) / (2 * bandwidth ** 2))
        return response


class Rod(Photoreceptor):
    """
    Bâtonnet photorécepteur.
    Sensible à faible luminosité.
    """
    
    def __init__(self,
                 scotopic_gain: float = 10.0,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.scotopic_gain = scotopic_gain
        self.response_gain = 5.0  # Plus sensible que les cônes
        
        # Bâtonnets plus lents
        self.tau_response = 100.0
    
    def spectral_response(self, wavelength: float) -> float:
        """Réponse spectrale des bâtonnets (peak à 498 nm)."""
        peak = 498.0  # nm
        bandwidth = 80.0
        
        response = math.exp(-((wavelength - peak) ** 2) / (2 * bandwidth ** 2))
        return response
    
    def forward(self, light_intensity: torch.Tensor, **kwargs) -> torch.Tensor:
        """Réponse avec gain scotopique."""
        # Gain plus élevé en faible luminosité
        low_light_gain = self.scotopic_gain / (1.0 + light_intensity.mean() + 1e-8)
        response = super().forward(light_intensity, **kwargs)
        return response * low_light_gain


class PhotoreceptorLayer(nn.Module):
    """
    Couche de photorécepteurs organisée spatialement.
    """
    
    def __init__(self,
                 mosaic_shape: Tuple[int, int],
                 cone_distribution: Dict[str, float] = None,
                 rod_density: float = 0.3,
                 foveal_region: bool = True,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.height, self.width = mosaic_shape
        self.rod_density = rod_density
        self.foveal_region = foveal_region
        self.device = device
        
        # Distribution des cônes (par défaut : vision humaine)
        self.cone_distribution = cone_distribution or {
            'S': 0.05,  # 5% cônes S
            'M': 0.60,  # 60% cônes M
            'L': 0.35   # 35% cônes L
        }
        
        # Créer la mosaïque
        self.mosaic = self._create_mosaic()
        
        # Nombre total de photorécepteurs
        self.n_receptors = len(self.mosaic)
        
    
    def _create_mosaic(self) -> List[Dict]:
        """Crée une mosaïque de photorécepteurs."""
        mosaic = []
        
        # Coordonnées normalisées
        y_coords = torch.linspace(-1, 1, self.height)
        x_coords = torch.linspace(-1, 1, self.width)
        
        for i in range(self.height):
            for j in range(self.width):
                # Position
                x, y = x_coords[j], y_coords[i]
                
                # Distance du centre (pour la fovéa)
                r = math.sqrt(x**2 + y**2)
                
                # Déterminer le type de photorécepteur
                if self.foveal_region:
                    # Dans la fovéa : seulement des cônes (si disponibles)
                    if r < 0.2:  # Fovéa centrale
                        receptor_type = 'cone'
                        cone_type = self._sample_cone_type(r)
                    else:
                        # Mélange cônes/bâtonnets
                        if torch.rand(1) < self.rod_density and self.rod_density > 0:
                            receptor_type = 'rod'
                            cone_type = None
                        elif self.cone_distribution:  # Vérifier qu'il y a des cônes
                            receptor_type = 'cone'
                            cone_type = self._sample_cone_type(r)
                        else:
                            # Pas de cônes, seulement bâtonnets
                            receptor_type = 'rod'
                            cone_type = None
                else:
                    # Distribution uniforme
                    if torch.rand(1) < self.rod_density and self.rod_density > 0:
                        receptor_type = 'rod'
                        cone_type = None
                    elif self.cone_distribution:
                        # Échantillonner parmi les types de cônes disponibles
                        cone_types = list(self.cone_distribution.keys())
                        probs = list(self.cone_distribution.values())
                        # Normaliser
                        total = sum(probs)
                        if total > 0:
                            probs = [p / total for p in probs]
                            cone_type = np.random.choice(cone_types, p=probs)
                        else:
                            cone_type = np.random.choice(cone_types)
                        receptor_type = 'cone'
                    else:
                        receptor_type = 'rod'
                        cone_type = None
                
                # Créer le photorécepteur
                if receptor_type == 'cone' and cone_type is not None:
                    receptor = Cone(cone_type=cone_type, device=self.device)
                else:
                    receptor = Rod(device=self.device)
                
                mosaic.append({
                    'receptor': receptor,
                    'position': (x, y),
                    'type': receptor_type,
                    'cone_type': cone_type,
                    'grid_position': (i, j)
                })
        
        return mosaic    


    def _sample_cone_type(self, distance_from_center: float) -> str:
        """Échantillonne un type de cône basé sur la distance."""
        # Dans la fovéa, distribution différente
        if distance_from_center < 0.1:
            # Fovéa centrale : pas de cônes S
            foveal_dist = {'M': 0.65, 'L': 0.35}
            types = list(foveal_dist.keys())
            probs = list(foveal_dist.values())
        else:
            types = list(self.cone_distribution.keys())
            probs = list(self.cone_distribution.values())
        
        # Normaliser les probabilités pour qu'elles somment à 1
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            # Fallback: distribution uniforme
            probs = [1.0 / len(types) for _ in types]
        
        return np.random.choice(types, p=probs)
        
    
    def reset_state(self):
        """Réinitialise tous les photorécepteurs."""
        for item in self.mosaic:
            item['receptor'].reset_state()
    
    def forward(self,
                image: torch.Tensor,
                wavelength: Optional[float] = None,
                dt: float = 1.0) -> torch.Tensor:
        """
        Traite une image à travers la mosaïque.
        
        Args:
            image: Image (height, width) ou (channels, height, width)
            wavelength: Longueur d'onde unique ou carte
            dt: Pas de temps
            
        Returns:
            Réponses des photorécepteurs
        """
        # Gestion des dimensions
        if len(image.shape) == 2:
            # Image 2D (H, W) -> traiter comme un seul canal
            image = image.unsqueeze(0)  # (1, H, W)
            n_channels = 1
        elif len(image.shape) == 3:
            # Image 3D (C, H, W)
            n_channels = image.shape[0]
        elif len(image.shape) == 4:
            # Image 4D (B, C, H, W)
            batch_size, n_channels, h, w = image.shape
            # For simplicity, if batch, process each item separately or just handle first one
            # But let's support batch by processing each channel
            pass
        else:
            raise ValueError(f"Shape d'image non supportée: {image.shape}")
        
        # Si multi-canal (RGB), traiter chaque canal séparément
        if n_channels == 3:
            responses = []
            for c in range(3):
                # Longueurs d'onde approximatives pour RGB
                wavelengths = [630.0, 530.0, 450.0][c]
                response = self._process_channel(image[:, c:c+1] if len(image.shape)==4 else image[c:c+1], wavelengths, dt)
                responses.append(response)
            output = torch.stack(responses, dim=1 if len(image.shape)==4 else 0)
        else:
            # Image à un canal
            output = self._process_channel(image, wavelength, dt)
        
        return output
    
    def _process_channel(self,
                        image_channel: torch.Tensor,
                        wavelength: Optional[float] = None,
                        dt: float = 1.0) -> torch.Tensor:
        """Traite un canal d'image."""
        # Handle batch dimension if present
        if len(image_channel.shape) == 4:
            batch_size = image_channel.shape[0]
            results = []
            for b in range(batch_size):
                results.append(self._process_single_channel(image_channel[b], wavelength, dt))
            return torch.stack(results, dim=0)
        else:
            return self._process_single_channel(image_channel, wavelength, dt)

    def _process_single_channel(self,
                               image_channel: torch.Tensor,
                               wavelength: Optional[float] = None,
                               dt: float = 1.0) -> torch.Tensor:
        """Traite une seule image à un canal."""
        # image_channel is (1, H, W) or (H, W)
        if len(image_channel.shape) == 3:
            image_channel = image_channel.squeeze(0)
            
        height, width = image_channel.shape
        
        # Interpolation si nécessaire
        if height != self.height or width != self.width:
            image_resized = F.interpolate(
                image_channel.unsqueeze(0).unsqueeze(0),
                size=(self.height, self.width),
                mode='bilinear'
            ).squeeze()
        else:
            image_resized = image_channel
        
        # Réponses individuelles
        responses = torch.zeros(self.height, self.width, device=self.device)
        
        for item in self.mosaic:
            i, j = item['grid_position']
            receptor = item['receptor']
            
            # Intensité au pixel
            intensity = image_resized[i, j].unsqueeze(0)
            
            # Réponse du photorécepteur
            response = receptor(intensity, wavelength=wavelength, dt=dt)
            
            # S'assurer que response est un scalaire
            if response.numel() > 1:
                response = response.mean()
            
            responses[i, j] = response
        
        return responses            

    def get_receptor_types(self) -> torch.Tensor:
        """
        Retourne une carte des types de récepteurs.
        
        Returns:
            Tensor: 0=rod, 1=S, 2=M, 3=L
        """
        type_map = torch.zeros(self.height, self.width, dtype=torch.long, device=self.device)
        
        type_codes = {'rod': 0, 'S': 1, 'M': 2, 'L': 3}
        
        for item in self.mosaic:
            i, j = item['grid_position']
            if item['type'] == 'rod':
                code = type_codes['rod']
            else:
                code = type_codes[item['cone_type']]
            type_map[i, j] = code
        
        return type_map


def create_foveal_distribution(resolution: int = 100,
                              foveal_radius: float = 0.2) -> torch.Tensor:
    """
    Crée une distribution fovéale de densité de photorécepteurs.
    
    Args:
        resolution: Résolution de la carte
        foveal_radius: Rayon de la fovéa
        
    Returns:
        Densité normalisée
    """
    # Coordonnées
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, resolution),
        torch.linspace(-1, 1, resolution),
        indexing='ij'
    )
    
    # Distance du centre
    r = torch.sqrt(x**2 + y**2)
    
    # Densité : haute au centre, décroît avec la distance
    density = torch.exp(-r**2 / (2 * foveal_radius**2))
    
    # Normalisation
    density = density / density.max()
    
    return density


def create_retinal_mosaic(shape: Tuple[int, int],
                         receptor_types: List[str] = None,
                         device: str = 'cpu') -> PhotoreceptorLayer:
    """
    Crée une mosaïque rétinienne standard.
    
    Args:
        shape: Forme de la mosaïque (height, width)
        receptor_types: Types de récepteurs à inclure
        device: Device
        
    Returns:
        Couche de photorécepteurs
    """
    if receptor_types is None:
        receptor_types = ['S', 'M', 'L', 'rod']
    
    # Calculer les proportions
    cone_types = [t for t in receptor_types if t in ['S', 'M', 'L']]
    rod_types = [t for t in receptor_types if t == 'rod']
    
    cone_proportion = len(cone_types) / len(receptor_types) if receptor_types else 0
    rod_proportion = len(rod_types) / len(receptor_types) if receptor_types else 0
    
    # Distribution des cônes
    cone_dist = {}
    if 'S' in cone_types:
        cone_dist['S'] = 0.05
    if 'M' in cone_types:
        cone_dist['M'] = 0.60
    if 'L' in cone_types:
        cone_dist['L'] = 0.35
    
    # Normaliser si nécessaire
    total_cone = sum(cone_dist.values())
    if total_cone > 0 and cone_proportion > 0:
        for k in cone_dist:
            cone_dist[k] = cone_dist[k] / total_cone * cone_proportion
    elif cone_proportion == 0:
        cone_dist = {}
    
    return PhotoreceptorLayer(
        mosaic_shape=shape,
        cone_distribution=cone_dist,
        rod_density=rod_proportion,
        foveal_region=True,
        device=device
    )
