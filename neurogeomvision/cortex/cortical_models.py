"""
Module cortical_models.py - Modèles complets du cortex visuel
Hiérarchie corticale, voies ventrale/dorsale, systèmes intégrés
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import math

from .cortical_areas import V1SimpleCell, V1ComplexCell, CorticalColumn, Hypercolumn, V2Cell, V4Cell, MTCell
from .orientation import OrientationMap, OrientationSelectivity
from .motion import MotionEnergyFilter, DirectionSelectivity, MotionDetector
from .color import ColorOpponency, DoubleOpponentCell, ColorProcessingStream
from .attention import SaliencyMap, SpatialAttention, FeatureBasedAttention, AttentionModel
from .microcircuits import CorticalMicrocircuit, CanonicalMicrocircuit, ExcitatoryInhibitoryBalance, LayerSpecificProcessing


class HierarchicalVisionModel(nn.Module):
    """
    Modèle hiérarchique du cortex visuel.
    Simule le flux de traitement de V1 vers les aires supérieures.
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int],
                 use_color: bool = True,
                 use_motion: bool = True,
                 n_orientations: int = 8,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.input_shape = input_shape
        self.use_color = use_color
        self.use_motion = use_motion
        self.n_orientations = n_orientations
        self.device = device
        
        # [Initialisation des modules existants...]
        # V1: Orientation et luminance
        print("Initialising V1...")
        self.v1_orientation = OrientationSelectivity(
            n_orientations=n_orientations,
            device=device
        )
        
        # V1: Hypercolonne pour l'organisation spatiale
        self.v1_hypercolumn = Hypercolumn(
            input_shape=input_shape,
            column_size=32,
            stride=16,
            n_orientations=n_orientations,
            device=device
        )
        
        # V2: Formes simples
        print("Initialising V2...")
        self.v2_contour = V2Cell(feature_type='contour', device=device)
        self.v2_angle = V2Cell(feature_type='angle', device=device)
        self.v2_junction = V2Cell(feature_type='junction', device=device)
        
        # V4: Formes complexes (si couleur activée)
        if use_color:
            print("Initialising V4...")
            self.v4_color = ColorProcessingStream(input_shape, device=device)
            self.v4_curve = V4Cell(shape_type='curve', device=device)
            self.v4_spiral = V4Cell(shape_type='spiral', device=device)
        
        # MT/V5: Mouvement (si mouvement activé)
        if use_motion:
            print("Initialising MT...")
            self.mt_motion = DirectionSelectivity(n_directions=8, device=device)
            self.motion_detector = MotionDetector(input_shape, device=device)
        
        # Intégration hiérarchique
        print("Initialising integration layers...")
        
        # Calculer la dimension des features dynamiquement
        print("  Calcul de la dimension des features...")
        feature_dim = self._compute_feature_dim()  # Note: J'ai ajouté un underscore
        
        print(f"HierarchicalVisionModel initialisé avec {feature_dim} dimensions de features")
        
        self.integration = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Classification (exemple: catégories d'objets)
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10),  # 10 catégories
            nn.LogSoftmax(dim=1)
        )
        
    def _compute_feature_dim(self) -> int:
        """Calcule dynamiquement la dimension des features."""
        with torch.no_grad():
            # Créer un tensor de test
            test_input = torch.randn(1, 3 if self.use_color else 1, 
                                   self.input_shape[0], self.input_shape[1],
                                   device=self.device)
            
            if not self.use_color and test_input.shape[1] == 3:
                test_input = test_input.mean(dim=1, keepdim=True)
            
            # Forward pass partiel
            v1_features = self.extract_v1_features(test_input)
            v2_features = self.extract_v2_features(v1_features)
            
            all_features = []
            
            # V1
            if 'complex_responses' in v1_features:
                v1_feat = v1_features['complex_responses']
                v1_pooled = F.adaptive_avg_pool2d(v1_feat, (4, 4))
                v1_flatten = v1_pooled.view(1, -1)
                all_features.append(v1_flatten)
            
            # V2
            if 'combined' in v2_features:
                v2_feat = v2_features['combined']
                if len(v2_feat.shape) == 4:
                    v2_pooled = F.adaptive_avg_pool2d(v2_feat, (2, 2))
                    v2_flatten = v2_pooled.view(1, -1)
                else:
                    v2_flatten = v2_feat.view(1, -1)
                all_features.append(v2_flatten)
            
            # V4
            if self.use_color:
                v4_features = self.extract_v4_features(test_input, v1_features)
                if 'combined' in v4_features:
                    v4_feat = v4_features['combined']
                    if len(v4_feat.shape) == 4:
                        v4_pooled = F.adaptive_avg_pool2d(v4_feat, (1, 1))
                        v4_flatten = v4_pooled.view(1, -1)
                    else:
                        v4_flatten = v4_feat
                    all_features.append(v4_flatten)
            
            # MT
            if self.use_motion:
                mt_features = self.extract_mt_features(test_input)
                if 'direction_features' in mt_features:
                    mt_feat = mt_features['direction_features']
                    if len(mt_feat.shape) == 4:
                        mt_pooled = F.adaptive_avg_pool2d(mt_feat, (2, 2))
                        mt_flatten = mt_pooled.view(1, -1)
                    else:
                        mt_flatten = mt_feat.view(1, -1)
                    all_features.append(mt_flatten)
            
            # Concaténer
            if all_features:
                features_concat = torch.cat(all_features, dim=1)
                return features_concat.shape[1]
            else:
                return 1
    
    def extract_v1_features(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extrait les caractéristiques V1."""
        # Orientation
        orientation_results = self.v1_orientation(image, cell_type='complex')
        
        # Hypercolonne pour l'organisation spatiale
        hypercolumn_results = self.v1_hypercolumn(image)
        
        return {
            'orientation': orientation_results,
            'hypercolumn': hypercolumn_results,
            'orientation_map': orientation_results['orientation_map'],
            'response_map': hypercolumn_results['response_map'],
            'complex_responses': orientation_results.get('energy', hypercolumn_results['response_map'])
        }
    
    def extract_v2_features(self, v1_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extrait les caractéristiques V2."""
        # Utiliser la carte de réponse de V1 comme entrée
        v1_response = v1_features['response_map']
    
        # Prendre la moyenne sur les orientations
        if len(v1_response.shape) == 4:
            v1_input = v1_response.mean(dim=1, keepdim=True)
        else:
            v1_input = v1_response
    
        # S'assurer que l'entrée a les bonnes dimensions
        if len(v1_input.shape) == 2:
            v1_input = v1_input.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif len(v1_input.shape) == 3:
            v1_input = v1_input.unsqueeze(1)
    
        # Extraire différentes caractéristiques V2
        contour_response = self.v2_contour(v1_input)
        angle_response = self.v2_angle(v1_input)
        junction_response = self.v2_junction(v1_input)
    
        # Préparer les réponses pour l'empilement
        responses = []
        for resp in [contour_response, angle_response, junction_response]:
            if len(resp.shape) == 2:
                resp = resp.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            elif len(resp.shape) == 3:
                if resp.shape[0] == 1 or resp.shape[0] == 3:  # (C, H, W) ou (1, H, W)
                    resp = resp.unsqueeze(0)  # (1, C, H, W)
                else:  # (B, H, W)
                    resp = resp.unsqueeze(1)  # (B, 1, H, W)
            responses.append(resp)
    
        # Redimensionner à la même taille (utiliser la taille de la première réponse)
        target_size = responses[0].shape[-2:]
        resized_responses = []
    
        for resp in responses:
            if resp.shape[-2:] != target_size:
                resp_resized = F.interpolate(resp, size=target_size, mode='bilinear')
            else:
                resp_resized = resp
            resized_responses.append(resp_resized)
    
        contour_resized, angle_resized, junction_resized = resized_responses
    
        # Empiler
        if resized_responses:
            combined = torch.cat(resized_responses, dim=1)
        else:
            batch_size = v1_input.shape[0] if len(v1_input.shape) > 1 else 1
            combined = torch.empty(batch_size, 0, device=v1_input.device)
    
        return {
            'contour': contour_resized,
            'angle': angle_resized,
            'junction': junction_resized,
            'combined': combined
        }        


    def extract_v4_features(self, image: torch.Tensor, v1_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract V4 features."""
        v4_results = {}
        
        # 1. Features de couleur (si activé)
        if self.use_color and hasattr(self, 'v4_color'):
            color_features = self.v4_color(image)
            v4_results['color'] = color_features
        
        # 2. Features de courbure (à partir de V1 orientation)
        if hasattr(self, 'v4_curve'):
            # Récupérer la carte d'orientation de V1
            v1_orientation = v1_features.get('orientation_map', torch.tensor(0.0))
            
            # S'assurer que v1_orientation a 4 dimensions
            if len(v1_orientation.shape) == 3:
                # (batch_size, height, width) -> (batch_size, 1, height, width)
                v1_orientation = v1_orientation.unsqueeze(1)
            elif len(v1_orientation.shape) == 2:
                # (height, width) -> (1, 1, height, width)
                v1_orientation = v1_orientation.unsqueeze(0).unsqueeze(0)
            
            curve_response = self.v4_curve(v1_orientation)
            v4_results['curve'] = curve_response
        
        # 3. Features de forme (à partir de V2)
        if hasattr(self, 'v4_shape'):
            v2_responses = []
            if hasattr(self, 'v2_contour'):
                v2_responses.append(self.v2_contour(v1_features.get('response_map', image)))
            if hasattr(self, 'v2_angle'):
                v2_responses.append(self.v2_angle(v1_features.get('response_map', image)))
            if hasattr(self, 'v2_junction'):
                v2_responses.append(self.v2_junction(v1_features.get('response_map', image)))
            
            if v2_responses:
                v2_combined = torch.cat([r.unsqueeze(1) if len(r.shape) == 3 else r for r in v2_responses], dim=1)
                shape_response = self.v4_shape(v2_combined.mean(dim=1, keepdim=True))
                v4_results['shape'] = shape_response
        
        # 4. Intégration des features V4
        v4_features_list = []
        
        # Récupérer le batch_size de l'image
        batch_size = image.shape[0] if len(image.shape) == 4 else 1
        
        if 'color' in v4_results:
            color_feat = v4_results['color']
            # Si c'est un dictionnaire, extraire les features
            if isinstance(color_feat, dict):
                if 'color_features' in color_feat:
                    color_feat = color_feat['color_features']
                elif 'features' in color_feat:
                    color_feat = color_feat['features']
            
            # S'assurer que color_feat a la bonne taille
            if len(color_feat.shape) == 1:
                # (features,) -> (1, features) ou (batch_size, features)
                if color_feat.shape[0] == 8:  # Le nombre de features de couleur
                    color_feat = color_feat.unsqueeze(0)
                    if batch_size > 1:
                        # Répéter pour chaque élément du batch
                        color_feat = color_feat.repeat(batch_size, 1)
            elif len(color_feat.shape) == 2:
                # (batch, features) - déjà bon
                pass
            
            v4_features_list.append(color_feat)
        
        if 'curve' in v4_results:
            curve_feat = v4_results['curve']
            if len(curve_feat.shape) == 1:
                # Un tenseur 1D est probablement (batch_size,), il faut le transformer en (batch_size, 1)
                curve_feat = curve_feat.unsqueeze(1)
            v4_features_list.append(curve_feat)
        
        if 'shape' in v4_results:
            shape_feat = v4_results['shape']
            if len(shape_feat.shape) == 1:
                # (features,) -> (1, features) ou (batch_size, features)
                shape_feat = shape_feat.unsqueeze(0)
                if batch_size > 1 and shape_feat.shape[0] == 1:
                    # Répéter pour chaque élément du batch
                    shape_feat = shape_feat.repeat(batch_size, 1)
            v4_features_list.append(shape_feat)
        
        # Vérifier que tous les tenseurs ont la même taille de batch
        if v4_features_list:
            # Ajuster tous à la même taille de batch
            target_batch_size = batch_size
            
            adjusted_features = []
            for feat in v4_features_list:
                if len(feat.shape) == 2:
                    if feat.shape[0] == 1 and target_batch_size > 1:
                        # (1, features) -> (batch_size, features)
                        feat = feat.repeat(target_batch_size, 1)
                    elif feat.shape[0] != target_batch_size:
                        # Taille inattendue, ajuster
                        feat = feat[:target_batch_size] if feat.shape[0] > target_batch_size else feat
                        if feat.shape[0] < target_batch_size:
                            # Padding si nécessaire
                            padding = torch.zeros(target_batch_size - feat.shape[0], feat.shape[1], 
                                                device=feat.device)
                            feat = torch.cat([feat, padding], dim=0)
                
                adjusted_features.append(feat)
            
            # Combiner toutes les features V4
            v4_combined = torch.cat(adjusted_features, dim=1)
            v4_results['combined'] = v4_combined
        else:
            v4_results['combined'] = torch.zeros(batch_size, 1, device=image.device)
        
        return v4_results
        
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass complet.
        
        Args:
            x: Input tensor (B, C, H, W) ou (C, H, W)
        
        Returns:
            Dictionnaire avec tous les résultats
        """
        # S'assurer que x a 4 dimensions
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # (1, C, H, W)
        elif len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        batch_size, channels, height, width = x.shape
        
        # Convertir en niveaux de gris si nécessaire
        if channels == 3 and not self.use_color:
            # Convertir RGB en niveaux de gris
            x_gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
            image = x_gray
        else:
            image = x
        
        # 1. Extraction des features V1
        v1_features = self.extract_v1_features(image)
        
        # 2. Extraction des features V2
        v2_features = self.extract_v2_features(v1_features)
        
        # 3. Extraction des features V4 (si couleur activée)
        v4_features = self.extract_v4_features(image, v1_features) if self.use_color else {}
        
        # 4. Extraction des features MT (si mouvement activé)
        mt_features = self.extract_mt_features(image) if self.use_motion else {}
        
        # 5. Intégration de toutes les features
        all_features = []
        
        # Features V1
        if 'complex_responses' in v1_features:
            v1_feat = v1_features['complex_responses']
            # Pooling spatial
            v1_pooled = F.adaptive_avg_pool2d(v1_feat, (4, 4))
            v1_flatten = v1_pooled.view(batch_size, -1)
            all_features.append(v1_flatten)
        
        # Features V2
        if 'combined' in v2_features:
            v2_feat = v2_features['combined']
            if len(v2_feat.shape) == 4:
                v2_pooled = F.adaptive_avg_pool2d(v2_feat, (2, 2))
                v2_flatten = v2_pooled.view(batch_size, -1)
            else:
                v2_flatten = v2_feat.view(batch_size, -1)
            all_features.append(v2_flatten)
        
        # Features V4
        if 'combined' in v4_features:
            v4_feat = v4_features['combined']
            if len(v4_feat.shape) == 4:
                v4_pooled = F.adaptive_avg_pool2d(v4_feat, (1, 1))
                v4_flatten = v4_pooled.view(batch_size, -1)
            else:
                v4_flatten = v4_feat
            all_features.append(v4_flatten)
        
        # Features MT
        if 'direction_features' in mt_features:
            mt_feat = mt_features['direction_features']
            if len(mt_feat.shape) == 4:
                mt_pooled = F.adaptive_avg_pool2d(mt_feat, (2, 2))
                mt_flatten = mt_pooled.view(batch_size, -1)
            else:
                mt_flatten = mt_feat.view(batch_size, -1)
            all_features.append(mt_flatten)
        
        # Concaténer toutes les features
        if all_features:
            # S'assurer que toutes les features ont le même batch_size
            for i, feat in enumerate(all_features):
                if feat.shape[0] != batch_size:
                    if feat.shape[0] == 1:
                        all_features[i] = feat.repeat(batch_size, 1)
                    else:
                        # Tronquer ou pad
                        all_features[i] = feat[:batch_size]
        
            features_flatten = torch.cat(all_features, dim=1)
        else:
            features_flatten = torch.zeros(batch_size, 1, device=x.device)
    
        # Appliquer la couche d'intégration
        integrated_features = self.integration(features_flatten)
    
        # 6. Classification finale
        classification = self.classifier(integrated_features)
    
        # 7. Retourner tous les résultats
        return {
            'v1': v1_features,
            'v2': v2_features,
            'v4': v4_features if self.use_color else None,
            'mt': mt_features if self.use_motion else None,
            'features': features_flatten,
            'integrated_features': integrated_features,
            'classification': classification
        }

    def extract_mt_features(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extrait les caractéristiques MT (mouvement)."""
        if not self.use_motion or not hasattr(self, 'mt_motion'):
            return {}
        
        # Extraire les features de direction
        direction_results = self.mt_motion(image)
        
        # Détection de mouvement
        motion_results = self.motion_detector(image)
        
        # S'assurer que les features sont 4D
        dir_feat = direction_results.get('direction_map', torch.tensor(0.0))
        if len(dir_feat.shape) == 3:
            dir_feat = dir_feat.unsqueeze(1)
            
        mot_feat = motion_results.get('motion_map', torch.tensor(0.0))
        if len(mot_feat.shape) == 3:
            mot_feat = mot_feat.unsqueeze(1)

        return {
            'direction': direction_results,
            'motion': motion_results,
            'direction_features': dir_feat,
            'motion_features': mot_feat
        }

    def get_feature_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Retourne les shapes de toutes les features."""
        test_input = torch.randn(1, 3, *self.input_shape, device=self.device)
        results = self.forward(test_input)
        
        shapes = {}
        for key, value in results.items():
            if value is not None and hasattr(value, 'shape'):
                shapes[key] = tuple(value.shape)
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if sub_value is not None and hasattr(sub_value, 'shape'):
                        shapes[f"{key}.{sub_key}"] = tuple(sub_value.shape)
        
        return shapes

    def summary(self) -> str:
        """Retourne un résumé du modèle."""
        shapes = self.get_feature_shapes()
        
        summary_str = f"HierarchicalVisionModel Summary:\n"
        summary_str += f"  Input shape: {self.input_shape}\n"
        summary_str += f"  Use color: {self.use_color}\n"
        summary_str += f"  Use motion: {self.use_motion}\n"
        summary_str += f"  N orientations: {self.n_orientations}\n\n"
        
        summary_str += "Feature shapes:\n"
        for key, shape in shapes.items():
            summary_str += f"  {key}: {shape}\n"
        
        # Compter les paramètres
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        summary_str += f"\nParameters: {total_params:,} total, {trainable_params:,} trainable\n"
        
        return summary_str

    def forward_with_memory_efficiency(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Version optimisée pour la mémoire."""
        # Nettoyer la mémoire si nécessaire
        torch.cuda.empty_cache() if x.is_cuda else None
        
        # Forward avec gestion de gradient
        with torch.cuda.amp.autocast(enabled=x.is_cuda):
            return self.forward(x)

    def set_eval_mode(self, eval_mode: bool = True):
        """Configure le mode évaluation avec optimisations."""
        if eval_mode:
            self.eval()
            torch.set_grad_enabled(False)
        else:
            self.train()
            torch.set_grad_enabled(True)

    def batch_forward(self, images: List[torch.Tensor], batch_size: int = 8) -> List[Dict[str, torch.Tensor]]:
        """Traite une liste d'images par batch."""
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_tensor = torch.stack(batch, dim=0)
            
            with torch.no_grad():
                batch_results = self.forward(batch_tensor)
            
            # Désemballer les résultats
            for j in range(len(batch)):
                single_result = {}
                for key, value in batch_results.items():
                    if isinstance(value, dict):
                        single_result[key] = {k: v[j] if hasattr(v, '__getitem__') else v 
                                            for k, v in value.items()}
                    elif hasattr(value, '__getitem__'):
                        single_result[key] = value[j]
                    else:
                        single_result[key] = value
                results.append(single_result)
        
        return results

    def visualize_features(self, image: torch.Tensor, save_path: str = None) -> Dict[str, any]:
        """Visualise les features à chaque étape."""
        import matplotlib.pyplot as plt
        
        results = self.forward(image)
        visualizations = {}
        
        # Visualiser les features V1
        if 'v1' in results and results['v1'] is not None:
            v1_data = results['v1']
            if 'orientation_map' in v1_data:
                fig, axes = plt.subplots(1, self.n_orientations, figsize=(20, 4))
                for i in range(self.n_orientations):
                    axes[i].imshow(v1_data['orientation_map'][0, i].cpu().detach().numpy(), cmap='viridis')
                    axes[i].set_title(f'Orientation {i}')
                    axes[i].axis('off')
                visualizations['v1_orientations'] = fig
        
        # Visualiser les features V2
        if 'v2' in results and results['v2'] is not None:
            v2_data = results['v2']
            if 'combined' in v2_data and v2_data['combined'].shape[1] >= 3:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                titles = ['Contour', 'Angle', 'Junction']
                for i, title in enumerate(titles):
                    if i < v2_data['combined'].shape[1]:
                        axes[i].imshow(v2_data['combined'][0, i].cpu().detach().numpy(), cmap='hot')
                        axes[i].set_title(title)
                        axes[i].axis('off')
                visualizations['v2_features'] = fig
        
        # Sauvegarder si nécessaire
        if save_path:
            for name, fig in visualizations.items():
                fig.savefig(f"{save_path}/{name}.png", dpi=150, bbox_inches='tight')
                plt.close(fig)
        
        return visualizations

    def save_checkpoint(self, path: str, optimizer=None, epoch: int = 0):
        """Sauvegarde un checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': {
                'input_shape': self.input_shape,
                'use_color': self.use_color,
                'use_motion': self.use_motion,
                'n_orientations': self.n_orientations
            },
            'epoch': epoch
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, path)
        print(f"Checkpoint sauvegardé: {path}")

    @classmethod
    def load_checkpoint(cls, path: str, device: str = 'cpu'):
        """Charge un checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        # Recréer le modèle
        model = cls(
            input_shape=config['input_shape'],
            use_color=config['use_color'],
            use_motion=config['use_motion'],
            n_orientations=config['n_orientations'],
            device=device
        )
        
        # Charger les poids
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        print(f"Checkpoint chargé depuis: {path}")
        print(f"Époch: {checkpoint.get('epoch', 'N/A')}")
        
        return model
        

class WhatWherePathways(nn.Module):
    """
    Séparation des voies ventrale (quoi) et dorsale (où).
    Modèle des deux voies visuelles.
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int],
                 input_channels: int = 1,
                 ventral_features: int = 256,
                 dorsal_features: int = 128,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.input_shape = input_shape
        self.input_channels = input_channels
        self.ventral_features = ventral_features
        self.dorsal_features = dorsal_features
        self.device = device
        
        # Voie ventrale (quoi): forme, couleur, identité
        print("Initialising ventral pathway (what)...")
        self.ventral_stream = nn.Sequential(
            # V1-V2: Orientation et formes
            nn.Conv2d(input_channels, 16, kernel_size=7, padding=3),  # Simule V1 simple cells
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # V4: Formes complexes
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # TEO/TE: Reconnaissance d'objets
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            
            # Intégration
            nn.Linear(128, ventral_features),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Classificateur ventral (catégories d'objets)
        self.ventral_classifier = nn.Sequential(
            nn.Linear(ventral_features, 64),
            nn.ReLU(),
            nn.Linear(64, 20),  # 20 catégories
            nn.LogSoftmax(dim=1)
        )
        
        # Voie dorsale (où): position, mouvement, espace
        print("Initialising dorsal pathway (where)...")
        self.dorsal_stream = nn.Sequential(
            # V1-MT: Mouvement et position
            nn.Conv2d(input_channels, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            
            # MST: Mouvement complexe
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # LIP/7a: Intégration spatiale
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            
            # Intégration
            nn.Linear(64, dorsal_features),
            nn.ReLU()
        )
        
        # Sorties dorsales: position, mouvement, attention
        self.dorsal_position = nn.Sequential(
            nn.Linear(dorsal_features, 2),  # (x, y)
            nn.Sigmoid()  # Normalisé [0, 1]
        )
        
        self.dorsal_motion = nn.Sequential(
            nn.Linear(dorsal_features, 2),  # (dx, dy)
            nn.Tanh()  # Normalisé [-1, 1]
        )
        
        self.dorsal_attention = nn.Sequential(
            nn.Linear(dorsal_features, dorsal_features // 2),
            nn.ReLU(),
            nn.Linear(dorsal_features // 2, 1),
            nn.Sigmoid()  # Score d'attention [0, 1]
        )
        
        # Intégration entre les voies (pour des tâches complexes)
        self.cross_integration = nn.Sequential(
            nn.Linear(ventral_features + dorsal_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),  # Tâche intégrée
            nn.LogSoftmax(dim=1)
        )
        
        print(f"WhatWherePathways initialisé: ventral={ventral_features}, dorsal={dorsal_features}")
    
    def forward(self, 
                image: torch.Tensor,
                previous_position: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Traite l'image à travers les deux voies.
        
        Args:
            image: Image d'entrée
            previous_position: Position précédente (pour le suivi)
            
        Returns:
            Sorties des deux voies et intégration
        """
        if len(image.shape) == 2:
            image = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif len(image.shape) == 3:
            if image.shape[0] == self.input_channels:
                image = image.unsqueeze(0)  # (1, C, H, W)
            elif self.input_channels == 1 and image.shape[0] == 3:
                # RGB, convertir en niveaux de gris (seulement si attendu 1 canal)
                image = 0.299 * image[0:1, :, :] + 0.587 * image[1:2, :, :] + 0.114 * image[2:3, :, :]
                image = image.unsqueeze(0)  # (1, 1, H, W)
            else:
                image = image.unsqueeze(1)  # (B, 1, H, W)
        
        batch_size, channels, height, width = image.shape
        
        # Voie ventrale (quoi)
        ventral_features = self.ventral_stream(image)
        ventral_classification = self.ventral_classifier(ventral_features)
        
        # Voie dorsale (où)
        dorsal_features = self.dorsal_stream(image)
        
        # Prédictions dorsales
        position_pred = self.dorsal_position(dorsal_features)  # Position normalisée
        motion_pred = self.dorsal_motion(dorsal_features)      # Mouvement
        attention_score = self.dorsal_attention(dorsal_features)  # Score d'attention
        
        # Si une position précédente est fournie, mettre à jour
        if previous_position is not None:
            updated_position = previous_position + motion_pred
            updated_position = torch.clamp(updated_position, 0, 1)
        else:
            updated_position = position_pred
        
        # Intégration croisée
        combined_features = torch.cat([ventral_features, dorsal_features], dim=1)
        integrated_prediction = self.cross_integration(combined_features)
        
        return {
            'ventral': {
                'features': ventral_features,
                'classification': ventral_classification,
                'dim': self.ventral_features
            },
            'dorsal': {
                'features': dorsal_features,
                'position': position_pred,
                'motion': motion_pred,
                'attention': attention_score,
                'updated_position': updated_position,
                'dim': self.dorsal_features
            },
            'integrated': {
                'features': combined_features,
                'prediction': integrated_prediction
            },
            'pathway_separation': True
        }


def create_ventral_stream(input_channels: int = 3,
                         feature_dims: List[int] = None,
                         device: str = 'cpu') -> nn.Sequential:
    """
    Crée un modèle simplifié de la voie ventrale.
    
    Args:
        input_channels: Nombre de canaux d'entrée
        feature_dims: Dimensions des caractéristiques à chaque étape
        device: Device
        
    Returns:
        Modèle de la voie ventrale
    """
    if feature_dims is None:
        feature_dims = [16, 32, 64, 128, 256]
    
    layers = []
    in_channels = input_channels
    
    for i, out_channels in enumerate(feature_dims):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        if i < len(feature_dims) - 1:  # Pas de pooling à la dernière couche
            layers.append(nn.MaxPool2d(2))
        
        in_channels = out_channels
    
    # Couche de sortie
    layers.append(nn.AdaptiveAvgPool2d(1))
    layers.append(nn.Flatten())
    
    return nn.Sequential(*layers).to(device)


def create_dorsal_stream(input_channels: int = 3,
                        temporal_steps: int = 3,
                        device: str = 'cpu') -> nn.Module:
    """
    Crée un modèle simplifié de la voie dorsale avec mémoire temporelle.
    
    Args:
        input_channels: Nombre de canaux d'entrée
        temporal_steps: Nombre de pas temporels
        device: Device
        
    Returns:
        Modèle de la voie dorsale
    """
    class DorsalStreamWithMemory(nn.Module):
        def __init__(self):
            super().__init__()
            
            self.temporal_steps = temporal_steps
            
            # Extraction spatiale
            self.spatial_extractor = nn.Sequential(
                nn.Conv2d(input_channels, 16, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            
            # Mémoire temporelle (GRU)
            self.gru = nn.GRU(
                input_size=32 * 7 * 7,  # Après pooling 2x sur 28x28
                hidden_size=128,
                num_layers=2,
                batch_first=True,
                dropout=0.3
            )
            
            # Prédictions
            self.position_head = nn.Linear(128, 2)
            self.motion_head = nn.Linear(128, 2)
            self.attention_head = nn.Linear(128, 1)
            
        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            """x: (B, T, C, H, W) ou (T, C, H, W)"""
            if len(x.shape) == 4:
                x = x.unsqueeze(0)  # (1, T, C, H, W)
            
            batch_size, time_steps, channels, height, width = x.shape
            
            # Traiter chaque frame
            spatial_features = []
            for t in range(time_steps):
                frame = x[:, t, :, :, :]
                features = self.spatial_extractor(frame)
                features = features.view(batch_size, -1)  # Flatten
                spatial_features.append(features)
            
            # Stack temporel
            spatial_stacked = torch.stack(spatial_features, dim=1)  # (B, T, features)
            
            # Mémoire temporelle
            gru_out, hidden = self.gru(spatial_stacked)
            last_hidden = gru_out[:, -1, :]  # Dernier pas temporel
            
            # Prédictions
            position = torch.sigmoid(self.position_head(last_hidden))
            motion = torch.tanh(self.motion_head(last_hidden))
            attention = torch.sigmoid(self.attention_head(last_hidden))
            
            return {
                'position': position,
                'motion': motion,
                'attention': attention,
                'hidden_state': hidden,
                'temporal_features': gru_out
            }
    
    return DorsalStreamWithMemory().to(device)


class BioInspiredCortex(nn.Module):
    """
    Cortex bio-inspiré complet intégrant tous les modules.
    """
    
    def __init__(self,
                 retinal_shape: Tuple[int, int],
                 cortical_shape: Tuple[int, int],
                 n_ganglion_cells: int = 100,
                 use_color: bool = True,
                 use_motion: bool = True,
                 n_orientations: int = 8,
                 include_retinotopic_mapping: bool = True,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.retinal_shape = retinal_shape
        self.cortical_shape = cortical_shape
        self.device = device
        
        print("=" * 60)
        print("INITIALISATION DU CORTEX BIO-INSPIRÉ")
        print("=" * 60)
        
        # 1. Rétine (si pas fournie séparément)
        print("1. Chargement du module rétine...")
        try:
            from ..retina import BioInspiredRetina
            self.retina = BioInspiredRetina(
                retinal_shape=retinal_shape,
                cortical_shape=cortical_shape,
                n_ganglion_cells=n_ganglion_cells,
                include_retinotopic_mapping=include_retinotopic_mapping,
                device=device
            )
            print("   ✓ Rétine chargée")
        except ImportError as e:
            print(f"   ✗ Erreur chargement rétine: {e}")
            self.retina = None
        
        # 2. Hiérarchie corticale
        print("2. Construction de la hiérarchie corticale...")
        self.hierarchy = HierarchicalVisionModel(
            input_shape=cortical_shape,
            use_color=use_color,
            use_motion=use_motion,
            n_orientations=n_orientations,
            device=device
        )
        print("   ✓ Hiérarchie corticale construite")
        
        # 3. Voies parallèles
        print("3. Initialisation des voies ventrale/dorsale...")
        
        # Déterminer le nombre de canaux d'entrée pour le cortex
        cortex_input_channels = 32 if self.retina is not None else (3 if use_color else 1)
        
        self.what_where = WhatWherePathways(
            input_shape=cortical_shape,
            input_channels=cortex_input_channels,
            device=device
        )
        print("   ✓ Voies ventrale/dorsale initialisées")
        
        # 4. Attention
        print("4. Initialisation du système d'attention...")
        self.attention = AttentionModel(
            input_shape=cortical_shape,
            feature_channels=32,  # Correspond à la sortie de l'intégration hiérarchique
            device=device
        )
        print("   ✓ Système d'attention initialisé")
        
        # 5. Microcircuits
        print("5. Construction des microcircuits...")
        self.microcircuits = CanonicalMicrocircuit(
            feature_dim=32,
            n_columns=8,
            device=device
        )
        print("   ✓ Microcircuits construits")
        
        # 6. Intégration finale
        print("6. Intégration finale...")
        self.final_integration = nn.Sequential(
            nn.Linear(256 + 128 + 32, 256),  # hierarchy + whatwhere + microcircuits
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),  # Exemple: 10 sorties
            nn.Softmax(dim=1)
        )
        
        print("=" * 60)
        print("CORTEX BIO-INSPIRÉ INITIALISÉ AVEC SUCCÈS")
        print("=" * 60)
    
    def forward(self,
                image: torch.Tensor,
                return_intermediate: bool = False) -> Dict[str, torch.Tensor]:
        """
        Traite une image à travers tout le système.
        
        Args:
            image: Image d'entrée
            return_intermediate: Retourner les résultats intermédiaires
            
        Returns:
            Résultats du traitement
        """
        intermediate_results = {}
        
        # Étape 1: Traitement rétinien
        if self.retina is not None:
            retinal_results = self.retina(image, return_cortical=True)
            cortical_image = retinal_results.get('cortical_representation', image)
            intermediate_results['retina'] = retinal_results
        else:
            cortical_image = image
        
        # Étape 2: Hiérarchie corticale
        hierarchy_results = self.hierarchy(image)
        intermediate_results['hierarchy'] = hierarchy_results
        
        # Étape 3: Voies ventrale/dorsale
        whatwhere_results = self.what_where(cortical_image)
        intermediate_results['whatwhere'] = whatwhere_results
        
        # Étape 4: Attention
        attention_results = self.attention(
            image=cortical_image,
            features=hierarchy_results['integrated_features'].unsqueeze(-1).unsqueeze(-1)
        )
        intermediate_results['attention'] = attention_results
        
        # Étape 5: Microcircuits
        microcircuit_input = hierarchy_results['integrated_features']
        
        microcircuit_results = self.microcircuits(microcircuit_input)
        intermediate_results['microcircuits'] = microcircuit_results
        
        # Étape 6: Intégration finale
        final_features = torch.cat([
            whatwhere_results['ventral']['features'],
            whatwhere_results['dorsal']['features'],
            microcircuit_results['final_output'].view(microcircuit_input.shape[0], -1)[:, :32]
        ], dim=1)
        
        final_output = self.final_integration(final_features)
        
        # Résultats complets
        results = {
            'final_output': final_output,
            'final_features': final_features,
            'retinal_output': cortical_image if self.retina is not None else image,
            'hierarchy_classification': hierarchy_results['classification'],
            'ventral_classification': whatwhere_results['ventral']['classification'],
            'dorsal_position': whatwhere_results['dorsal']['position'],
            'attention_map': attention_results['spatial_attention']['attention_map'],
            'saliency_map': attention_results['saliency']['saliency_map'],
            'n_modules': 6
        }
        
        if return_intermediate:
            results['intermediate'] = intermediate_results
        
        return results
    
    def reset_state(self):
        """Réinitialise l'état de tous les modules."""
        if self.retina is not None:
            self.retina.reset_state()
        
        # Réinitialiser d'autres modules avec état si nécessaire
        print("État du cortex réinitialisé")


class IntegratedVisionSystem(nn.Module):
    """
    Système visuel intégré - Combine rétine et cortex.
    Interface unifiée pour l'expérimentation.
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int],
                 use_retina: bool = True,
                 use_cortex: bool = True,
                 config: Dict = None,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.input_shape = input_shape
        self.use_retina = use_retina
        self.use_cortex = use_cortex
        self.device = device
        
        # Configuration par défaut
        default_config = {
            'retina': {
                'n_ganglion_cells': 100,
                'use_color': True,
                'include_retinotopic_mapping': True
            },
            'cortex': {
                'use_color': True,
                'use_motion': True,
                'n_orientations': 8
            }
        }
        
        if config is not None:
            # Mettre à jour avec la configuration fournie
            for key, value in config.items():
                if key in default_config and isinstance(value, dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value
        
        self.config = default_config
        
        # Modules
        self.vision_modules = nn.ModuleDict()
        
        if use_retina:
            try:
                from ..retina import BioInspiredRetina
                self.vision_modules['retina'] = BioInspiredRetina(
                    retinal_shape=input_shape,
                    cortical_shape=input_shape,  # Même taille pour simplifier
                    **self.config['retina'],
                    device=device
                )
                print("✓ Module rétine chargé")
            except ImportError as e:
                print(f"✗ Erreur chargement rétine: {e}")
                self.use_retina = False
        
        if use_cortex:
            self.vision_modules['cortex'] = BioInspiredCortex(
                retinal_shape=input_shape,
                cortical_shape=input_shape,
                **self.config['cortex'],
                device=device
            )
            print("✓ Module cortex chargé")
        
        # Connexion directe si un seul module est utilisé
        if use_retina and not use_cortex:
            self.forward = self._forward_retina_only
        elif not use_retina and use_cortex:
            self.forward = self._forward_cortex_only
        else:
            self.forward = self._forward_integrated
    
    def _forward_retina_only(self, image: torch.Tensor, **kwargs):
        """Forward pass avec seulement la rétine."""
        return self.vision_modules['retina'](image, **kwargs)
    
    def _forward_cortex_only(self, image: torch.Tensor, **kwargs):
        """Forward pass avec seulement le cortex."""
        return self.vision_modules['cortex'](image, **kwargs)
    
    def _forward_integrated(self, image: torch.Tensor, **kwargs):
        """Forward pass avec rétine et cortex."""
        # Traitement rétinien
        retinal_results = self.vision_modules['retina'](image, return_cortical=True)
        
        # Extraire la représentation corticale
        cortical_image = retinal_results.get('cortical_representation', image)
        
        # Traitement cortical
        cortical_results = self.vision_modules['cortex'](cortical_image, **kwargs)
        
        # Combiner les résultats
        combined_results = {
            'retinal': retinal_results,
            'cortical': cortical_results,
            'final_output': cortical_results['final_output'],
            'processing_stages': ['retina', 'cortex']
        }
        
        return combined_results
    
    def reset_state(self):
        """Réinitialise l'état de tous les modules."""
        for module in self.vision_modules.values():
            if hasattr(module, 'reset_state'):
                module.reset_state()
    
    def get_module_info(self) -> Dict[str, any]:
        """Retourne des informations sur les modules chargés."""
        info = {
            'use_retina': self.use_retina,
            'use_cortex': self.use_cortex,
            'device': self.device,
            'input_shape': self.input_shape,
            'config': self.config,
            'modules_loaded': list(self.vision_modules.keys())
        }
        
        # Ajouter des informations spécifiques à chaque module
        for name, module in self.vision_modules.items():
            info[f'{name}_parameters'] = sum(p.numel() for p in module.parameters())
            info[f'{name}_trainable'] = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        return info
