"""
Module microcircuits.py - Microcircuits corticaux
Colonnes corticales, circuits excitation/inhibition, traitement par couches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import math


class CorticalMicrocircuit(nn.Module):
    """
    Microcircuit cortical de base - Modélise les interactions dans une colonne corticale.
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_layers: int = 6,  # Correspond aux couches corticales I-VI
                 ei_ratio: float = 4.0,  # Ratio excitation/inhibition
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.ei_ratio = ei_ratio
        self.device = device
        
        # Populations neuronales par couche
        self.excitatory = nn.ModuleList()
        self.inhibitory = nn.ModuleList()
        
        # Connexions entre couches
        self.ff_exc = nn.ModuleList()  # Feedforward excitation
        self.ff_inh = nn.ModuleList()  # Feedforward inhibition
        self.fb_exc = nn.ModuleList()  # Feedback excitation
        self.fb_inh = nn.ModuleList()  # Feedback inhibition
        self.lat_exc = nn.ModuleList()  # Latéral excitation
        self.lat_inh = nn.ModuleList()  # Latéral inhibition
        
        # Initialiser les populations et connexions
        for layer in range(n_layers):
            # Populations
            exc_pop = self._create_population(hidden_size, 'excitatory')
            inh_pop = self._create_population(int(hidden_size / ei_ratio), 'inhibitory')
            
            self.excitatory.append(exc_pop)
            self.inhibitory.append(inh_pop)
            
            # Connexions feedforward (de la couche inférieure à supérieure)
            if layer > 0:
                ff_exc_conn = nn.Linear(hidden_size, hidden_size)
                ff_inh_conn = nn.Linear(int(hidden_size / ei_ratio), int(hidden_size / ei_ratio))
                self.ff_exc.append(ff_exc_conn)
                self.ff_inh.append(ff_inh_conn)
            
            # Connexions feedback (de la couche supérieure à inférieure)
            if layer < n_layers - 1:
                fb_exc_conn = nn.Linear(hidden_size, hidden_size)
                fb_inh_conn = nn.Linear(int(hidden_size / ei_ratio), int(hidden_size / ei_ratio))
                self.fb_exc.append(fb_exc_conn)
                self.fb_inh.append(fb_inh_conn)
            
            # Connexions latérales (dans la même couche)
            lat_exc_conn = nn.Linear(hidden_size, hidden_size)
            lat_inh_conn = nn.Linear(int(hidden_size / ei_ratio), int(hidden_size / ei_ratio))
            self.lat_exc.append(lat_exc_conn)
            self.lat_inh.append(lat_inh_conn)
        
        # Connexion d'entrée (couche 4 principalement)
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Paramètres de dynamique
        self.tau_exc = nn.Parameter(torch.tensor(10.0, device=device))  # Constante de temps excitation
        self.tau_inh = nn.Parameter(torch.tensor(5.0, device=device))   # Constante de temps inhibition
        self.threshold = nn.Parameter(torch.tensor(0.5, device=device)) # Seuil de déclenchement
        
        # État
        self.reset_state()
    
    def _create_population(self, size: int, pop_type: str) -> nn.Module:
        """Crée une population neuronale."""
        if pop_type == 'excitatory':
            # Neurones excitateurs (pyramidaux)
            return nn.Sequential(
                nn.Linear(size, size),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        else:  # 'inhibitory'
            # Neurones inhibiteurs (interneurones)
            return nn.Sequential(
                nn.Linear(size, size),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
    
    def reset_state(self):
        """Réinitialise l'état du circuit."""
        self.exc_activity = [torch.zeros(self.hidden_size, device=self.device) 
                            for _ in range(self.n_layers)]
        self.inh_activity = [torch.zeros(int(self.hidden_size / self.ei_ratio), device=self.device) 
                            for _ in range(self.n_layers)]
        
        self.exc_current = [torch.zeros(self.hidden_size, device=self.device) 
                           for _ in range(self.n_layers)]
        self.inh_current = [torch.zeros(int(self.hidden_size / self.ei_ratio), device=self.device) 
                           for _ in range(self.n_layers)]
    
    def forward(self, 
                input_signal: torch.Tensor,
                dt: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Exécute une étape de simulation du microcircuit.
        
        Args:
            input_signal: Signal d'entrée
            dt: Pas de temps
            
        Returns:
            Activité de toutes les couches
        """
        if len(input_signal.shape) > 1:
            input_signal = input_signal.flatten()
        
        # Projeter l'entrée sur la couche 4 (entrée principale dans le cortex)
        input_layer = 3  # Couche 4 (index 0-based)
        input_current = self.input_projection(input_signal)
        
        # Mettre à jour les courants d'entrée
        self.exc_current[input_layer] = self.exc_current[input_layer] + input_current
        
        # Mettre à jour toutes les couches
        new_exc_activity = []
        new_inh_activity = []
        
        for layer in range(self.n_layers):
            # Somme des courants pour cette couche
            total_exc_current = self.exc_current[layer].clone()
            total_inh_current = self.inh_current[layer].clone()
            
            # Connexions feedforward (des couches inférieures)
            if layer > 0:
                # Excitation feedforward
                ff_exc_input = self.ff_exc[layer-1](self.exc_activity[layer-1])
                total_exc_current = total_exc_current + ff_exc_input
                
                # Inhibition feedforward
                ff_inh_input = self.ff_inh[layer-1](self.inh_activity[layer-1])
                total_inh_current = total_inh_current + ff_inh_input
            
            # Connexions feedback (des couches supérieures)
            if layer < self.n_layers - 1:
                # Excitation feedback
                fb_exc_input = self.fb_exc[layer](self.exc_activity[layer+1])
                total_exc_current = total_exc_current + fb_exc_input
                
                # Inhibition feedback
                fb_inh_input = self.fb_inh[layer](self.inh_activity[layer+1])
                total_inh_current = total_inh_current + fb_inh_input
            
            # Connexions latérales (même couche)
            lat_exc_input = self.lat_exc[layer](self.exc_activity[layer])
            lat_inh_input = self.lat_inh[layer](self.inh_activity[layer])
            
            total_exc_current = total_exc_current + lat_exc_input
            total_inh_current = total_inh_current + lat_inh_input
            
            # Inhibition récurrente (des inhibiteurs locaux aux excitateurs)
            recurrent_inhibition = self.inh_activity[layer].sum() * 0.1
            total_exc_current = total_exc_current - recurrent_inhibition
            
            # Mettre à jour l'activité avec la dynamique
            # Équation différentielle: tau * dx/dt = -x + I
            alpha_exc = math.exp(-dt / self.tau_exc.item())
            alpha_inh = math.exp(-dt / self.tau_inh.item())
            
            new_exc = alpha_exc * self.exc_activity[layer] + (1 - alpha_exc) * total_exc_current
            new_inh = alpha_inh * self.inh_activity[layer] + (1 - alpha_inh) * total_inh_current
            
            # Non-linéarité
            new_exc = F.relu(new_exc - self.threshold)
            new_inh = F.relu(new_inh - self.threshold)
            
            # Normalisation
            new_exc = new_exc / (new_exc.norm() + 1e-8)
            new_inh = new_inh / (new_inh.norm() + 1e-8)
            
            # Passer à travers les populations
            new_exc = self.excitatory[layer](new_exc)
            new_inh = self.inhibitory[layer](new_inh)
            
            new_exc_activity.append(new_exc)
            new_inh_activity.append(new_inh)
        
        # Mettre à jour l'état
        self.exc_activity = new_exc_activity
        self.inh_activity = new_inh_activity
        
        # Réinitialiser les courants pour la prochaine étape
        for layer in range(self.n_layers):
            self.exc_current[layer] = torch.zeros_like(self.exc_current[layer])
            self.inh_current[layer] = torch.zeros_like(self.inh_current[layer])
        
        return {
            'exc_activity': self.exc_activity,
            'inh_activity': self.inh_activity,
            'layer_outputs': self.exc_activity,  # Sortie = activité excitatrice
            'n_layers': self.n_layers,
            'ei_ratio': self.ei_ratio
        }


class CanonicalMicrocircuit(nn.Module):
    """
    Circuit canonique - Modèle standard des interactions corticales.
    Based on Douglas & Martin 2004.
    """
    
    def __init__(self,
                 feature_dim: int,
                 n_columns: int = 8,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.feature_dim = feature_dim
        self.n_columns = n_columns
        self.device = device
        
        # Colonnes corticales
        self.columns = nn.ModuleList([
            CorticalMicrocircuit(
                input_size=feature_dim,
                hidden_size=feature_dim * 2,
                n_layers=4,  # Simplifié: L2/3, L4, L5, L6
                device=device
            )
            for _ in range(n_columns)
        ])
        
        # Connexions entre colonnes (latérales)
        self.lateral_connections = nn.ModuleList()
        for i in range(n_columns):
            col_connections = nn.ModuleList()
            for j in range(n_columns):
                if i != j:
                    # Connexion excitatrice entre colonnes
                    conn = nn.Linear(feature_dim * 2, feature_dim * 2)
                    col_connections.append(conn)
                else:
                    col_connections.append(None)
            self.lateral_connections.append(col_connections)
        
        # Inhibition latérale globale
        self.global_inhibition = nn.Sequential(
            nn.Linear(feature_dim * 2 * n_columns, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim * 2 * n_columns),
            nn.Sigmoid()
        )
    
    def forward(self, 
                features: torch.Tensor,
                dt: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Traite les caractéristiques à travers le circuit canonique.
        
        Args:
            features: Caractéristiques d'entrée (n_columns, feature_dim) ou (feature_dim * n_columns)
            dt: Pas de temps
            
        Returns:
            Sortie du circuit
        """
        # Gérer le cas (1, D) -> (D) pour permettre l'expansion
        if len(features.shape) == 2 and features.shape[0] == 1:
            features = features.squeeze(0)
            
        if len(features.shape) == 1:
            # Remodeler en colonnes
            if features.shape[0] == self.feature_dim * self.n_columns:
                features = features.view(self.n_columns, self.feature_dim)
            else:
                # Répliquer pour toutes les colonnes
                features = features.unsqueeze(0).expand(self.n_columns, -1)
        
        # Traiter chaque colonne
        column_outputs = []
        for i, (column, feature) in enumerate(zip(self.columns, features)):
            col_result = column(feature, dt)
            # Prendre la sortie de la couche supérieure (L2/3)
            col_output = col_result['layer_outputs'][0]  # L2/3
            column_outputs.append(col_output)
        
        # Connexions latérales entre colonnes
        lateral_inputs = []
        for i in range(self.n_columns):
            lateral_sum = torch.zeros_like(column_outputs[i])
            
            for j in range(self.n_columns):
                if i != j and self.lateral_connections[i][j] is not None:
                    lateral_input = self.lateral_connections[i][j](column_outputs[j])
                    lateral_sum = lateral_sum + lateral_input
            
            lateral_inputs.append(lateral_sum)
        
        # Inhibition latérale globale
        all_outputs = torch.cat(column_outputs, dim=0)
        global_inhibition = self.global_inhibition(all_outputs)
        
        # Appliquer l'inhibition globale
        inhibited_outputs = []
        for i in range(self.n_columns):
            inhibited = column_outputs[i] * (1.0 - global_inhibition[i*self.feature_dim*2:(i+1)*self.feature_dim*2])
            inhibited_outputs.append(inhibited)
        
        # Intégration finale
        final_output = torch.cat(inhibited_outputs, dim=0)
        
        return {
            'column_outputs': column_outputs,
            'lateral_inputs': lateral_inputs,
            'global_inhibition': global_inhibition,
            'final_output': final_output,
            'n_columns': self.n_columns
        }


class ExcitatoryInhibitoryBalance(nn.Module):
    """
    Maintien de l'équilibre excitation/inhibition (E/I balance).
    """
    
    def __init__(self,
                 population_size: int,
                 target_ratio: float = 4.0,
                 adaptation_rate: float = 0.01,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.population_size = population_size
        self.target_ratio = target_ratio
        self.adaptation_rate = adaptation_rate
        self.device = device
        
        # Poids adaptatifs
        self.exc_weights = nn.Parameter(torch.ones(population_size, device=device))
        self.inh_weights = nn.Parameter(torch.ones(population_size, device=device))
        
        # Gain d'échelle
        self.exc_gain = nn.Parameter(torch.ones(1, device=device))
        self.inh_gain = nn.Parameter(torch.ones(1, device=device))
        
        # Suivi de l'activité
        self.register_buffer('exc_activity_avg', torch.zeros(population_size, device=device))
        self.register_buffer('inh_activity_avg', torch.zeros(population_size, device=device))
        
    def update_balance(self,
                      exc_activity: torch.Tensor,
                      inh_activity: torch.Tensor):
        """Met à jour l'équilibre E/I en fonction de l'activité."""
        # Mettre à jour les moyennes d'activité
        alpha = 0.1  # Taux de lissage
        self.exc_activity_avg = alpha * exc_activity + (1 - alpha) * self.exc_activity_avg
        self.inh_activity_avg = alpha * inh_activity + (1 - alpha) * self.inh_activity_avg
        
        # Calculer le ratio E/I actuel
        current_ratio = (self.exc_activity_avg.mean() + 1e-8) / (self.inh_activity_avg.mean() + 1e-8)
        
        # Ajuster les gains pour atteindre le ratio cible
        ratio_error = current_ratio - self.target_ratio
        
        # Ajustement homéostatique
        self.exc_gain.data = self.exc_gain.data * (1.0 - self.adaptation_rate * ratio_error)
        self.inh_gain.data = self.inh_gain.data * (1.0 + self.adaptation_rate * ratio_error)
        
        # Limiter les gains
        self.exc_gain.data = torch.clamp(self.exc_gain.data, 0.1, 10.0)
        self.inh_gain.data = torch.clamp(self.inh_gain.data, 0.1, 10.0)
        
        return current_ratio, ratio_error
    
    def forward(self,
                exc_input: torch.Tensor,
                inh_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applique l'équilibre E/I aux entrées.
        
        Args:
            exc_input: Entrée excitatrice
            inh_input: Entrée inhibitrice
            
        Returns:
            Sorties avec équilibre E/I appliqué
        """
        # Appliquer les poids et gains
        exc_output = exc_input * self.exc_weights * self.exc_gain
        inh_output = inh_input * self.inh_weights * self.inh_gain
        
        # Mettre à jour l'équilibre
        current_ratio, ratio_error = self.update_balance(exc_output, inh_output)
        
        return {
            'exc_output': exc_output,
            'inh_output': inh_output,
            'current_ratio': current_ratio,
            'ratio_error': ratio_error,
            'exc_gain': self.exc_gain,
            'inh_gain': self.inh_gain
        }


class LayerSpecificProcessing(nn.Module):
    """
    Traitement spécifique à chaque couche corticale.
    """
    
    def __init__(self,
                 input_dim: int,
                 layer_config: Dict[str, Dict] = None,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.device = device
        
        # Configuration par défaut des couches corticales
        default_config = {
            'L1': {'size': input_dim // 4, 'function': 'sparse'},
            'L2/3': {'size': input_dim, 'function': 'association'},
            'L4': {'size': input_dim, 'function': 'input'},
            'L5': {'size': input_dim * 2, 'function': 'output'},
            'L6': {'size': input_dim // 2, 'function': 'feedback'}
        }
        
        if layer_config is not None:
            default_config.update(layer_config)
        
        self.layer_config = default_config
        
        # Créer les couches
        self.layers = nn.ModuleDict()
        
        for layer_name, config in self.layer_config.items():
            size = config['size']
            function = config['function']
            
            if function == 'input':
                # Couche d'entrée (L4): traitement linéaire
                layer = nn.Sequential(
                    nn.Linear(input_dim, size),
                    nn.ReLU(),
                    nn.LayerNorm(size)
                )
            elif function == 'association':
                # Couche d'association (L2/3): traitement non-linéaire
                layer = nn.Sequential(
                    nn.Linear(input_dim, size),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(size, size),
                    nn.ReLU(),
                    nn.LayerNorm(size)
                )
            elif function == 'output':
                # Couche de sortie (L5): intégration
                layer = nn.Sequential(
                    nn.Linear(input_dim, size),
                    nn.ReLU(),
                    nn.Linear(size, size),
                    nn.Tanh(),  # Pour la sortie bornée
                    nn.LayerNorm(size)
                )
            elif function == 'feedback':
                # Couche de feedback (L6): traitement récurrent
                layer = nn.Sequential(
                    nn.Linear(input_dim, size),
                    nn.ReLU(),
                    nn.Linear(size, input_dim),  # Projette de retour vers l'entrée
                    nn.Sigmoid()
                )
            elif function == 'sparse':
                # Couche sparse (L1): représentation éparse
                layer = nn.Sequential(
                    nn.Linear(input_dim, size),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(size, size),
                    nn.Sigmoid()  # Pour une activité sparse
                )
            else:
                raise ValueError(f"Fonction de couche inconnue: {function}")
            
            self.layers[layer_name] = layer
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Traite l'entrée à travers les couches corticales.
        
        Args:
            x: Entrée
            
        Returns:
            Sorties de toutes les couches
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        # Traitement séquentiel selon la hiérarchie corticale
        # L4 d'abord (entrée), puis L2/3, L5, L6, L1
        layer_order = ['L4', 'L2/3', 'L5', 'L6', 'L1']
        
        outputs = {}
        current = x
        
        for layer_name in layer_order:
            if layer_name in self.layers:
                layer_output = self.layers[layer_name](current)
                outputs[layer_name] = layer_output
                
                # Pour L6, le feedback est ajouté à l'entrée
                if layer_name == 'L6':
                    feedback = layer_output
                    if feedback.shape == x.shape:
                        current = x + feedback  # Feedback vers l'entrée
                    else:
                        current = feedback
                else:
                    current = layer_output
        
        return {
            'layer_outputs': outputs,
            'final_output': current,
            'layer_order': layer_order,
            'config': self.layer_config
        }
