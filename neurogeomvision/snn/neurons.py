"""
Module neurons.py - Modèles de neurones à impulsions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional
import math


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire Neuron (LIF)
    """
    
    def __init__(self,
                 tau_m: float = 20.0,        # Constante de temps membranaire (ms)
                 v_rest: float = -65.0,      # Potentiel de repos (mV)
                 v_thresh: float = -50.0,    # Seuil de déclenchement (mV)
                 v_reset: float = -65.0,     # Potentiel de réinitialisation (mV)
                 dt: float = 1.0,           # Pas de temps (ms)
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.dt = dt
        self.device = device
        
        # État
        self.register_buffer('voltage', torch.tensor(v_rest, dtype=torch.float32, device=device))
    
    def reset_state(self):
        """Réinitialise l'état."""
        self.voltage = torch.tensor(self.v_rest, dtype=torch.float32, device=self.device)
    
    def forward(self, current: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Un pas de temps du neurone LIF.
        
        Args:
            current: Courant d'entrée (nA)
            
        Returns:
            spike: 1 si spike, 0 sinon
            voltage: Potentiel membranaire
        """
        # Intégration leaky
        dv = (-(self.voltage - self.v_rest) + current) / self.tau_m
        self.voltage = self.voltage + dv * self.dt
        
        # Vérification du seuil
        if self.voltage > self.v_thresh:
            spike = torch.tensor(1.0, device=self.device)
            self.voltage = torch.tensor(self.v_reset, device=self.device)
        else:
            spike = torch.tensor(0.0, device=self.device)
        
        return spike, self.voltage
    
    def simulate(self, current_input: List[float], duration: float = None) -> Dict:
        """Simule sur plusieurs pas de temps."""
        if duration is not None:
            n_steps = int(duration / self.dt)
        else:
            n_steps = len(current_input)
        
        self.reset_state()
        
        voltages = []
        spikes = []
        
        for t in range(n_steps):
            if t < len(current_input):
                current = torch.tensor(current_input[t], dtype=torch.float32, device=self.device)
            else:
                current = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            
            spike, voltage = self.forward(current)
            spikes.append(spike.item())
            voltages.append(voltage.item())
        
        return {
            'voltages': voltages,
            'spikes': spikes,
            'times': [t * self.dt for t in range(n_steps)],
            'spike_times': [t * self.dt for t, s in enumerate(spikes) if s > 0.5]
        }


class IzhikevichNeuron(nn.Module):
    """
    Neurone d'Izhikevich (2003)
    """
    
    def __init__(self,
                 a: float = 0.02,
                 b: float = 0.2,
                 c: float = -65.0,
                 d: float = 2.0,
                 dt: float = 1.0,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.dt = dt
        self.device = device
        
        # État
        self.register_buffer('v', torch.tensor(c, dtype=torch.float32, device=device))
        self.register_buffer('u', torch.tensor(b * c, dtype=torch.float32, device=device))
    
    def reset_state(self):
        """Réinitialise l'état."""
        self.v = torch.tensor(self.c, dtype=torch.float32, device=self.device)
        self.u = torch.tensor(self.b * self.c, dtype=torch.float32, device=self.device)
    
    def forward(self, I: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Un pas de temps."""
        if I.dtype != torch.float32:
            I = I.float()
        
        # Équations d'Izhikevich
        dv = 0.04 * self.v**2 + 5 * self.v + 140 - self.u + I
        du = self.a * (self.b * self.v - self.u)
        
        self.v = self.v + dv * self.dt
        self.u = self.u + du * self.dt
        
        # Condition de spike
        if self.v >= 30:
            spike = torch.tensor(1.0, device=self.device)
            self.v = torch.tensor(self.c, dtype=torch.float32, device=self.device)
            self.u = self.u + self.d
        else:
            spike = torch.tensor(0.0, device=self.device)
        
        return spike, self.v


class LIFLayer(nn.Module):
    """
    Couche de neurones LIF.
    """
    
    def __init__(self,
                 n_neurons: int,
                 tau_m: float = 20.0,
                 v_thresh: float = 1.0,
                 v_reset: float = 0.0,
                 device: str = 'cpu'):
        
        super().__init__()
        
        self.n_neurons = n_neurons
        self.tau_m = tau_m
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.device = device
        
        # États
        self.register_buffer('voltages', torch.zeros(n_neurons, dtype=torch.float32, device=device))
        self.register_buffer('spikes', torch.zeros(n_neurons, dtype=torch.float32, device=device))
    
    def reset_state(self):
        """Réinitialise l'état."""
        self.voltages = torch.zeros(self.n_neurons, dtype=torch.float32, device=self.device)
        self.spikes = torch.zeros(self.n_neurons, dtype=torch.float32, device=self.device)
    
    def forward(self, currents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            currents: Courants d'entrée (n_neurons,) ou (batch_size, n_neurons)
            
        Returns:
            spikes: Sorties binaires
            voltages: Potentiels
        """
        if len(currents.shape) == 1:
            currents = currents.unsqueeze(0)
        
        if currents.dtype != torch.float32:
            currents = currents.float()
        
        batch_size = currents.shape[0]
        
        # Intégration LIF
        alpha = math.exp(-1.0 / self.tau_m)
        beta = 1.0 - alpha
        
        batch_spikes = []
        batch_voltages = []
        
        for b in range(batch_size):
            current_batch = currents[b]
            self.voltages = alpha * self.voltages + beta * current_batch
            
            # Génération de spikes
            self.spikes = (self.voltages >= self.v_thresh).float()
            
            # Réinitialisation
            self.voltages = self.voltages * (1 - self.spikes) + self.v_reset * self.spikes
            
            batch_spikes.append(self.spikes.clone())
            batch_voltages.append(self.voltages.clone())
        
        if batch_size == 1:
            return batch_spikes[0], batch_voltages[0]
        else:
            return torch.stack(batch_spikes), torch.stack(batch_voltages)
