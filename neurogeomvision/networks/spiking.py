import torch
import torch.nn as nn

class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire Neuron.
    """
    def __init__(self, tau: float = 1.0, v_threshold: float = 1.0, v_reset: float = 0.0):
        super().__init__()
        self.tau = tau
        self.v_th = v_threshold
        self.v_reset = v_reset

    def forward(self, x: torch.Tensor, v_mem: torch.Tensor) -> torch.Tensor:
        # Simple Euler integration
        v_mem = v_mem + (x - v_mem) / self.tau
        spike = (v_mem > self.v_th).float()
        v_mem = v_mem * (1.0 - spike) + self.v_reset * spike
        return spike, v_mem

class IzhikevichNeuron(nn.Module):
    """
    Izhikevich Neuron Model.
    """
    def __init__(self, a=0.02, b=0.2, c=-65, d=8):
        super().__init__()
        self.params = (a, b, c, d)
        
    def forward(self, i_in: torch.Tensor, state: tuple) -> tuple:
        v, u = state
        a, b, c, d = self.params
        
        # Dynamics
        dv = 0.04*v**2 + 5*v + 140 - u + i_in
        du = a*(b*v - u)
        
        v_next = v + dv
        u_next = u + du
        
        spike = (v_next >= 30).float()
        
        # Reset
        v_next = torch.where(spike > 0.5, torch.tensor(c, device=v.device), v_next)
        u_next = torch.where(spike > 0.5, u_next + d, u_next)
        
        return spike, (v_next, u_next)