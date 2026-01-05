"""
Module snn - Spiking Neural Networks for NeuroGeomVision
Implémentation complète des réseaux de neurones à impulsions
"""

from .neurons import LIFNeuron, IzhikevichNeuron, LIFLayer
from .layers import SNNLinear, SNNConv2d, TemporalPooling
from .networks import SNNClassifier, SNNVisualEncoder
from .learning import STDP_SNN, SurrogateGradient
from .utils import encode_image_to_spikes, calculate_spike_stats, visualize_spike_train
from .visual_processing import RetinaEncoder

__all__ = [
    # Neurons
    'LIFNeuron', 'IzhikevichNeuron', 'LIFLayer',
    
    # Layers
    'SNNLinear', 'SNNConv2d', 'TemporalPooling',
    
    # Networks
    'SNNClassifier', 'SNNVisualEncoder',
    
    # Learning
    'STDP_SNN', 'SurrogateGradient',
    
    # Utils
    'encode_image_to_spikes', 'calculate_spike_stats', 'visualize_spike_train',
    
    # Visual Processing
    'RetinaEncoder'
]
