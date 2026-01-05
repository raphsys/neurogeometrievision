"""
Module learning_plasticity - Plasticité synaptique et apprentissage
VERSION COMPLÈTE avec intégration aux modules existants
"""

from .hebbian import HebbianLearning, CovarianceLearning
from .stdp import STDPLearning, ExponentialSTDP, TripletSTDP
from .bcm import BCMLearning, DynamicBCM
from .oja import OjaLearning, SangerLearning
from .developmental import DevelopmentalLearning, OcularDominance
from .natural_statistics import NaturalStatistics, ICA_Learning, SparseCoding
from .integration import PlasticityIntegrator, learn_gabor_filters, learn_association_field

__all__ = [
    # Hebbian
    'HebbianLearning', 'CovarianceLearning',
    
    # STDP
    'STDPLearning', 'ExponentialSTDP', 'TripletSTDP',
    
    # BCM
    'BCMLearning', 'DynamicBCM',
    
    # Oja
    'OjaLearning', 'SangerLearning',
    
    # Developmental
    'DevelopmentalLearning', 'OcularDominance',
    
    # Natural Statistics
    'NaturalStatistics', 'ICA_Learning', 'SparseCoding',
    
    # Integration
    'PlasticityIntegrator', 'learn_gabor_filters', 'learn_association_field'
]
