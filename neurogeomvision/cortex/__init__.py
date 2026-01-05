"""
Module cortex - Modélisation du cortex visuel
Aires corticales, sélectivité à l'orientation, détection du mouvement, attention
"""

from .cortical_areas import (
    V1SimpleCell, V1ComplexCell,
    V2Cell, V4Cell, MTCell,
    CorticalColumn, Hypercolumn,
    create_cortical_hierarchy
)

from .orientation import (
    OrientationMap, OrientationSelectivity,
    create_orientation_filters, extract_orientation_features
)

from .motion import (
    MotionEnergyFilter, DirectionSelectivity,
    MotionDetector, OpticalFlow
)

from .color import (
    ColorOpponency, DoubleOpponentCell,
    ColorConstancy, ColorProcessingStream
)

from .attention import (
    SaliencyMap, SpatialAttention,
    FeatureBasedAttention, AttentionModel
)

from .microcircuits import (
    CorticalMicrocircuit, CanonicalMicrocircuit,
    ExcitatoryInhibitoryBalance, LayerSpecificProcessing
)

from .cortical_models import (
    HierarchicalVisionModel, WhatWherePathways,
    create_ventral_stream, create_dorsal_stream,
    BioInspiredCortex, IntegratedVisionSystem
)

__all__ = [
    # Cortical areas
    'V1SimpleCell', 'V1ComplexCell',
    'V2Cell', 'V4Cell', 'MTCell',
    'CorticalColumn', 'Hypercolumn',
    'create_cortical_hierarchy',
    
    # Orientation
    'OrientationMap', 'OrientationSelectivity',
    'create_orientation_filters', 'extract_orientation_features',
    
    # Motion
    'MotionEnergyFilter', 'DirectionSelectivity',
    'MotionDetector', 'OpticalFlow',
    
    # Color
    'ColorOpponency', 'DoubleOpponentCell',
    'ColorConstancy', 'ColorProcessingStream',
    
    # Attention
    'SaliencyMap', 'SpatialAttention',
    'FeatureBasedAttention', 'AttentionModel',
    
    # Microcircuits
    'CorticalMicrocircuit', 'CanonicalMicrocircuit',
    'ExcitatoryInhibitoryBalance', 'LayerSpecificProcessing',
    
    # Models
    'HierarchicalVisionModel', 'WhatWherePathways',
    'create_ventral_stream', 'create_dorsal_stream',
    'BioInspiredCortex', 'IntegratedVisionSystem'
]

print("Module cortex chargé - Modélisation du cortex visuel")
