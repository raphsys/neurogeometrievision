"""
Module illusory_contours - Complétion modale et contours illusoires
"""

from .kanizsa import KanizsaTriangle, KanizsaSquare
from .modal_completion import ModalCompletion
from .visual_illusions import EhrensteinIllusion, PetterEffect

__all__ = [
    'KanizsaTriangle',
    'KanizsaSquare',
    'ModalCompletion',
    'EhrensteinIllusion',
    'PetterEffect'
]
