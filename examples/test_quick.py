"""
Test rapide du module rétine corrigé.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

from neurogeomvision.retina_lgn.filters import apply_dog_filters

# Test simple
print("Test du filtre DoG corrigé...")

image = torch.randn(64, 64)
print(f"Image shape: {image.shape}")

try:
    filtered = apply_dog_filters(image)
    print(f"✓ Succès! Filtered shape: {filtered.shape}")
    print(f"  Min: {filtered.min():.3f}, Max: {filtered.max():.3f}")
    print(f"  Mean: {filtered.mean():.3f}, Std: {filtered.std():.3f}")
except Exception as e:
    print(f"✗ Erreur: {e}")
    import traceback
    traceback.print_exc()
