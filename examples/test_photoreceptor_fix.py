"""
Test spécifique pour la correction de PhotoreceptorLayer.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

print("Test spécifique PhotoreceptorLayer")
print("="*60)

try:
    from neurogeomvision.retina import PhotoreceptorLayer
    
    # Créer une petite couche
    layer = PhotoreceptorLayer(mosaic_shape=(8, 8), device='cpu')
    print("✓ PhotoreceptorLayer créée")
    
    # Test avec une image
    image = torch.randn(1, 8, 8, dtype=torch.float32)
    print(f"Image shape: {image.shape}")
    
    # Forward pass
    response = layer(image)
    print(f"✓ Forward pass réussi")
    print(f"Response shape: {response.shape}")
    print(f"Response min/max: {response.min():.3f}, {response.max():.3f}")
    
    # Test reset_state
    layer.reset_state()
    print("✓ reset_state fonctionne")
    
    # Test avec image 2D (sans canal)
    image_2d = torch.randn(8, 8, dtype=torch.float32)
    response_2d = layer(image_2d)
    print(f"✓ Image 2D traitée, shape: {response_2d.shape}")
    
    print("\n" + "="*60)
    print("✅ PHOTORECEPTORLAYER FONCTIONNEL !")
    
except Exception as e:
    print(f"✗ Erreur: {e}")
    import traceback
    traceback.print_exc()
