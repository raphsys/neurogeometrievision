import torch
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

print("Test HierarchicalVisionModel...")

# Test 1: Avec couleur (nécessite RGB)
print("\n1. Test avec couleur (RGB)...")
try:
    from neurogeomvision.cortex.cortical_models import HierarchicalVisionModel
    
    model = HierarchicalVisionModel(
        input_shape=(64, 64),
        use_color=True,
        use_motion=False,
        device='cpu'
    )
    
    test_rgb = torch.randn(3, 64, 64)
    results = model(test_rgb)
    print(f"✓ RGB: Features intégrés: {results['integrated_features'].shape}")
    print(f"✓ RGB: Classification: {results['classification'].shape}")
except Exception as e:
    print(f"✗ Erreur RGB: {e}")

# Test 2: Sans couleur (image grise)
print("\n2. Test sans couleur (niveaux de gris)...")
try:
    model = HierarchicalVisionModel(
        input_shape=(64, 64),
        use_color=False,
        use_motion=False,
        device='cpu'
    )
    
    test_gray = torch.randn(1, 64, 64)
    results = model(test_gray)
    print(f"✓ Gris: Features intégrés: {results['integrated_features'].shape}")
    print(f"✓ Gris: Classification: {results['classification'].shape}")
except Exception as e:
    print(f"✗ Erreur gris: {e}")

# Test 3: Avec mouvement
print("\n3. Test avec mouvement...")
try:
    model = HierarchicalVisionModel(
        input_shape=(64, 64),
        use_color=False,
        use_motion=True,
        device='cpu'
    )
    
    test_image = torch.randn(1, 64, 64)
    # Séquence pour mouvement (3 frames)
    test_sequence = torch.randn(3, 64, 64)
    results = model(test_image, image_sequence=test_sequence)
    print(f"✓ Mouvement: Features intégrés: {results['integrated_features'].shape}")
except Exception as e:
    print(f"✗ Erreur mouvement: {e}")

print("\n" + "="*80)
print("Tests HierarchicalVisionModel terminés")
print("="*80)
