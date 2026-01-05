import torch
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

print("Test dimensions avec mouvement...")

# Afficher les dimensions calculées
from neurogeomvision.cortex.cortical_models import HierarchicalVisionModel

# Sans couleur, avec mouvement
model = HierarchicalVisionModel(
    input_shape=(64, 64),
    use_color=False,
    use_motion=True,
    device='cpu'
)

# Vérifier la dimension de la première couche linéaire
print(f"Intégration layer input dim: {model.integration[0].in_features}")
print(f"Intégration layer output dim: {model.integration[0].out_features}")

# Test avec image simple
test_image = torch.randn(1, 64, 64)
test_sequence = torch.randn(3, 64, 64)

print("\nTest forward...")
try:
    results = model(test_image, image_sequence=test_sequence)
    print(f"✓ Succès!")
    print(f"  Features: {results['integrated_features'].shape}")
    
    # Vérifier le chemin des features
    print(f"\nFeature shapes:")
    print(f"  V1 response_map: {results['v1']['response_map'].shape}")
    print(f"  V2 combined: {results['v2']['combined'].shape}")
    if results['mt']:
        print(f"  MT direction_strength: {results['mt']['direction_strength'].shape}")
    
except Exception as e:
    print(f"✗ Erreur: {e}")
    import traceback
    traceback.print_exc()
