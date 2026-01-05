import torch
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from neurogeomvision.cortex.cortical_models import HierarchicalVisionModel

print("Test V4 features avec image grise...")
try:
    model = HierarchicalVisionModel(
        input_shape=(64, 64),
        use_color=True,  # Activer couleur
        use_motion=False,
        device='cpu'
    )
    
    # Image en niveaux de gris (1 canal)
    test_image = torch.randn(1, 64, 64)
    
    # Simuler des features V1
    v1_features = {
        'response_map': torch.randn(1, 8, 64, 64),
        'orientation_map': torch.randn(1, 64, 64)
    }
    
    v4_results = model.extract_v4_features(test_image, v1_features)
    print(f"✓ V4 features extraites avec image grise!")
    print(f"  Couleur features: {'color' in v4_results}")
    print(f"  Courbe: {v4_results.get('curve', torch.tensor([])).shape}")
    print(f"  Spirale: {v4_results.get('spiral', torch.tensor([])).shape}")
    
    # Test avec image RGB
    print("\nTest V4 features avec image RGB...")
    test_rgb = torch.randn(3, 64, 64)
    v4_results_rgb = model.extract_v4_features(test_rgb, v1_features)
    print(f"✓ V4 features extraites avec image RGB!")
    
except Exception as e:
    print(f"✗ Erreur: {e}")
    import traceback
    traceback.print_exc()
