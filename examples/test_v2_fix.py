import torch
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from neurogeomvision.cortex.cortical_models import HierarchicalVisionModel

print("Test V2 features...")
try:
    model = HierarchicalVisionModel(
        input_shape=(64, 64),
        use_color=False,
        use_motion=False,
        device='cpu'
    )
    
    # Simuler des features V1
    v1_features = {
        'response_map': torch.randn(1, 8, 64, 64)
    }
    
    v2_results = model.extract_v2_features(v1_features)
    print(f"✓ V2 features extraites!")
    print(f"  Contour: {v2_results['contour'].shape}")
    print(f"  Angle: {v2_results['angle'].shape}")
    print(f"  Junction: {v2_results['junction'].shape}")
    print(f"  Combined: {v2_results['combined'].shape}")
except Exception as e:
    print(f"✗ Erreur: {e}")
    import traceback
    traceback.print_exc()
