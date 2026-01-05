import torch
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

print("Test HierarchicalVisionModel avec RGB...")

try:
    from neurogeomvision.cortex.cortical_models import HierarchicalVisionModel
    
    model = HierarchicalVisionModel(
        input_shape=(64, 64),
        use_color=True,
        use_motion=False,
        device='cpu'
    )
    
    test_rgb = torch.randn(3, 64, 64)
    print(f"Input: {test_rgb.shape}")
    
    results = model(test_rgb)
    print(f"✓ Succès avec RGB!")
    print(f"  Features intégrés: {results['integrated_features'].shape}")
    print(f"  Classification: {results['classification'].shape}")
    
except Exception as e:
    print(f"✗ Erreur: {e}")
    import traceback
    traceback.print_exc()
