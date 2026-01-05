import torch
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from neurogeomvision.cortex.attention import SaliencyMap

print("Test SaliencyMap...")
try:
    model = SaliencyMap(device='cpu')
    test_image = torch.randn(1, 3, 64, 64)
    results = model(test_image)
    print(f"✓ SaliencyMap fonctionne! Carte: {results['saliency_map'].shape}")
except Exception as e:
    print(f"✗ Erreur: {e}")
