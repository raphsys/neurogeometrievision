import torch
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

print("Test complet avec RGB...")

# D'abord tester ColorProcessingStream seul
print("\n1. Test ColorProcessingStream...")
try:
    from neurogeomvision.cortex.color import ColorProcessingStream
    
    model = ColorProcessingStream(input_shape=(64, 64), device='cpu')
    test_rgb = torch.randn(1, 3, 64, 64)
    results = model(test_rgb)
    print(f"  ✓ ColorProcessingStream fonctionne!")
    print(f"  Color features: {results['color_features'].shape}")
except Exception as e:
    print(f"  ✗ Erreur: {e}")

# Ensuite tester HierarchicalVisionModel avec couleur
print("\n2. Test HierarchicalVisionModel avec couleur...")
try:
    from neurogeomvision.cortex.cortical_models import HierarchicalVisionModel
    
    model = HierarchicalVisionModel(
        input_shape=(64, 64),
        use_color=True,
        use_motion=False,
        device='cpu'
    )
    
    # Test avec batch
    test_rgb = torch.randn(2, 3, 64, 64)
    results = model(test_rgb)
    print(f"  ✓ HierarchicalVisionModel avec couleur fonctionne!")
    print(f"  Features intégrés: {results['integrated_features'].shape}")
    print(f"  Classification: {results['classification'].shape}")
    
    # Vérifier les dimensions intermédiaires
    print(f"\n  Dimensions intermédiaires:")
    print(f"    V1 response_map: {results['v1']['response_map'].shape}")
    print(f"    V2 combined: {results['v2']['combined'].shape}")
    if results['v4']:
        print(f"    V4 présente: Oui")
        if 'color' in results['v4']:
            print(f"    V4 color features: {results['v4']['color']['color_features'].shape}")
    
except Exception as e:
    print(f"  ✗ Erreur: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("Tests RGB terminés")
print("="*80)
