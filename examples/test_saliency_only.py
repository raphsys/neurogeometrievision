"""
Test spécifique du module saliency.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

print("="*80)
print("TEST SPÉCIFIQUE DU MODULE SALIENCY")
print("="*80)

def test_saliency():
    """Test de la carte de saillance."""
    try:
        from neurogeomvision.cortex.attention import SaliencyMap
        
        print("✓ Import réussi")
        
        device = 'cpu'
        
        # Test 1: SaliencyMap avec image simple
        print("\n1. Test SaliencyMap avec image grise...")
        model = SaliencyMap(device=device)
        
        # Image grise
        test_image = torch.randn(1, 32, 32, device=device)
        results = model(test_image.unsqueeze(0))
        print(f"  Carte de saillance: {results['saliency_map'].shape}")
        
        # Test 2: SaliencyMap avec image RGB
        print("\n2. Test SaliencyMap avec image RGB...")
        test_rgb = torch.randn(3, 64, 64, device=device)
        results = model(test_rgb.unsqueeze(0))
        print(f"  Carte de saillance: {results['saliency_map'].shape}")
        
        # Vérifier les cartes de caractéristiques
        if 'feature_maps' in results:
            print(f"  Cartes de caractéristiques: {len(results['feature_maps'])}")
            for name, feature_map in results['feature_maps'].items():
                if hasattr(feature_map, 'shape'):
                    print(f"    {name}: {feature_map.shape}")
        
        print("\n" + "="*80)
        print("✅ TESTS SALIENCY PASSÉS AVEC SUCCÈS !")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_saliency()
    
    if success:
        print("\n🎉 Module saliency fonctionnel !")
    else:
        print("\n⚠️  Problèmes détectés dans le module saliency")
