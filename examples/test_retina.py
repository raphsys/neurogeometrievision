"""
Test du module retina.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

print("="*80)
print("TEST DU MODULE RETINA")
print("="*80)

def test_imports():
    """Test les imports."""
    print("\n1. TEST DES IMPORTS")
    print("-" * 60)
    
    try:
        import neurogeomvision.retina
        print("✓ Module retina importé")
        
        from neurogeomvision.retina import (
            Cone, Rod, PhotoreceptorLayer,
            HorizontalCell, BipolarCell,
            GanglionCell, ONGanglionCell, OFFGanglionCell,
            RetinotopicMap, SimpleRetinaModel
        )
        
        print("✓ Toutes les classes importées")
        return True
        
    except ImportError as e:
        print(f"✗ Erreur d'import: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_photoreceptors():
    """Test les photorécepteurs."""
    print("\n2. TEST PHOTORÉCEPTEURS")
    print("-" * 60)
    
    try:
        from neurogeomvision.retina import Cone, Rod, PhotoreceptorLayer
        
        # Test cône
        cone = Cone(cone_type='M', device='cpu')
        response = cone(torch.tensor([0.5], dtype=torch.float32))
        print(f"✓ Cône M: réponse = {response.item():.3f}")
        
        # Test bâtonnet
        rod = Rod(device='cpu')
        response = rod(torch.tensor([0.1], dtype=torch.float32))
        print(f"✓ Bâtonnet: réponse = {response.item():.3f}")
        
        # Test couche
        layer = PhotoreceptorLayer(mosaic_shape=(10, 10), device='cpu')
        image = torch.randn(1, 10, 10, dtype=torch.float32)
        response = layer(image)
        print(f"✓ Couche photorécepteurs: shape = {response.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_retina_model():
    """Test le modèle de rétine."""
    print("\n3. TEST MODÈLE DE RÉTINE")
    print("-" * 60)
    
    try:
        from neurogeomvision.retina import SimpleRetinaModel
        
        # Créer modèle
        model = SimpleRetinaModel(
            input_shape=(32, 32),
            n_ganglion_cells=50,
            use_color=False,
            device='cpu'
        )
        
        # Réinitialiser
        model.reset_state()
        
        # Test avec une image
        image = torch.randn(32, 32, dtype=torch.float32)
        results = model(image, return_intermediate=False)
        
        print(f"✓ Modèle créé et exécuté")
        print(f"  Nombre de cellules ganglionnaires: {results['n_ganglion_cells']}")
        
        if 'ganglion_spikes' in results:
            on_spikes = results['ganglion_spikes']['on']
            off_spikes = results['ganglion_spikes']['off']
            print(f"  Spikes ON: {on_spikes.shape if hasattr(on_spikes, 'shape') else 'N/A'}")
            print(f"  Spikes OFF: {off_spikes.shape if hasattr(off_spikes, 'shape') else 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Exécute tous les tests."""
    print("\n" + "="*80)
    print("DÉMARRAGE DES TESTS")
    print("="*80)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Photorecepteurs", test_photoreceptors()))
    results.append(("Modèle rétine", test_retina_model()))
    
    print("\n" + "="*80)
    print("RÉSUMÉ")
    print("="*80)
    
    all_passed = True
    for test_name, success in results:
        status = "✓ PASSÉ" if success else "✗ ÉCHOUÉ"
        print(f"  {test_name:<20} {status}")
        if not success:
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 MODULE RETINA FONCTIONNEL !")
        print("\nExemple d'utilisation:")
        print("""
from neurogeomvision.retina import SimpleRetinaModel
import torch

# Créer modèle
model = SimpleRetinaModel(input_shape=(64, 64), n_ganglion_cells=100)

# Traiter une image
image = torch.randn(64, 64)
model.reset_state()
results = model(image)

print(f"Nombre de cellules: {results['n_ganglion_cells']}")
""")
    else:
        print("⚠ Certains tests ont échoué.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
