"""
Test complet du module retina.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

print("="*80)
print("TEST COMPLET DU MODULE RETINA")
print("="*80)

def test_all_components():
    """Test tous les composants du module retina."""
    try:
        from neurogeomvision.retina import (
            Cone, Rod, PhotoreceptorLayer,
            HorizontalCell, BipolarCell, AmacrineCell,
            GanglionCell, ONGanglionCell, OFFGanglionCell,
            RetinotopicMap, SimpleRetinaModel, BioInspiredRetina
        )
        
        print("✓ Tous les composants importés")
        
        # Test 1: Photorecepteurs
        print("\n1. Test photorécepteurs...")
        layer = PhotoreceptorLayer(mosaic_shape=(10, 10), device='cpu')
        image = torch.randn(10, 10)
        response = layer(image)
        print(f"  PhotoreceptorLayer: input={image.shape}, output={response.shape}")
        
        # Test 2: Circuit rétinien
        print("\n2. Test circuit rétinien...")
        circuit = HorizontalCell(device='cpu')
        response = circuit(torch.randn(10, 10))
        print(f"  HorizontalCell: output shape={response.shape}")
        
        # Test 3: Cellules ganglionnaires
        print("\n3. Test cellules ganglionnaires...")
        ganglion = GanglionCell(cell_type='midget', device='cpu')
        spikes, potential = ganglion(torch.randn(8, 8))
        print(f"  GanglionCell: spikes={spikes.shape}, potential={potential.shape}")
        
        # Test 4: Modèle complet
        print("\n4. Test modèle complet...")
        model = SimpleRetinaModel(
            input_shape=(32, 32),
            n_ganglion_cells=30,
            use_color=False,
            device='cpu'
        )
        model.reset_state()
        image = torch.randn(32, 32)
        results = model(image, return_intermediate=False)
        print(f"  SimpleRetinaModel: n_cells={results['n_ganglion_cells']}")
        
        # Test 5: Rétine bio-inspirée
        print("\n5. Test rétine bio-inspirée...")
        bio_retina = BioInspiredRetina(
            retinal_shape=(64, 64),
            cortical_shape=(100, 100),
            n_ganglion_cells=50,
            include_retinotopic_mapping=True,
            device='cpu'
        )
        bio_retina.reset_state()
        image = torch.randn(64, 64)
        results = bio_retina(image, return_cortical=False)
        print(f"  BioInspiredRetina: fonctionnel")
        
        print("\n" + "="*80)
        print("✅ MODULE RETINA COMPLÈTEMENT FONCTIONNEL !")
        print("="*80)
        
        # Exemple d'utilisation
        print("\nExemple d'utilisation :")
        print("""
from neurogeomvision.retina import SimpleRetinaModel
import torch

# Créer un modèle de rétine
model = SimpleRetinaModel(
    input_shape=(64, 64),
    n_ganglion_cells=100,
    use_color=True
)

# Réinitialiser les états
model.reset_state()

# Traiter une image
image = torch.randn(3, 64, 64)  # Image RGB
results = model(image)

print(f"Nombre de cellules ganglionnaires: {results['n_ganglion_cells']}")
print(f"Spikes ON: {results['ganglion_spikes']['on'].shape}")
print(f"Spikes OFF: {results['ganglion_spikes']['off'].shape}")
""")
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_all_components()
    sys.exit(0 if success else 1)
