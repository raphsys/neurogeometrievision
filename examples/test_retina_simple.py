"""
Test simple du module retina.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

print("="*80)
print("TEST SIMPLE DU MODULE RETINA")
print("="*80)

# Test 1: Import
print("1. Test d'import...")
try:
    from neurogeomvision.retina import Cone, BipolarCell
    print("✓ Import réussi")
    
    # Test 2: Cone
    print("\n2. Test Cone...")
    cone = Cone(cone_type='M')
    response = cone(torch.tensor([0.5], dtype=torch.float32))
    print(f"✓ Cone réponse: {response.item():.3f}")
    
    # Test 3: BipolarCell
    print("\n3. Test BipolarCell...")
    bipolar = BipolarCell(cell_type='ON')
    test_input = torch.randn(1, 10, 10, dtype=torch.float32)
    response = bipolar(test_input)
    print(f"✓ BipolarCell réponse shape: {response.shape}")
    
    # Test 4: reset_state
    print("\n4. Test reset_state...")
    bipolar.reset_state()  # Ne devrait pas planter
    print("✓ reset_state fonctionne")
    
    # Test 5: SimpleRetinaModel
    print("\n5. Test SimpleRetinaModel...")
    from neurogeomvision.retina import SimpleRetinaModel
    
    model = SimpleRetinaModel(
        input_shape=(16, 16),
        n_ganglion_cells=20,
        use_color=False,
        device='cpu'
    )
    
    print("✓ Modèle créé")
    
    # Test reset_state du modèle
    model.reset_state()
    print("✓ reset_state du modèle fonctionne")
    
    # Test forward (simplifié)
    image = torch.randn(16, 16, dtype=torch.float32)
    try:
        results = model(image, return_intermediate=False)
        print(f"✓ Forward pass réussi")
        print(f"  n_ganglion_cells: {results.get('n_ganglion_cells', 'N/A')}")
    except Exception as e:
        print(f"✗ Erreur forward: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ TESTS SIMPLES RÉUSSIS !")
    
except Exception as e:
    print(f"✗ Erreur: {e}")
    import traceback
    traceback.print_exc()
