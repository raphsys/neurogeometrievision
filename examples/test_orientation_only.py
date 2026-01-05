"""
Test spécifique du module orientation.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

print("="*80)
print("TEST SPÉCIFIQUE DU MODULE ORIENTATION")
print("="*80)

def test_orientation_selectivity():
    """Test de la sélectivité à l'orientation."""
    try:
        from neurogeomvision.cortex.orientation import (
            OrientationSelectivity,
            create_orientation_filters,
            extract_orientation_features
        )
        
        print("✓ Import réussi")
        
        device = 'cpu'
        
        # Test 1: Création de filtres d'orientation
        print("\n1. Test création de filtres d'orientation...")
        n_orientations = 8
        filter_size = 15
        filters = create_orientation_filters(n_orientations, filter_size, device)
        print(f"  Filtres créés: {filters.shape}")  # Devrait être (8, 1, 15, 15)
        
        # Test 2: Modèle de sélectivité
        print("\n2. Test OrientationSelectivity...")
        model = OrientationSelectivity(n_orientations=n_orientations, device=device)
        
        # Test avec différentes tailles d'image
        test_sizes = [(32, 32), (64, 64), (128, 128)]
        
        for h, w in test_sizes:
            print(f"\n  Image {h}x{w}:")
            test_image = torch.randn(h, w, device=device)
            
            # Cellules simples
            simple_results = model(test_image, cell_type='simple')
            print(f"    Simple - Réponses: {simple_results['responses'].shape}")
            print(f"    Simple - Carte orientation: {simple_results['orientation_map'].shape}")
            
            # Cellules complexes
            complex_results = model(test_image, cell_type='complex')
            print(f"    Complexe - Énergie: {complex_results['energy'].shape}")
            print(f"    Complexe - Carte orientation: {complex_results['orientation_map'].shape}")
        
        # Test 3: Extraction de caractéristiques
        print("\n3. Test extraction de caractéristiques...")
        test_image = torch.randn(3, 64, 64, device=device)  # Image RGB
        features = extract_orientation_features(test_image, filters, pooling='max')
        
        print(f"  Réponses: {features['responses'].shape}")
        print(f"  Carte d'orientation: {features['orientation_map'].shape}")
        print(f"  Force: {features['strength_map'].shape}")
        
        print("\n" + "="*80)
        print("✅ TESTS ORIENTATION PASSÉS AVEC SUCCÈS !")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n>>> Début du test orientation")
    success = test_orientation_selectivity()
    
    if success:
        print("\n🎉 Module orientation fonctionnel !")
    else:
        print("\n⚠️  Problèmes détectés dans le module orientation")
