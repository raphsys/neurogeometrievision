"""
Test de tous les modules optimisés.
"""

import torch
import time
import numpy as np
import sys
import os

# Ajoute le chemin parent pour les imports
sys.path.insert(0, os.path.abspath('..'))

def test_retina():
    """Test des filtres rétine optimisés."""
    print("\n" + "="*60)
    print("TEST FILTRES RÉTINE OPTIMISÉS")
    print("="*60)
    
    from neurogeomvision.retina_lgn.filters import apply_dog_filters
    
    # Crée une image de test
    image = torch.randn(64, 64)
    
    start = time.time()
    filtered = apply_dog_filters(image)
    elapsed = time.time() - start
    
    print(f"✓ DoG appliqué en {elapsed:.4f}s")
    print(f"  Shape: {filtered.shape}")
    print(f"  Range: [{filtered.min():.3f}, {filtered.max():.3f}]")
    
    return filtered

def test_v1():
    """Test des filtres V1 optimisés."""
    print("\n" + "="*60)
    print("TEST FILTRES V1 OPTIMISÉS")
    print("="*60)
    
    from neurogeomvision.v1_simple_cells.gabor_filters import GaborFilterBank
    
    # Crée un banc de filtres
    gabor = GaborFilterBank(img_size=(64, 64), n_orientations=8)
    
    # Image de test
    image = torch.randn(64, 64)
    
    start = time.time()
    results = gabor.apply_filters(image)
    elapsed = time.time() - start
    
    print(f"✓ {len(gabor.filters)} filtres appliqués en {elapsed:.4f}s")
    print(f"  Réponses: {results['responses'].shape}")
    print(f"  Orientation max: {results['dominant_orientation']['amplitude'].max():.3f}")
    
    return results

def test_association_field():
    """Test du champ d'association optimisé."""
    print("\n" + "="*60)
    print("TEST CHAMP D'ASSOCIATION OPTIMISÉ")
    print("="*60)
    
    from neurogeomvision.association_field.field_models import AssociationField
    
    # Crée le champ
    af = AssociationField(spatial_shape=(50, 50), orientation_bins=12)
    
    # Test création de champ
    start = time.time()
    stats = af.visualize_field(reference_orientation=0.0, field_size=15)
    elapsed = time.time() - start
    
    print(f"✓ Champ créé en {elapsed:.4f}s")
    print(f"  Excitateurs: {stats['excitatory_count']}")
    print(f"  Inhibiteurs: {stats['inhibitory_count']}")
    
    # Test propagation
    activity = torch.zeros(50, 50)
    activity[25, 25] = 1.0
    orientation = torch.zeros(50, 50)
    
    start = time.time()
    propagated = af.propagate_activity(activity, orientation, n_iterations=2)
    elapsed = time.time() - start
    
    print(f"✓ Activité propagée en {elapsed:.4f}s")
    print(f"  Activité max: {propagated.max():.3f}")
    
    return af, propagated

def test_pipeline():
    """Test d'un pipeline complet optimisé."""
    print("\n" + "="*60)
    print("TEST PIPELINE COMPLET OPTIMISÉ")
    print("="*60)
    
    from neurogeomvision.retina_lgn.filters import apply_dog_filters
    from neurogeomvision.v1_simple_cells.gabor_filters import GaborFilterBank
    from neurogeomvision.association_field.field_models import AssociationField
    
    total_start = time.time()
    
    # 1. Crée une image
    image = torch.randn(64, 64)
    print("1. Image créée")
    
    # 2. Filtrage rétine
    start = time.time()
    retina_output = apply_dog_filters(image)
    print(f"2. Rétine - {time.time() - start:.3f}s")
    
    # 3. Filtres V1
    gabor = GaborFilterBank(img_size=(64, 64), n_orientations=8)
    start = time.time()
    v1_output = gabor.apply_filters(retina_output)
    print(f"3. V1 - {time.time() - start:.3f}s")
    
    # 4. Champ d'association
    af = AssociationField(spatial_shape=(64, 64), orientation_bins=12)
    activity = v1_output['dominant_orientation']['amplitude']
    orientation = v1_output['dominant_orientation']['angle']
    
    start = time.time()
    association_output = af.propagate_activity(activity, orientation, n_iterations=2)
    print(f"4. Association - {time.time() - start:.3f}s")
    
    total_time = time.time() - total_start
    
    print(f"\n✓ Pipeline complet: {total_time:.3f}s")
    print(f"  Image: {image.shape}")
    print(f"  Rétine: {retina_output.shape}")
    print(f"  V1: {v1_output['responses'].shape}")
    print(f"  Association: {association_output.shape}")
    
    return {
        'image': image,
        'retina': retina_output,
        'v1': v1_output,
        'association': association_output,
        'total_time': total_time
    }

def main():
    """Test tous les modules optimisés."""
    print("\n" + "="*80)
    print("NEUROGEOMVISION - TESTS DES MODULES OPTIMISÉS")
    print("="*80)
    
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    results = {}
    
    try:
        print("\n▶ Début des tests...")
        
        # Test 1: Rétine
        results['retina'] = test_retina()
        
        # Test 2: V1
        results['v1'] = test_v1()
        
        # Test 3: Association field
        results['association'] = test_association_field()
        
        # Test 4: Pipeline complet
        results['pipeline'] = test_pipeline()
        
    except Exception as e:
        print(f"\n✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print("\n" + "="*80)
    print("✅ TOUS LES TESTS RÉUSSIS!")
    print("="*80)
    
    print(f"\n📊 RÉSULTATS:")
    print(f"  • Rétine: ✓")
    print(f"  • V1: ✓ ({len(results['v1']['filter_bank'])} filtres)")
    print(f"  • Association field: ✓")
    print(f"  • Pipeline complet: {results['pipeline']['total_time']:.3f}s")
    
    return results

if __name__ == "__main__":
    print("Démarrage des tests optimisés...")
    results = main()
    
    if results:
        print("\n🎉 Tests terminés avec succès!")
        print("Les modules optimisés fonctionnent correctement.")
    else:
        print("\n❌ Certains tests ont échoué.")
