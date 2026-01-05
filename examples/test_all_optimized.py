"""
Test de tous les modules optimisés.
"""

import torch
import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath('.'))

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
    print(f"  Orientation dominante: {results['dominant_orientation']['angle'].shape}")
    
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
    print(f"  Field: {stats['field'].shape}")
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

def test_geometric_hallucinations():
    """Test des hallucinations optimisées."""
    print("\n" + "="*60)
    print("TEST HALLUCINATIONS GÉOMÉTRIQUES OPTIMISÉES")
    print("="*60)
    
    from neurogeomvision.entoptic_patterns.geometric_hallucinations import GeometricHallucinations
    
    # Crée le modèle (petit pour rapidité)
    gh = GeometricHallucinations(spatial_shape=(32, 32), orientation_bins=8)
    
    start = time.time()
    activity = gh.generate_hallucination(n_steps=10)
    elapsed = time.time() - start
    
    print(f"✓ Hallucination générée en {elapsed:.4f}s")
    print(f"  Activity shape: {activity.shape}")
    
    # Visualise
    visual = gh.project_to_visual_field(activity, 'mean')
    print(f"  Visual field: {visual.shape}")
    print(f"  Range: [{visual.min():.3f}, {visual.max():.3f}]")
    
    return gh, activity

def main():
    """Test tous les modules optimisés."""
    print("\n" + "="*80)
    print("TEST COMPLET DES MODULES OPTIMISÉS")
    print("="*80)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Test 1: Rétine
    retina_result = test_retina()
    
    # Test 2: V1
    v1_result = test_v1()
    
    # Test 3: Association field
    af_result = test_association_field()
    
    # Test 4: Hallucinations
    gh_result = test_geometric_hallucinations()
    
    print("\n" + "="*80)
    print("TOUS LES TESTS OPTIMISÉS TERMINÉS AVEC SUCCÈS!")
    print("="*80)
    
    return {
        'retina': retina_result,
        'v1': v1_result,
        'association_field': af_result,
        'geometric_hallucinations': gh_result
    }

if __name__ == "__main__":
    results = main()
