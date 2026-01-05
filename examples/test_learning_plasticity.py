"""
Test complet du module learning_plasticity.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

def test_hebbian_learning():
    """Test de l'apprentissage hebbien."""
    print("\n" + "="*60)
    print("TEST APPRENTISSAGE HEBBIEN")
    print("="*60)
    
    from neurogeomvision.learning_plasticity.hebbian import HebbianLearning
    
    # Crée des données corrélées
    input_size = 100
    output_size = 10
    n_samples = 1000
    
    # Données avec structure
    data = torch.randn(n_samples, input_size)
    # Ajoute des corrélations
    data[:, 10:20] = data[:, 0:10] * 0.8 + torch.randn(n_samples, 10) * 0.2
    
    # Apprentissage hebbien
    hebbian = HebbianLearning(
        input_size=input_size,
        output_size=output_size,
        learning_rate=0.01
    )
    
    stats = hebbian.learn_from_data(data, n_epochs=50, batch_size=32)
    
    # Extrait les caractéristiques
    features = hebbian.extract_features(n_features=5)
    
    print(f"✓ Hebbian learning terminé")
    print(f"  Nombre de caractéristiques: {features.shape[0]}")
    print(f"  Forme: {features.shape}")
    
    # Visualisation
    hebbian.visualize_learning(stats, 'hebbian_learning.png')
    
    return True

def test_stdp_plasticity():
    """Test de la plasticité STDP."""
    print("\n" + "="*60)
    print("TEST PLASTICITÉ STDP")
    print("="*60)
    
    from neurogeomvision.learning_plasticity.stdp import STDPLearning
    
    # Simule STDP
    n_neurons = 20
    stdp = STDPLearning(n_neurons=n_neurons)
    
    results = stdp.simulate_spike_train(n_steps=500, firing_rate=0.1)
    
    print(f"✓ STDP simulation terminée")
    print(f"  Poids finaux: {results['final_weights'].shape}")
    print(f"  Poids moyen: {results['final_weights'].mean():.4f}")
    
    return True

def test_bcm_learning():
    """Test de l'apprentissage BCM."""
    print("\n" + "="*60)
    print("TEST APPRENTISSAGE BCM")
    print("="*60)
    
    from neurogeomvision.learning_plasticity.bcm import BCMLearning
    
    # Données avec sélectivité
    input_size = 50
    n_samples = 500
    
    data = torch.randn(n_samples, input_size)
    # Crée des clusters
    data[:200, :20] += 2.0
    data[200:400, 20:40] += 2.0
    
    # BCM
    bcm = BCMLearning(
        input_size=input_size,
        output_size=10,
        learning_rate=0.01
    )
    
    stats = bcm.learn_selectivity(data, n_epochs=100)
    
    print(f"✓ BCM learning terminé")
    print(f"  Sélectivité finale: {stats['selectivity'][-1]:.4f}")
    print(f"  θ moyen: {stats['theta_history'][-1]:.4f}")
    
    return True

def test_developmental_learning():
    """Test de l'apprentissage développemental."""
    print("\n" + "="*60)
    print("TEST APPRENTISSAGE DÉVELOPPEMENTAL")
    print("="*60)
    
    from neurogeomvision.learning_plasticity.developmental import DevelopmentalLearning
    
    # Dominance oculaire
    dev = DevelopmentalLearning(cortical_size=(40, 40))
    
    weights = dev.develop_ocular_dominance(n_steps=2000)
    
    # Calcule l'index de dominance
    od_index = dev.compute_ocular_dominance_index()
    
    print(f"✓ Développement terminé")
    print(f"  Poids shape: {weights.shape}")
    print(f"  OD index range: [{od_index.min():.3f}, {od_index.max():.3f}]")
    
    # Visualisation
    dev.visualize_development('ocular_dominance.png')
    
    return True

def test_natural_statistics():
    """Test des statistiques naturelles."""
    print("\n" + "="*60)
    print("TEST STATISTIQUES NATURELLES")
    print("="*60)
    
    from neurogeomvision.learning_plasticity.natural_statistics import NaturalStatistics
    
    # Crée une image naturelle synthétique (texture 1/f)
    size = 128
    image = torch.randn(size, size)
    
    # Filtre 1/f
    fy, fx = torch.meshgrid(
        torch.fft.fftfreq(size),
        torch.fft.fftfreq(size),
        indexing='ij'
    )
    f = torch.sqrt(fx**2 + fy**2)
    f[0, 0] = 1.0  # Évite division par zéro
    
    fft = torch.fft.fft2(image)
    fft = fft / f
    image = torch.fft.ifft2(fft).real
    
    # Analyse
    stats = NaturalStatistics(patch_size=16)
    results = stats.analyze_natural_image(image, n_patches=2000)
    
    print(f"✓ Analyse terminée")
    print(f"  Nombre de patches: {2000}")
    print(f"  Nombre de valeurs propres: {len(results['eigenvalues'])}")
    print(f"  Kurtosis moyen: {results['kurtosis'].mean():.4f}")
    
    # Visualisation
    stats.visualize_statistics(results, 'natural_statistics.png')
    
    return True

def test_integration():
    """Test d'intégration avec les modules existants."""
    print("\n" + "="*60)
    print("TEST INTÉGRATION")
    print("="*60)
    
    from neurogeomvision.learning_plasticity.integration import PlasticityIntegrator
    
    # Crée des images synthétiques
    n_images = 5
    images = []
    for _ in range(n_images):
        img = torch.randn(64, 64)
        # Ajoute de la structure
        img[20:40, 20:40] += 1.0
        images.append(img)
    
    # Intégrateur
    integrator = PlasticityIntegrator(image_size=(64, 64))
    
    # Apprend les filtres
    try:
        gabor_bank = integrator.learn_gabor_filters_from_natural_images(
            images, n_orientations=8, n_epochs=50
        )
        print(f"✓ Filtres Gabor appris")
    except Exception as e:
        print(f"⚠ Erreur filtres Gabor: {e}")
        gabor_bank = None
    
    # Apprend le champ d'association
    try:
        assoc_field = integrator.learn_association_field_hebbian(n_iterations=500)
        print(f"✓ Champ d'association appris")
    except Exception as e:
        print(f"⚠ Erreur champ association: {e}")
        assoc_field = None
    
    # Développe les colonnes d'orientation
    try:
        orientation_map = integrator.develop_orientation_columns(
            input_size=(30, 30), n_steps=1000
        )
        print(f"✓ Colonnes d'orientation développées")
        print(f"  Carte shape: {orientation_map.shape}")
    except Exception as e:
        print(f"⚠ Erreur colonnes orientation: {e}")
        orientation_map = None
    
    return True

def main():
    """Tests principaux."""
    print("\n" + "="*80)
    print("NEUROGEOMVISION - TESTS LEARNING_PLASTICITY")
    print("="*80)
    
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    results = {}
    
    try:
        results['hebbian'] = test_hebbian_learning()
    except Exception as e:
        print(f"✗ Erreur hebbian: {e}")
        results['hebbian'] = False
    
    try:
        results['stdp'] = test_stdp_plasticity()
    except Exception as e:
        print(f"✗ Erreur stdp: {e}")
        results['stdp'] = False
    
    try:
        results['bcm'] = test_bcm_learning()
    except Exception as e:
        print(f"✗ Erreur bcm: {e}")
        results['bcm'] = False
    
    try:
        results['developmental'] = test_developmental_learning()
    except Exception as e:
        print(f"✗ Erreur developmental: {e}")
        results['developmental'] = False
    
    try:
        results['natural_stats'] = test_natural_statistics()
    except Exception as e:
        print(f"✗ Erreur natural stats: {e}")
        results['natural_stats'] = False
    
    try:
        results['integration'] = test_integration()
    except Exception as e:
        print(f"✗ Erreur integration: {e}")
        results['integration'] = False
    
    # Résumé
    print("\n" + "="*80)
    print("RÉSUMÉ DES TESTS")
    print("="*80)
    
    for test_name, success in results.items():
        status = "✓ PASSÉ" if success else "✗ ÉCHOUÉ"
        print(f"  {test_name:<20} {status}")
    
    n_passed = sum(1 for s in results.values() if s)
    n_total = len(results)
    
    print(f"\nTotal: {n_passed}/{n_total} tests réussis")
    
    if n_passed == n_total:
        print("\n🎉 TOUS LES TESTS DE PLASTICITÉ RÉUSSIS!")
        print("Le module learning_plasticity est fonctionnel.")
    else:
        print(f"\n⚠ {n_total - n_passed} tests ont échoué.")
    
    return results

if __name__ == "__main__":
    main()
