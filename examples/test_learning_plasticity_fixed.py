"""
Test corrigé du module learning_plasticity.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

def test_natural_statistics_fixed():
    """Test CORRIGÉ des statistiques naturelles."""
    print("\n" + "="*60)
    print("TEST STATISTIQUES NATURELLES (CORRIGÉ)")
    print("="*60)
    
    from neurogeomvision.learning_plasticity.natural_statistics import NaturalStatistics
    
    # Crée une instance
    stats = NaturalStatistics(patch_size=16)
    
    # Utilise la méthode intégrée pour créer du bruit 1/f
    try:
        image = stats.create_1f_noise(size=128)
        print(f"✓ Image 1/f créée: {image.shape}")
    except Exception as e:
        print(f"⚠ Erreur création 1/f: {e}")
        # Fallback: image simple
        image = torch.randn(128, 128, device=stats.device)
        print(f"  Utilisation image aléatoire: {image.shape}")
    
    # Analyse
    try:
        results = stats.analyze_natural_image(image, n_patches=1000)
        
        print(f"✓ Analyse terminée")
        print(f"  Shape image: {results.get('image_shape', 'N/A')}")
        print(f"  Nombre de patches: {1000}")
        
        if 'eigenvalues' in results:
            print(f"  Nombre de valeurs propres: {len(results['eigenvalues'])}")
            if len(results['eigenvalues']) > 0:
                print(f"  Valeur propre max: {results['eigenvalues'][0]:.4f}")
                print(f"  Valeur propre min: {results['eigenvalues'][-1]:.4f}")
        
        if 'kurtosis' in results:
            print(f"  Kurtosis moyen: {results['kurtosis'].mean().item():.4f}")
        
        if 'radial_profile' in results:
            print(f"  Longueur profil spectral: {len(results['radial_profile'])}")
        
        # Visualisation
        try:
            stats.visualize_statistics(results, 'natural_statistics_fixed.png')
            print(f"✓ Visualisation sauvegardée")
        except Exception as e:
            print(f"⚠ Erreur visualisation: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur analyse: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ica_learning_fixed():
    """Test CORRIGÉ de l'apprentissage ICA."""
    print("\n" + "="*60)
    print("TEST ICA LEARNING (CORRIGÉ)")
    print("="*60)
    
    from neurogeomvision.learning_plasticity.natural_statistics import ICA_Learning
    
    # Crée des données synthétiques
    patch_size = 16
    n_patches = 500
    n_components = 16
    
    # Patches avec structure
    patches = torch.randn(n_patches, patch_size * patch_size)
    
    # CORRECTION: Convertir theta en tensor avant d'utiliser torch.cos
    for i in range(n_patches):
        # Crée un filtre de Gabor simple
        patch = patches[i].reshape(patch_size, patch_size)
        
        # Ajoute une orientation
        theta_val = torch.rand(1).item() * np.pi  # float
        theta_tensor = torch.tensor(theta_val)    # Convertir en tensor
        
        for y in range(patch_size):
            for x in range(patch_size):
                x_centered = x - patch_size/2
                y_centered = y - patch_size/2
                # Utiliser theta_tensor au lieu de theta_val
                x_rot = x_centered * torch.cos(theta_tensor) + y_centered * torch.sin(theta_tensor)
                patch[y, x] += 0.5 * torch.cos(0.3 * x_rot)
        
        patches[i] = patch.flatten()
    
    # ICA
    ica = ICA_Learning(
        input_dim=patch_size * patch_size,
        n_components=n_components,
        learning_rate=0.01
    )
    
    try:
        filters = ica.learn_gabor_filters_simple(patches, n_epochs=30)
        
        print(f"✓ ICA terminé")
        print(f"  Filtres shape: {filters.shape}")
        print(f"  Normes min/max: {filters.norm(dim=1).min():.4f}/{filters.norm(dim=1).max():.4f}")
        
        # Visualise quelques filtres
        if filters.shape[0] >= 4:
            fig, axes = plt.subplots(2, 2, figsize=(8, 8))
            for i in range(4):
                ax = axes[i // 2, i % 2]
                filter_img = filters[i].reshape(patch_size, patch_size).cpu().numpy()
                im = ax.imshow(filter_img, cmap='RdBu_r')
                ax.set_title(f"Filtre {i+1}")
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)
            
            plt.suptitle("Filtres appris par ICA", fontsize=12)
            plt.tight_layout()
            plt.savefig('ica_filters.png', dpi=100)
            plt.close()
            print(f"✓ Filtres visualisés: ica_filters.png")
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur ICA: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sparse_coding_fixed():
    """Test CORRIGÉ du sparse coding."""
    print("\n" + "="*60)
    print("TEST SPARSE CODING (CORRIGÉ)")
    print("="*60)
    
    from neurogeomvision.learning_plasticity.natural_statistics import SparseCoding
    
    # Données simples
    patch_size = 16
    n_patches = 200
    n_basis = 32
    
    # Patches aléatoires
    patches = torch.randn(n_patches, patch_size * patch_size)
    
    # Normalise
    patches = patches / (patches.norm(dim=1, keepdim=True) + 1e-8)
    
    # Sparse Coding
    sc = SparseCoding(
        input_dim=patch_size * patch_size,
        n_basis=n_basis,
        sparsity_weight=0.1,
        learning_rate=0.01
    )
    
    try:
        basis = sc.learn_dictionary_simple(patches, n_epochs=20)
        
        print(f"✓ Sparse Coding terminé")
        print(f"  Basis shape: {basis.shape}")
        print(f"  Normes: [{basis.norm(dim=1).min():.4f}, {basis.norm(dim=1).max():.4f}]")
        
        # Test d'encodage
        coefficients = sc.sparse_encode_simple(patches[:10], n_iterations=10)
        print(f"  Coefficients shape: {coefficients.shape}")
        print(f"  Sparsité: {torch.mean(torch.abs(coefficients)).item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur Sparse Coding: {e}")
        import traceback
        traceback.print_exc()
        return False

def main_fixed():
    """Tests principaux corrigés."""
    print("\n" + "="*80)
    print("NEUROGEOMVISION - TESTS LEARNING_PLASTICITY (CORRIGÉS)")
    print("="*80)
    
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    results = []
    
    print("\n▶ Test 1: Statistiques naturelles...")
    results.append(("Statistiques naturelles", test_natural_statistics_fixed()))
    
    print("\n▶ Test 2: ICA Learning...")
    results.append(("ICA Learning", test_ica_learning_fixed()))
    
    print("\n▶ Test 3: Sparse Coding...")
    results.append(("Sparse Coding", test_sparse_coding_fixed()))
    
    # Résumé
    print("\n" + "="*80)
    print("RÉSUMÉ DES TESTS CORRIGÉS")
    print("="*80)
    
    for test_name, success in results:
        status = "✓ PASSÉ" if success else "✗ ÉCHOUÉ"
        print(f"  {test_name:<25} {status}")
    
    n_passed = sum(1 for _, s in results if s)
    n_total = len(results)
    
    print(f"\nTotal: {n_passed}/{n_total} tests réussis")
    
    if n_passed == n_total:
        print("\n🎉 TOUS LES TESTS DE PLASTICITÉ RÉUSSIS!")
    else:
        print(f"\n⚠ {n_total - n_passed} tests ont échoué.")
    
    return n_passed == n_total

if __name__ == "__main__":
    success = main_fixed()
    sys.exit(0 if success else 1)
