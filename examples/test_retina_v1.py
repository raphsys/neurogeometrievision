import torch
import matplotlib.pyplot as plt
import sys
import os

# Ajoute le chemin du projet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neurogeomvision.retina_lgn.filters import ParvoMagnoPathway
from neurogeomvision.retina_lgn.coding import SpikeEncoder, TemporalProcessor
from neurogeomvision.v1_simple_cells.gabor_filters import GaborFilterBank


def test_retina_lgn():
    """Teste les filtres rétine/LGN sur une image simple."""
    print("Test des filtres Rétine/LGN...")
    
    # Crée une image de test (bord vertical)
    h, w = 128, 128
    device = 'cpu'  # Définir le device
    image = torch.zeros(h, w, device=device)
    image[:, w//2-10:w//2+10] = 1.0
    
    # Ajoute un peu de bruit
    image += torch.randn(h, w, device=device) * 0.1
    
    # Traite avec les voies parvo/magno
    pathway = ParvoMagnoPathway(img_size=(h, w), device=device)
    results = pathway.process_frame(image)
    
    # Visualise les résultats
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Image originale
    axes[0, 0].imshow(image.cpu().numpy(), cmap='gray')
    axes[0, 0].set_title("Image originale")
    axes[0, 0].axis('off')
    
    # Filtre parvo
    axes[0, 1].imshow(results['parvo_kernel'].cpu().numpy(), cmap='RdBu_r')
    axes[0, 1].set_title("Filtre Parvo (DoG)")
    axes[0, 1].axis('off')
    
    # Réponse parvo
    im = axes[0, 2].imshow(results['parvo'].cpu().numpy(), cmap='RdBu_r')
    axes[0, 2].set_title("Réponse Parvo")
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2])
    
    # Filtre magno
    axes[1, 0].imshow(results['magno_kernel'].cpu().numpy(), cmap='RdBu_r')
    axes[1, 0].set_title("Filtre Magno (∂DoG/∂t)")
    axes[1, 0].axis('off')
    
    # Réponse magno
    im = axes[1, 1].imshow(results['magno'].cpu().numpy(), cmap='RdBu_r')
    axes[1, 1].set_title("Réponse Magno")
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1])
    
    # Différence
    diff = results['parvo'] - results['magno']
    im = axes[1, 2].imshow(diff.cpu().numpy(), cmap='RdBu_r')
    axes[1, 2].set_title("Différence Parvo-Magno")
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('retina_lgn_test.png', dpi=150)
    
    # N'affiche pas si non-interactif
    try:
        plt.show()
    except Exception:
        print("(Graphique sauvegardé dans retina_lgn_test.png)")
    
    # Test du codage en spikes
    print("\nTest du codage en spikes...")
    encoder = SpikeEncoder()
    
    # Utilise la réponse parvo pour générer des spikes
    parvo_norm = (results['parvo'] - results['parvo'].min()) / (results['parvo'].max() - results['parvo'].min() + 1e-6)
    
    # Codage par fréquence
    rate_spikes = encoder.rate_coding(parvo_norm, max_rate=50, time_steps=5)
    print(f"Shape des spikes (rate coding): {rate_spikes.shape}")
    print(f"Nombre total de spikes: {rate_spikes.sum().item()}")
    
    # Codage par rang
    rank_spikes = encoder.rank_coding(parvo_norm, time_steps=5)
    print(f"Nombre total de spikes (rank coding): {rank_spikes.sum().item()}")
    
    # Test de l'intégration temporelle
    processor = TemporalProcessor(tau=20.0, dt=1.0, device=device)  # CORRECTION ICI
    voltage = processor.leaky_integrate(rate_spikes)
    print(f"Voltage shape: {voltage.shape}")
    print(f"Voltage max: {voltage.max().item():.3f}, min: {voltage.min().item():.3f}")
    
    return results


def test_v1_simple_cells():
    """Teste les filtres de Gabor pour les neurones simples de V1."""
    print("\n\nTest des neurones simples de V1 (filtres de Gabor)...")
    
    device = 'cpu'  # Ajouter cette ligne
    
    # Crée une image avec plusieurs orientations
    h, w = 128, 128
    image = torch.zeros(h, w, device=device)
    
    # Ajoute des barres à différentes orientations
    orientations = [0, 45, 90, 135]  # En degrés
    for angle in orientations:
        rad = angle * 3.14159 / 180
        for i in range(-50, 51, 5):
            x = w//2 + int(i * torch.cos(torch.tensor(rad)).item())
            y = h//2 + int(i * torch.sin(torch.tensor(rad)).item())
            if 0 <= x < w and 0 <= y < h:
                # Dessine un petit segment
                for dx in [-2, -1, 0, 1, 2]:
                    for dy in [-2, -1, 0, 1, 2]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            image[ny, nx] = 1.0
    
    image += torch.randn(h, w, device=device) * 0.05
    
    # Crée la banque de filtres de Gabor
    gabor_bank = GaborFilterBank(
        img_size=(h, w),
        n_orientations=8,
        spatial_freqs=[0.1, 0.2],
        phases=[0, 1.57],  # 0 et pi/2
        device=device  # Ajouter le device ici
    )
    
    # Applique les filtres
    results = gabor_bank.apply_filters(image)
    
    # Visualise les résultats
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    # Image originale
    axes[0, 0].imshow(image.cpu().numpy(), cmap='gray')
    axes[0, 0].set_title("Image originale")
    axes[0, 0].axis('off')
    
    # Carte d'orientation
    im = axes[0, 1].imshow(results['orientation_map'].cpu().numpy(), cmap='hsv')
    axes[0, 1].set_title("Carte d'orientation (pinwheels simplifiée)")
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1])
    
    # Orientation dominante
    orient_plot = axes[0, 2].imshow(results['dominant_orientation']['angle'].cpu().numpy(), 
                                    cmap='hsv', vmin=0, vmax=3.14159)
    axes[0, 2].set_title("Orientation dominante détectée")
    axes[0, 2].axis('off')
    plt.colorbar(orient_plot, ax=axes[0, 2])
    
    # Cohérence
    coher_plot = axes[1, 0].imshow(results['dominant_orientation']['coherence'].cpu().numpy(), 
                                   cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title("Cohérence d'orientation")
    axes[1, 0].axis('off')
    plt.colorbar(coher_plot, ax=axes[1, 0])
    
    # Amplitude
    amp_plot = axes[1, 1].imshow(results['dominant_orientation']['amplitude'].cpu().numpy(), 
                                 cmap='gray')
    axes[1, 1].set_title("Amplitude de réponse")
    axes[1, 1].axis('off')
    plt.colorbar(amp_plot, ax=axes[1, 1])
    
    # Visualise la réponse d'un filtre spécifique
    if results['filter_responses']:
        first_key = list(results['filter_responses'].keys())[0]
        first_response = results['filter_responses'][first_key]['response']
        resp_plot = axes[1, 2].imshow(first_response.cpu().numpy(), cmap='RdBu_r')
        axes[1, 2].set_title(f"Réponse d'un filtre (θ={results['filter_responses'][first_key]['params']['theta']:.2f})")
        axes[1, 2].axis('off')
        plt.colorbar(resp_plot, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('v1_simple_cells_test.png', dpi=150)
    try:
        plt.show()
    except:
        print("(Graphique sauvegardé dans v1_simple_cells_test.png)")
    
    # Visualise quelques filtres
    fig = gabor_bank.visualize_filters(n_filters=6)
    plt.savefig('gabor_filters.png', dpi=150)
    try:
        plt.show()
    except:
        print("(Filtres sauvegardés dans gabor_filters.png)")
    
    # Affiche un résumé
    summary = gabor_bank.get_filter_summary()
    print(f"\nRésumé de la banque de filtres:")
    print(f"- Nombre total de filtres: {summary['total_filters']}")
    print(f"- Orientations: {summary['orientations']['min']:.2f} à {summary['orientations']['max']:.2f} rad")
    print(f"- Fréquences spatiales: {summary['frequencies']['min']:.2f} à {summary['frequencies']['max']:.2f} cycles/pixel")
    print(f"- Tailles des noyaux: {summary['sizes']['min']} à {summary['sizes']['max']} pixels")
    
    return results

def test_integration():
    """Test l'intégration complète du pipeline."""
    print("\n\nTest d'intégration du pipeline...")
    
    # Crée une image synthétique
    h, w = 128, 128
    image = torch.zeros(h, w)
    
    # Ajoute un cercle
    center_x, center_y = w//2, h//2
    radius = 30
    for y in range(h):
        for x in range(w):
            if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                image[y, x] = 1.0
    
    # Ajoute du bruit
    image += torch.randn(h, w) * 0.1
    
    # 1. Traitement rétine/LGN
    print("1. Traitement rétine/LGN...")
    pathway = ParvoMagnoPathway(img_size=(h, w))
    retina_results = pathway.process_frame(image)
    
    # 2. Neurones V1
    print("2. Neurones simples de V1...")
    gabor_bank = GaborFilterBank(
        img_size=(h, w),
        n_orientations=8,
        spatial_freqs=[0.1, 0.2],
        phases=[0, 1.57]
    )
    v1_results = gabor_bank.apply_filters(image)
    
    # 3. Codage en spikes
    print("3. Codage en spikes...")
    encoder = SpikeEncoder()
    
    # Utilise la réponse parvo normalisée
    parvo_response = retina_results['parvo']
    parvo_norm = (parvo_response - parvo_response.min()) / (parvo_response.max() - parvo_response.min() + 1e-6)
    
    # Codage par rang (plus plausible biologiquement)
    spikes = encoder.rank_coding(parvo_norm, time_steps=10)
    
    # 4. Visualisation intégrée
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Image originale
    axes[0, 0].imshow(image.numpy(), cmap='gray')
    axes[0, 0].set_title("Image originale")
    axes[0, 0].axis('off')
    
    # Réponse parvo
    axes[0, 1].imshow(retina_results['parvo'].numpy(), cmap='RdBu_r')
    axes[0, 1].set_title("Réponse Parvo (LGN)")
    axes[0, 1].axis('off')
    
    # Orientation dominante V1
    im1 = axes[0, 2].imshow(v1_results['dominant_orientation']['angle'].cpu().numpy(), 
                            cmap='hsv', vmin=0, vmax=3.14159)
    axes[0, 2].set_title("Orientation V1")
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2])
    
    # Cohérence V1
    im2 = axes[0, 3].imshow(v1_results['dominant_orientation']['coherence'].cpu().numpy(), 
                            cmap='hot', vmin=0, vmax=1)
    axes[0, 3].set_title("Cohérence V1")
    axes[0, 3].axis('off')
    plt.colorbar(im2, ax=axes[0, 3])
    
    # Spikes (premier pas de temps)
    axes[1, 0].imshow(spikes[0].numpy(), cmap='hot')
    axes[1, 0].set_title("Spikes (t=0)")
    axes[1, 0].axis('off')
    
    # Spikes (somme temporelle)
    axes[1, 1].imshow(spikes.sum(dim=0).numpy(), cmap='hot')
    axes[1, 1].set_title("Spikes (somme temporelle)")
    axes[1, 1].axis('off')
    
    # Réponse magno
    axes[1, 2].imshow(retina_results['magno'].numpy(), cmap='RdBu_r')
    axes[1, 2].set_title("Réponse Magno")
    axes[1, 2].axis('off')
    
    # Carte d'orientation V1
    im3 = axes[1, 3].imshow(v1_results['orientation_map'].cpu().numpy(), cmap='hsv')
    axes[1, 3].set_title("Carte orientation V1")
    axes[1, 3].axis('off')
    plt.colorbar(im3, ax=axes[1, 3])
    
    plt.tight_layout()
    plt.savefig('integration_test.png', dpi=150)
    plt.show()
    
    print(f"\nRésumé :")
    print(f"- Nombre de filtres Gabor : {len(gabor_bank.filters)}")
    print(f"- Spikes générés : {spikes.sum().item():.0f}")
    print(f"- Cohérence moyenne V1 : {v1_results['dominant_orientation']['coherence'].mean().item():.3f}")
    
    return {
        'image': image,
        'retina_results': retina_results,
        'v1_results': v1_results,
        'spikes': spikes
    }


def main():
    """Fonction principale pour exécuter tous les tests."""
    print("=" * 60)
    print("TEST DE LA BIBLIOTHÈQUE NEUROGEOMVISION")
    print("=" * 60)
    
    # Test 1: Rétine/LGN
    print("\n" + "=" * 60)
    print("TEST 1: Système Rétine/LGN")
    print("=" * 60)
    retina_results = test_retina_lgn()
    
    # Test 2: Neurones V1
    print("\n" + "=" * 60)
    print("TEST 2: Neurones simples de V1")
    print("=" * 60)
    v1_results = test_v1_simple_cells()
    
    # Test 3: Intégration complète
    print("\n" + "=" * 60)
    print("TEST 3: Pipeline complet (Rétine → V1 → Spikes)")
    print("=" * 60)
    integration_results = test_integration()
    
    print("\n" + "=" * 60)
    print("TOUS LES TESTS TERMINÉS AVEC SUCCÈS!")
    print("=" * 60)
    print("\nFichiers générés :")
    print("- retina_lgn_test.png")
    print("- v1_simple_cells_test.png")
    print("- gabor_filters.png")
    print("- integration_test.png")
    
    return retina_results, v1_results, integration_results


if __name__ == "__main__":
    retina_results, v1_results, integration_results = main()
