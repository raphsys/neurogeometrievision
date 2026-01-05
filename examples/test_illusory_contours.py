"""
Test rapide du module illusory_contours.
"""

import torch
import matplotlib.pyplot as plt
import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neurogeomvision.illusory_contours import (
    KanizsaTriangle,
    KanizsaSquare,
    ModalCompletion,
    EhrensteinIllusion
)


def test_kanizsa_triangle():
    """Test du triangle de Kanizsa."""
    print("Test du triangle de Kanizsa...")
    
    kanizsa = KanizsaTriangle(size=200)
    results = kanizsa.visualize_kanizsa(show_prediction=True)
    
    plt.savefig('kanizsa_triangle.png', dpi=120)
    plt.close()
    
    print("✓ Triangle de Kanizsa généré")
    return results


def test_kanizsa_square():
    """Test du carré de Kanizsa."""
    print("\nTest du carré de Kanizsa...")
    
    kanizsa = KanizsaSquare(size=200)
    stimulus = kanizsa.generate_stimulus()
    horiz, vert = kanizsa.predict_contours(stimulus)
    
    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(stimulus.cpu().numpy(), cmap='gray')
    axes[0].set_title("Stimulus")
    axes[0].axis('off')
    
    axes[1].imshow(horiz.cpu().numpy(), cmap='hot')
    axes[1].set_title("Contours horizontaux")
    axes[1].axis('off')
    
    axes[2].imshow(vert.cpu().numpy(), cmap='hot')
    axes[2].set_title("Contours verticaux")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('kanizsa_square.png', dpi=120)
    plt.close()
    
    print("✓ Carré de Kanizsa généré")
    return stimulus, horiz, vert


def test_modal_completion():
    """Test de complétion modale simple."""
    print("\nTest de complétion modale...")
    
    completion = ModalCompletion()
    
    # Génère des fragments
    fragments = completion.generate_fragmented_line(n_fragments=4)
    
    # Complète le contour avec méthode LINÉAIRE (plus stable)
    contour = completion.complete_contour(fragments, method='linear')  # 'linear' au lieu de 'bezier'
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Fragments
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_aspect('equal')
    axes[0].set_title("Fragments alignés")
    axes[0].grid(True)
    
    for x, y, theta in fragments:
        # Dessine un petit segment orienté
        length = 0.05
        dx = length * math.cos(theta)
        dy = length * math.sin(theta)
        
        axes[0].plot([x - dx/2, x + dx/2], 
                    [y - dy/2, y + dy/2], 
                    'b-', linewidth=3)
        axes[0].scatter([x], [y], c='r', s=50)
    
    # Ajoute des étiquettes
    for i, (x, y, _) in enumerate(fragments):
        axes[0].text(x, y + 0.05, f'F{i+1}', 
                    ha='center', va='bottom', fontsize=10)
    
    # Contour complété
    axes[1].imshow(contour.cpu().numpy(), cmap='hot')
    axes[1].set_title("Contour complété (méthode linéaire)")
    axes[1].axis('off')
    
    # Ajoute les positions des fragments sur le contour
    size = 256
    for x, y, _ in fragments:
        px = int(x * size)
        py = int(y * size)
        axes[1].scatter([px], [py], c='blue', s=50, 
                       edgecolors='white', linewidth=2)
    
    plt.tight_layout()
    plt.savefig('modal_completion.png', dpi=120)
    plt.close()
    
    print("✓ Complétion modale testée (méthode linéaire)")
    return fragments, contour


def test_ehrenstein():
    """Test de l'illusion d'Ehrenstein."""
    print("\nTest de l'illusion d'Ehrenstein...")
    
    ehrenstein = EhrensteinIllusion(size=200)
    stimulus = ehrenstein.generate_stimulus(n_lines=16)
    contour = ehrenstein.predict_contour(stimulus)
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(stimulus.cpu().numpy(), cmap='gray')
    axes[0].set_title("Stimulus d'Ehrenstein")
    axes[0].axis('off')
    
    axes[1].imshow(contour.cpu().numpy(), cmap='hot')
    axes[1].set_title("Contour illusoire prédit")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('ehrenstein_illusion.png', dpi=120)
    plt.close()
    
    print("✓ Illusion d'Ehrenstein générée")
    return stimulus, contour


def main():
    """Tests rapides des contours illusoires."""
    print("=" * 70)
    print("TESTS RAPIDES DES CONTOURS ILLUSOIRES")
    print("=" * 70)
    
    # Test 1: Triangle de Kanizsa
    print("\n[1/4] Triangle de Kanizsa...")
    kanizsa_triangle = test_kanizsa_triangle()
    
    # Test 2: Carré de Kanizsa
    print("\n[2/4] Carré de Kanizsa...")
    kanizsa_square = test_kanizsa_square()
    
    # Test 3: Complétion modale
    print("\n[3/4] Complétion modale...")
    modal = test_modal_completion()
    
    # Test 4: Ehrenstein
    print("\n[4/4] Illusion d'Ehrenstein...")
    ehrenstein = test_ehrenstein()
    
    print("\n" + "=" * 70)
    print("TESTS TERMINÉS AVEC SUCCÈS!")
    print("=" * 70)
    
    print("\n📁 Fichiers générés:")
    print("  - kanizsa_triangle.png")
    print("  - kanizsa_square.png")
    print("  - modal_completion.png")
    print("  - ehrenstein_illusion.png")
    
    return {
        'kanizsa_triangle': kanizsa_triangle,
        'kanizsa_square': kanizsa_square,
        'modal_completion': modal,
        'ehrenstein': ehrenstein
    }


if __name__ == "__main__":
    results = main()
