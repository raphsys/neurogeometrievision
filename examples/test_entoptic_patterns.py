"""
Test rapide du module entoptic_patterns.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neurogeomvision.entoptic_patterns import (
    WilsonCowanModel,
    TuringPatterns,
    GeometricHallucinations
)


def test_wilson_cowan():
    """Test du modèle de Wilson-Cowan."""
    print("Test du modèle Wilson-Cowan...")
    
    # Crée le modèle
    model = WilsonCowanModel(spatial_shape=(80, 80))
    
    # Génère différents patterns
    patterns = {}
    
    for pattern_type in ['stripes', 'hexagons', 'spirals', 'mazes']:
        print(f"  Génération pattern: {pattern_type}")
        pattern = model.generate_pattern(pattern_type)
        patterns[pattern_type] = pattern
        
        # Réinitialise pour le prochain pattern
        model.initialize_state(noise_level=0.1)
    
    # Visualisation
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    for idx, (pattern_type, pattern) in enumerate(patterns.items()):
        if idx < len(axes):
            im = axes[idx].imshow(pattern.cpu().numpy(), cmap='hot')
            axes[idx].set_title(f"Pattern: {pattern_type}")
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx], fraction=0.046)
    
    plt.suptitle("Patterns hallucinatoires - Wilson-Cowan", fontsize=14)
    plt.tight_layout()
    plt.savefig('wilson_cowan_patterns.png', dpi=120)
    plt.close()
    
    print("✓ Wilson-Cowan testé")
    return patterns


def test_turing_patterns():
    """Test des patterns de Turing."""
    print("\nTest des patterns de Turing...")
    
    # Crée le modèle
    turing = TuringPatterns(spatial_shape=(100, 100))
    
    # Génère différents patterns
    patterns = {}
    
    for pattern_type in ['spots', 'stripes', 'labyrinth', 'hexagons']:
        print(f"  Génération pattern: {pattern_type}")
        pattern = turing.generate_pattern(pattern_type)
        patterns[pattern_type] = pattern
        
        # Réinitialise
        turing.initialize_state(noise_level=0.1)
    
    # Visualisation
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    for idx, (pattern_type, pattern) in enumerate(patterns.items()):
        if idx < len(axes):
            im = axes[idx].imshow(pattern.cpu().numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
            axes[idx].set_title(f"Turing: {pattern_type}")
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx], fraction=0.046)
    
    plt.suptitle("Patterns de Turing - Réaction-Diffusion", fontsize=14)
    plt.tight_layout()
    plt.savefig('turing_patterns.png', dpi=120)
    plt.close()
    
    print("✓ Patterns de Turing testés")
    return patterns


def test_geometric_hallucinations():
    """Test des hallucinations géométriques."""
    print("\nTest des hallucinations géométriques...")
    
    # Crée le modèle (petit pour la rapidité)
    hallucinations = GeometricHallucinations(
        spatial_shape=(64, 64),
        orientation_bins=12  # Peu d'orientations pour être rapide
    )
    
    # Génère une hallucination
    print("  Génération d'hallucination...")
    activity = hallucinations.generate_hallucination(
        pattern_type='pinwheels',
        n_steps=50  # Peu d'itérations pour être rapide
    )
    
    # Visualise
    print("  Visualisation...")
    results = hallucinations.visualize_hallucination(activity)
    
    plt.savefig('geometric_hallucinations.png', dpi=120)
    plt.close()
    
    print(f"✓ Hallucination générée: {results['classification']['type']}")
    return results


def test_combined_pipeline():
    """Test combiné des trois modèles."""
    print("\nTest combiné des modèles...")
    
    # 1. Wilson-Cowan pour l'activité corticale
    wc_model = WilsonCowanModel((60, 60))
    wc_pattern = wc_model.generate_pattern('stripes')
    
    # 2. Patterns de Turing pour la morphogénèse
    turing_model = TuringPatterns((60, 60))
    turing_pattern = turing_model.generate_pattern('spots')
    
    # 3. Hallucinations géométriques
    halluc_model = GeometricHallucinations((60, 60), orientation_bins=8)
    halluc_activity = halluc_model.generate_hallucination(n_steps=30)
    halluc_visual = halluc_model.project_to_visual_field(halluc_activity, 'max')
    
    # Visualisation combinée
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Wilson-Cowan
    im1 = axes[0, 0].imshow(wc_pattern.cpu().numpy(), cmap='hot')
    axes[0, 0].set_title("Wilson-Cowan: Stripes")
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    # Turing
    im2 = axes[0, 1].imshow(turing_pattern.cpu().numpy(), cmap='RdBu_r')
    axes[0, 1].set_title("Turing: Spots")
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    # Hallucinations
    im3 = axes[0, 2].imshow(halluc_visual.cpu().numpy(), cmap='viridis')
    axes[0, 2].set_title("Hallucinations géométriques")
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
    
    # Comparaisons
    axes[1, 0].axis('off')
    axes[1, 0].text(0.1, 0.5, 
                   "Wilson-Cowan:\n• Dynamiques corticales\n• Excitation/Inhibition\n• Patterns d'activité",
                   fontsize=10, verticalalignment='center')
    
    axes[1, 1].axis('off')
    axes[1, 1].text(0.1, 0.5,
                   "Turing:\n• Réaction-Diffusion\n• Morphogénèse\n• Patterns spatiaux",
                   fontsize=10, verticalalignment='center')
    
    axes[1, 2].axis('off')
    axes[1, 2].text(0.1, 0.5,
                   "Hallucinations:\n• Symétries E(2)\n• Espace de contact\n• Géométrie V1",
                   fontsize=10, verticalalignment='center')
    
    plt.suptitle("Modèles de Patterns Hallucinatoires", fontsize=16)
    plt.tight_layout()
    plt.savefig('combined_entoptic_patterns.png', dpi=120)
    plt.close()
    
    print("✓ Pipeline combiné testé")
    return {
        'wilson_cowan': wc_pattern,
        'turing': turing_pattern,
        'hallucinations': halluc_visual
    }


def main():
    """Tests rapides des patterns entoptiques."""
    print("=" * 70)
    print("TESTS DES PATTERNS ENTOPTIQUES ET HALLUCINATIONS")
    print("=" * 70)
    
    # Test 1: Wilson-Cowan
    print("\n[1/4] Modèle Wilson-Cowan...")
    wc_results = test_wilson_cowan()
    
    # Test 2: Turing
    print("\n[2/4] Patterns de Turing...")
    turing_results = test_turing_patterns()
    
    # Test 3: Hallucinations géométriques
    print("\n[3/4] Hallucinations géométriques...")
    halluc_results = test_geometric_hallucinations()
    
    # Test 4: Combiné
    print("\n[4/4] Pipeline combiné...")
    combined_results = test_combined_pipeline()
    
    print("\n" + "=" * 70)
    print("TESTS TERMINÉS AVEC SUCCÈS!")
    print("=" * 70)
    
    print("\n📁 Fichiers générés:")
    print("  - wilson_cowan_patterns.png")
    print("  - turing_patterns.png")
    print("  - geometric_hallucinations.png")
    print("  - combined_entoptic_patterns.png")
    
    print("\n🧠 Types de patterns générés:")
    print("  • Wilson-Cowan: Rayures, hexagones, spirales, labyrinthes")
    print("  • Turing: Taches, rayures, labyrinthes, hexagones")
    print("  • Hallucinations géométriques: Basées sur les symétries de V1")
    
    return {
        'wilson_cowan': wc_results,
        'turing': turing_results,
        'hallucinations': halluc_results,
        'combined': combined_results
    }


if __name__ == "__main__":
    results = main()
