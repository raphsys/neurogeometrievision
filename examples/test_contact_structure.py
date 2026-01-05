"""
Test du module contact_structure.
"""

import torch
import matplotlib.pyplot as plt
import sys
import os
import math 

# Ajoute le chemin du projet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neurogeomvision.contact_structure import (
    JetSpace, 
    ContactStructureV1, 
    LegendrianLifts
)


def test_jet_space():
    """Test de l'espace des jets."""
    print("Test de l'espace des jets J¹(R²)...")
    
    # Crée l'espace des jets
    jet_space = JetSpace(spatial_shape=(64, 64))
    
    # Test des coordonnées
    jet_grid = jet_space.create_jet_coordinates()
    print(f"Shape de la grille jet: {jet_grid.shape}")
    print(f"Dimensions: spatial ({jet_space.height}x{jet_space.width}) × orientations ({len(jet_space.p_range)})")
    
    # Test de la forme de contact
    test_point = torch.tensor([10.0, 20.0, 0.5])  # (x, y, p)
    omega = jet_space.contact_form(test_point)
    print(f"Forme de contact en {test_point}: ω = {omega:.3f}")
    
    # Test de relevement
    base_points = torch.tensor([[10, 20], [30, 40], [50, 60]])
    orientations = torch.tensor([0, math.pi/4, math.pi/2])  # 0°, 45°, 90°
    jet_points = jet_space.lift_from_base(base_points, orientations)
    print(f"\nPoints de base: {base_points}")
    print(f"Points jet correspondants: {jet_points}")
    
    # Test de courbure
    if jet_points.shape[0] >= 3:
        curvature = jet_space.compute_curvature(jet_points)
        print(f"Courbure: {curvature}")
    
    return jet_space


def test_contact_structure():
    """Test de la structure de contact de V1."""
    print("\n\nTest de la structure de contact de V1...")
    
    # Crée la structure de contact
    contact = ContactStructureV1(spatial_shape=(128, 128), orientation_bins=36)
    
    # Test d'invariance E(2)
    is_invariant = contact.e2_invariance(translation=(5, 5), rotation=math.pi/6)
    print(f"Invariance E(2): {is_invariant}")
    
    # Test de champ d'association
    association_field = contact.create_association_field(
        reference_orientation=math.pi/4,  # 45°
        spatial_range=10
    )
    print(f"Champ d'association shape: {association_field.shape}")
    
    # Test de transport parallèle
    start_point = torch.tensor([50.0, 50.0, 1.0])  # p=1 (45°)
    direction = torch.tensor([1.0, 0.0])  # vers la droite
    path = contact.parallel_transport(start_point, direction, steps=20)
    print(f"Transport parallèle: {path.shape[0]} points générés")
    
    # Crée une carte d'orientation de test
    orientation_map = contact.jet_space.create_orientation_field(frequency=0.05)
    
    # Test d'intégration de contour
    seed_points = [
        torch.tensor([30.0, 30.0, 0.0]),  # Horizontal
        torch.tensor([30.0, 70.0, float('inf')]),  # Vertical
    ]
    
    contours = contact.integrate_contour(
        seed_points=seed_points,
        orientation_map=orientation_map,
        max_steps=50
    )
    print(f"Contours intégrés: {len(contours)}")
    for i, contour in enumerate(contours):
        print(f"  Contour {i}: {contour.shape[0]} points")
    
    # Visualisation
    fig = contact.visualize_contact_space(plane='xy')
    plt.savefig('contact_space_xy.png', dpi=150)
    plt.close()
    
    fig = contact.visualize_contact_space(plane='xp')
    plt.savefig('contact_space_3d.png', dpi=150)
    plt.close()
    
    return contact, contours


def test_legendrian_lifts():
    """Test des relevées legendriennes."""
    print("\n\nTest des relevées legendriennes...")
    
    # Crée l'espace des jets
    jet_space = JetSpace(spatial_shape=(100, 100))
    lifts = LegendrianLifts(jet_space)
    
    # Génère différents types de contours
    contour_types = ['simple', 'circle', 'spiral']
    
    all_contours = []
    energies = []
    
    for ctype in contour_types:
        contours = lifts.generate_test_contours(contour_type=ctype)
        all_contours.extend(contours)
        
        for contour in contours:
            energy = lifts.compute_legendrian_energy(contour)
            energies.append(energy)
            print(f"Contour {ctype}, énergie legendrienne: {energy:.6f}")
    
    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, contour in enumerate(all_contours[:3]):
        ax = axes[idx]
        
        # Projection sur la base
        base_contour = contour[:, :2].cpu().numpy()
        
        # Affiche la courbe
        ax.plot(base_contour[:, 0], base_contour[:, 1], 'b-', linewidth=2)
        
        # Affiche quelques tangentes
        n_tangents = min(5, contour.shape[0])
        step = contour.shape[0] // n_tangents
        
        for i in range(0, contour.shape[0], step):
            x, y, p = contour[i].cpu().numpy()
            
            # Dessine la tangente
            length = 10
            if abs(p) < 1000:  # Évite les valeurs infinies
                dx = length / math.sqrt(1 + p**2)
                dy = p * dx
                ax.arrow(x, y, dx, dy, head_width=2, head_length=3, fc='r', ec='r')
        
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title(f"Contour {idx+1} (Énergie: {energies[idx]:.6f})")
    
    plt.tight_layout()
    plt.savefig('legendrian_contours.png', dpi=150)
    plt.close()
    
    return lifts, all_contours


def test_integration_pipeline():
    """Test du pipeline complet : V1 → Structure de contact."""
    print("\n\nTest du pipeline d'intégration...")
    
    # Étape 1: Simuler la sortie de V1 (orientation map)
    h, w = 128, 128
    device = 'cpu'
    
    # Crée une carte d'orientation avec un contour
    orientation_map = torch.zeros(h, w, device=device)
    
    # Dessine un contour circulaire
    center_x, center_y = w//2, h//2
    radius = 30
    
    for y in range(h):
        for x in range(w):
            # Distance au centre
            dx = x - center_x
            dy = y - center_y
            dist = math.sqrt(dx**2 + dy**2)
            
            if abs(dist - radius) < 2:
                # Orientation tangentielle au cercle
                if abs(dy) > 1e-6:
                    p = -dx / dy
                    theta = math.atan(p)
                else:
                    theta = math.pi/2 if dx < 0 else -math.pi/2
                
                orientation_map[y, x] = theta % math.pi
    
    # Étape 2: Crée la structure de contact
    contact = ContactStructureV1(spatial_shape=(h, w), device=device)
    
    # Étape 3: Points seeds sur le contour
    seed_points = []
    for angle in [0, math.pi/2, math.pi, 3*math.pi/2]:
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        
        # Orientation tangentielle
        if abs(math.sin(angle)) > 1e-6:
            p = -math.cos(angle) / math.sin(angle)
        else:
            p = float('inf')
        
        seed_points.append(torch.tensor([x, y, p], device=device))
    
    # Étape 4: Intègre le contour
    contours = contact.integrate_contour(
        seed_points=seed_points,
        orientation_map=orientation_map,
        max_steps=30,
        threshold=0.3
    )
    
    # Étape 5: Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Carte d'orientation
    im0 = axes[0].imshow(orientation_map.cpu().numpy(), cmap='hsv', vmin=0, vmax=math.pi)
    axes[0].set_title("Carte d'orientation (V1 output)")
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0])
    
    # Contours intégrés
    axes[1].imshow(torch.zeros(h, w).cpu().numpy(), cmap='gray')
    axes[1].set_title("Contours intégrés par structure de contact")
    axes[1].axis('off')
    
    colors = ['r', 'g', 'b', 'y']
    for idx, contour in enumerate(contours):
        if contour.shape[0] > 1:
            base_contour = contour[:, :2].cpu().numpy()
            axes[1].plot(base_contour[:, 0], base_contour[:, 1], 
                        color=colors[idx % len(colors)], linewidth=2, marker='o')
    
    plt.tight_layout()
    plt.savefig('integration_pipeline.png', dpi=150)
    plt.close()
    
    print(f"Pipeline terminé. {len(contours)} contours intégrés.")
    
    return contact, contours


def main():
    """Fonction principale de test."""
    print("=" * 70)
    print("TEST DU MODULE CONTACT_STRUCTURE")
    print("=" * 70)
    
    # Test 1: Espace des jets
    print("\n" + "=" * 70)
    print("TEST 1: Espace des 1-jets J¹(R²)")
    print("=" * 70)
    jet_space = test_jet_space()
    
    # Test 2: Structure de contact
    print("\n" + "=" * 70)
    print("TEST 2: Structure de contact de V1")
    print("=" * 70)
    contact_structure, contours1 = test_contact_structure()
    
    # Test 3: Relevées legendriennes
    print("\n" + "=" * 70)
    print("TEST 3: Relevées legendriennes")
    print("=" * 70)
    legendrian_lifts, contours2 = test_legendrian_lifts()
    
    # Test 4: Pipeline d'intégration
    print("\n" + "=" * 70)
    print("TEST 4: Pipeline complet d'intégration")
    print("=" * 70)
    contact, contours3 = test_integration_pipeline()
    
    print("\n" + "=" * 70)
    print("TESTS TERMINÉS AVEC SUCCÈS!")
    print("=" * 70)
    
    print("\nFichiers générés:")
    print("- contact_space_xy.png")
    print("- contact_space_3d.png")
    print("- legendrian_contours.png")
    print("- integration_pipeline.png")
    
    return {
        'jet_space': jet_space,
        'contact_structure': contact_structure,
        'legendrian_lifts': legendrian_lifts,
        'contours': contours1 + contours2 + contours3
    }


if __name__ == "__main__":
    results = main()
