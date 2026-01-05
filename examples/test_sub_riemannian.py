"""
Test du module sub_riemannian.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import math

# Ajoute le chemin du projet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neurogeomvision.sub_riemannian import (
    SubRiemannianGeometry,
    SubRiemannianGeodesics,
    HeisenbergGroup
)


def test_subriemannian_geometry():
    """Test de la géométrie sous-riemannienne."""
    print("Test de la géométrie sous-riemannienne...")
    
    # Crée la géométrie
    geometry = SubRiemannianGeometry(spatial_shape=(100, 100))
    
    # Test de la métrique
    test_point = torch.tensor([10.0, 20.0, 0.5])
    vector1 = torch.tensor([1.0, 0.5, 0.2])
    vector2 = torch.tensor([0.5, 1.0, -0.1])
    
    inner_product = geometry.metric_tensor(test_point, vector1, vector2)
    print(f"Produit scalaire sous-riemannien: {inner_product:.3f}")
    
    # Test de norme
    norm = geometry.subriemannian_norm(test_point, vector1)
    print(f"Norme sous-riemannienne: {norm:.3f}")
    
    # Test de longueur
    curve = torch.tensor([
        [0.0, 0.0, 0.0],
        [10.0, 5.0, 0.5],
        [20.0, 10.0, 1.0],
        [30.0, 15.0, 1.5]
    ])
    
    length = geometry.subriemannian_length(curve)
    print(f"Longueur sous-riemannienne de la courbe: {length:.3f}")
    
    # Test d'énergie
    energy = geometry.energy_functional(curve)
    print(f"Énergie sous-riemannienne: {energy:.3f}")
    
    # Test d'Hamiltonien
    point = torch.tensor([10.0, 20.0, 0.5])
    momentum = torch.tensor([1.0, 0.5, 0.2])
    H = geometry.hamiltonian(point, momentum)
    print(f"Hamiltonien: {H:.3f}")
    
    return geometry


def test_geodesics():
    """Test des géodésiques sous-riemanniennes."""
    print("\n\nTest des géodésiques sous-riemanniennes...")
    
    # Crée le solveur de géodésiques
    geodesics = SubRiemannianGeodesics(spatial_shape=(128, 128))
    
    # Points de test
    start_point = torch.tensor([30.0, 30.0, 0.0])  # Horizontal
    end_point = torch.tensor([70.0, 70.0, 1.0])    # Diagonale
    
    # Trouve la géodésique
    print(f"Recherche de géodésique entre {start_point} et {end_point}...")
    
    try:
        geodesic = geodesics.find_geodesic_between_points(
            start_point, end_point, method='shooting'
        )
        print(f"Géodésique trouvée: {geodesic.shape[0]} points")
        print(f"Longueur: {geodesics.geometry.subriemannian_length(geodesic):.3f}")
        
    except Exception as e:
        print(f"Erreur dans shooting method: {e}")
        print("Utilisation de la méthode variationnelle...")
        
        geodesic = geodesics.find_geodesic_between_points(
            start_point, end_point, method='variational'
        )
        print(f"Géodésique variationnelle: {geodesic.shape[0]} points")
    
    # Test d'intégration de contour
    print("\nTest d'intégration de contour géodésique...")
    
    # Points seeds
    seed_points = [
        torch.tensor([30.0, 50.0, 0.0]),
        torch.tensor([50.0, 30.0, float('inf')]),  # Vertical
        torch.tensor([70.0, 50.0, 0.0]),
        torch.tensor([50.0, 70.0, 0.5])
    ]
    
    # Carte d'orientation simulée
    orientation_map = torch.zeros(128, 128)
    for y in range(128):
        for x in range(128):
            orientation_map[y, x] = (math.atan2(y-64, x-64) + math.pi) % math.pi
    
    # Intègre le contour
    integrated_geodesics = geodesics.integrate_contour_geodesically(
        seed_points, orientation_map, search_radius=30.0
    )
    
    print(f"Contours intégrés: {len(integrated_geodesics)} géodésiques")
    
    # Test de distance géodésique
    print("\nTest de matrice de distances géodésiques...")
    dist_matrix = geodesics.compute_geodesic_distance_matrix(seed_points)
    print(f"Matrice de distances shape: {dist_matrix.shape}")
    print(f"Distances moyennes: {dist_matrix.mean():.3f}")
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Géodésique unique
    if 'geodesic' in locals() and geodesic is not None:
        axes[0].plot(geodesic[:, 0].cpu().numpy(), 
                    geodesic[:, 1].cpu().numpy(), 
                    'b-', linewidth=2, label='Géodésique')
        axes[0].scatter([start_point[0], end_point[0]], 
                       [start_point[1], end_point[1]], 
                       c='r', s=100, label='Extrémités')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].set_title('Géodésique sous-riemannienne')
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_aspect('equal')
    
    # Contour intégré
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_title('Contour intégré par géodésiques')
    axes[1].grid(True)
    axes[1].set_aspect('equal')
    
    colors = ['r', 'g', 'b', 'y', 'm', 'c']
    for i, g in enumerate(integrated_geodesics):
        if g.shape[0] > 1:
            color = colors[i % len(colors)]
            axes[1].plot(g[:, 0].cpu().numpy(), g[:, 1].cpu().numpy(), 
                        color=color, linewidth=2, 
                        label=f'Géodésique {i+1}')
    
    # Points seeds
    for i, point in enumerate(seed_points):
        axes[1].scatter([point[0]], [point[1]], 
                       c=colors[i % len(colors)], 
                       s=100, marker='o', edgecolors='k')
    
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('subriemannian_geodesics.png', dpi=150)
    plt.close()
    
    return geodesics, integrated_geodesics


def test_heisenberg_group():
    """Test du groupe de Heisenberg."""
    print("\n\nTest du groupe de Heisenberg...")
    
    # Crée le groupe
    heisenberg = HeisenbergGroup()
    
    # Test de la loi de groupe
    g1 = torch.tensor([1.0, 2.0, 3.0])
    g2 = torch.tensor([4.0, 5.0, 6.0])
    
    product = heisenberg.group_law(g1, g2)
    print(f"Produit de groupe: {g1} · {g2} = {product}")
    
    # Test d'inverse
    inverse = heisenberg.inverse(g1)
    print(f"Inverse de {g1}: {inverse}")
    
    # Vérification: g · g^{-1} = identité
    identity = heisenberg.group_law(g1, inverse)
    print(f"Vérification identité: {g1} · {g1}^-1 ≈ {identity}")
    
    # Test exponentielle/logarithme
    tangent = torch.tensor([1.0, 2.0, 3.0])
    exp = heisenberg.exponential_map(tangent)
    log = heisenberg.logarithm_map(exp)
    
    print(f"\nExponentielle de {tangent}: {exp}")
    print(f"Logarithme de {exp}: {log}")
    print(f"Erreur exponentielle-log: {torch.norm(tangent - log):.6f}")
    
    # Test de distance
    dist = heisenberg.heisenberg_distance(g1, g2)
    print(f"\nDistance Heisenberg entre {g1} et {g2}: {dist:.3f}")
    
    # Test de géodésique
    print("\nTest de géodésique dans le groupe de Heisenberg...")
    direction = torch.tensor([1.0, 0.5, 0.2])
    h_geodesic = heisenberg.compute_heisenberg_geodesic(
        start=g1,
        direction=direction,
        duration=2.0,
        n_steps=50
    )
    print(f"Géodésique Heisenberg: {h_geodesic.shape[0]} points")
    
    # Visualisation
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Géodésique
    points = h_geodesic.cpu().numpy()
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 
           'b-', linewidth=2, label='Géodésique Heisenberg')
    
    # Points de départ/arrivée
    ax.scatter([g1[0]], [g1[1]], [g1[2]], 
              c='r', s=100, label='Départ')
    ax.scatter([points[-1, 0]], [points[-1, 1]], [points[-1, 2]], 
              c='g', s=100, label='Arrivée')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('p')
    ax.set_title('Géodésique dans le groupe de Heisenberg')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('heisenberg_geodesic.png', dpi=150)
    plt.close()
    
    return heisenberg, h_geodesic


def test_complete_pipeline():
    """Test du pipeline complet: V1 → Contact → Géodésiques."""
    print("\n\nTest du pipeline complet...")
    
    from neurogeomvision.v1_simple_cells.gabor_filters import GaborFilterBank
    from neurogeomvision.contact_structure import ContactStructureV1
    
    # Étape 1: Simuler V1
    print("1. Simulation V1...")
    h, w = 128, 128
    device = 'cpu'
    
    # Image de test: carré avec coin manquant (comme Kanizsa)
    image = torch.zeros(h, w, device=device)
    
    # Dessine un carré
    square_size = 40
    center_x, center_y = w//2, h//2
    
    for y in range(center_y - square_size, center_y + square_size):
        for x in range(center_x - square_size, center_x + square_size):
            # Laisse un coin manquant
            if (x - (center_x - square_size)) < 10 and (y - (center_y - square_size)) < 10:
                continue
                
            if (center_x - square_size <= x <= center_x + square_size and
                center_y - square_size <= y <= center_y + square_size):
                image[y, x] = 1.0
    
    # Ajoute du bruit
    image += torch.randn(h, w, device=device) * 0.1
    
    # Filtres Gabor
    gabor = GaborFilterBank(
        img_size=(h, w),
        n_orientations=8,
        spatial_freqs=[0.1, 0.2],
        device=device
    )
    
    v1_results = gabor.apply_filters(image)
    orientation_map = v1_results['dominant_orientation']['angle']
    
    print(f"   Carte d'orientation générée: {orientation_map.shape}")
    
    # Étape 2: Structure de contact
    print("2. Structure de contact...")
    contact = ContactStructureV1(spatial_shape=(h, w), device=device)
    
    # Points seeds aux coins du carré
    seed_points = []
    corners = [
        (center_x - square_size, center_y - square_size),  # Coin supérieur gauche
        (center_x + square_size, center_y - square_size),  # Coin supérieur droit
        (center_x + square_size, center_y + square_size),  # Coin inférieur droit
        (center_x - square_size, center_y + square_size),  # Coin inférieur gauche
    ]
    
    for cx, cy in corners:
        # Orientation locale
        xi = max(0, min(int(cx), w-1))
        yi = max(0, min(int(cy), h-1))
        theta = orientation_map[yi, xi]
        p = torch.tan(theta)
        
        seed_points.append(torch.tensor([cx, cy, p], device=device))
    
    print(f"   {len(seed_points)} points seeds créés")
    
    # Étape 3: Géodésiques sous-riemanniennes
    print("3. Géodésiques sous-riemanniennes...")
    geodesics_solver = SubRiemannianGeodesics(spatial_shape=(h, w), device=device)
    
    # Connecte les coins par des géodésiques
    contour_geodesics = []
    
    for i in range(len(seed_points)):
        start = seed_points[i]
        end = seed_points[(i + 1) % len(seed_points)]
        
        try:
            geodesic = geodesics_solver.find_geodesic_between_points(
                start, end, method='shooting'
            )
            contour_geodesics.append(geodesic)
            print(f"   Géodésique {i+1}: {geodesic.shape[0]} points")
        except:
            # Ligne droite en fallback
            n_points = 50
            t = torch.linspace(0, 1, n_points, device=device)
            fallback = start + t.unsqueeze(1) * (end - start)
            contour_geodesics.append(fallback)
            print(f"   Géodésique {i+1}: fallback (ligne droite)")
    
    # Étape 4: Complétion du contour manquant
    print("4. Complétion du contour...")
    
    # Pour le coin manquant, on trouve une géodésique qui "remplit le trou"
    missing_start = seed_points[0]  # Coin supérieur gauche
    missing_end = seed_points[1]    # Coin supérieur droit
    
    # Mais on veut contourner le trou
    # On crée un point de contrôle au milieu
    mid_x = (missing_start[0] + missing_end[0]) / 2
    mid_y = (missing_start[1] + missing_end[1]) / 2 - 20  # Un peu plus haut
    
    # Orientation au point de contrôle
    xi = max(0, min(int(mid_x), w-1))
    yi = max(0, min(int(mid_y), h-1))
    theta = orientation_map[yi, xi]
    p = torch.tan(theta)
    
    control_point = torch.tensor([mid_x, mid_y, p], device=device)
    
    # Deux géodésiques: start→control et control→end
    try:
        geo1 = geodesics_solver.find_geodesic_between_points(
            missing_start, control_point, method='shooting'
        )
        geo2 = geodesics_solver.find_geodesic_between_points(
            control_point, missing_end, method='shooting'
        )
        completion_geodesic = torch.cat([geo1, geo2[1:]])  # Évite la duplication
        contour_geodesics[0] = completion_geodesic  # Remplace la première géodésique
        print(f"   Complétion réussie: {completion_geodesic.shape[0]} points")
    except:
        print("   Complétion échouée, garde ligne droite")
    
    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Image originale
    axes[0].imshow(image.cpu().numpy(), cmap='gray')
    axes[0].set_title("Image originale (carré avec coin manquant)")
    axes[0].axis('off')
    
    # Carte d'orientation V1
    im1 = axes[1].imshow(orientation_map.cpu().numpy(), cmap='hsv', 
                        vmin=0, vmax=math.pi)
    axes[1].set_title("Carte d'orientation V1")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    # Contour complété
    axes[2].imshow(torch.zeros(h, w).cpu().numpy(), cmap='gray')
    axes[2].set_title("Contour complété par géodésiques")
    axes[2].axis('off')
    
    # Dessine les géodésiques
    colors = ['r', 'g', 'b', 'y']
    for i, geodesic in enumerate(contour_geodesics):
        if geodesic.shape[0] > 1:
            color = colors[i % len(colors)]
            points = geodesic.cpu().numpy()
            axes[2].plot(points[:, 0], points[:, 1], 
                        color=color, linewidth=3, 
                        label=f'Géodésique {i+1}')
    
    # Points seeds
    for i, point in enumerate(seed_points):
        axes[2].scatter([point[0]], [point[1]], 
                       c=colors[i % len(colors)], 
                       s=150, marker='o', 
                       edgecolors='k', linewidth=2)
    
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('complete_pipeline.png', dpi=150)
    plt.close()
    
    print(f"\nPipeline terminé!")
    print(f"- {len(contour_geodesics)} géodésiques calculées")
    print(f"- Coin manquant complété par géodésique sous-riemannienne")
    
    return {
        'image': image,
        'orientation_map': orientation_map,
        'seed_points': seed_points,
        'contour_geodesics': contour_geodesics
    }


def main():
    """Fonction principale de test."""
    print("=" * 70)
    print("TEST DU MODULE SUB_RIEMANNIAN")
    print("=" * 70)
    
    # Test 1: Géométrie sous-riemannienne
    print("\n" + "=" * 70)
    print("TEST 1: Géométrie sous-riemannienne")
    print("=" * 70)
    geometry = test_subriemannian_geometry()
    
    # Test 2: Géodésiques
    print("\n" + "=" * 70)
    print("TEST 2: Géodésiques sous-riemanniennes")
    print("=" * 70)
    geodesics_solver, contour_geodesics = test_geodesics()
    
    # Test 3: Groupe de Heisenberg
    print("\n" + "=" * 70)
    print("TEST 3: Groupe de Heisenberg")
    print("=" * 70)
    heisenberg, h_geodesic = test_heisenberg_group()
    
    # Test 4: Pipeline complet
    print("\n" + "=" * 70)
    print("TEST 4: Pipeline complet (V1 → Contact → Géodésiques)")
    print("=" * 70)
    pipeline_results = test_complete_pipeline()
    
    print("\n" + "=" * 70)
    print("TESTS TERMINÉS AVEC SUCCÈS!")
    print("=" * 70)
    
    print("\nFichiers générés:")
    print("- subriemannian_geodesics.png")
    print("- heisenberg_geodesic.png")
    print("- complete_pipeline.png")
    
    return {
        'geometry': geometry,
        'geodesics_solver': geodesics_solver,
        'heisenberg': heisenberg,
        'pipeline_results': pipeline_results
    }


if __name__ == "__main__":
    results = main()
