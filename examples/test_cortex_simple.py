"""
Test simple du module cortex.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

print("="*80)
print("TEST SIMPLE DU MODULE CORTEX")
print("="*80)

def test_basic_components():
    """Test des composants de base du cortex."""
    try:
        from neurogeomvision.cortex import (
            V1SimpleCell, V1ComplexCell, CorticalColumn,
            OrientationSelectivity, MotionEnergyFilter,
            ColorOpponency, SaliencyMap
        )
        
        print("✓ Import des composants réussi")
        
        device = 'cpu'
        
        # Test 1: Cellule simple V1
        print("\n1. Test cellule simple V1...")
        simple_cell = V1SimpleCell(orientation=0.0, device=device)
        test_image = torch.randn(32, 32)
        simple_response = simple_cell(test_image)
        print(f"  Entrée: {test_image.shape} -> Sortie: {simple_response.shape}")
        
        # Test 2: Cellule complexe V1
        print("\n2. Test cellule complexe V1...")
        complex_cell = V1ComplexCell(orientation=0.0, device=device)
        complex_response = complex_cell(test_image)
        print(f"  Entrée: {test_image.shape} -> Sortie: {complex_response.shape}")
        
        # Test 3: Colonne corticale
        print("\n3. Test colonne corticale...")
        column = CorticalColumn(input_size=32, device=device)
        column_results = column(test_image.unsqueeze(0))
        print(f"  Entrée: {test_image.shape}")
        print(f"  Réponses simples: {column_results['simple_responses'].shape}")
        print(f"  Réponses complexes: {column_results['complex_responses'].shape}")
        print(f"  Carte d'orientation: {column_results['orientation_map'].shape}")
        
        # Test 4: Sélectivité à l'orientation
        print("\n4. Test sélectivité à l'orientation...")
        orientation_model = OrientationSelectivity(n_orientations=8, device=device)
        orientation_results = orientation_model(test_image.unsqueeze(0))
        print(f"  Réponses: {orientation_results['responses'].shape}")
        print(f"  Carte d'orientation: {orientation_results['orientation_map'].shape}")
        
        # Test 5: Opponence des couleurs
        print("\n5. Test opponence des couleurs...")
        color_model = ColorOpponency(device=device)
        test_rgb = torch.randn(3, 32, 32)
        color_results = color_model(test_rgb)
        print(f"  Image RGB: {test_rgb.shape}")
        print(f"  Image opposée: {color_results['opponent_image'].shape}")
        print(f"  Luminance: {color_results['luminance'].shape}")
        print(f"  RG opponent: {color_results['rg_opponent'].shape}")
        
        # Test 6: Saillance
        print("\n6. Test carte de saillance...")
        saliency_model = SaliencyMap(device=device)
        saliency_results = saliency_model(test_rgb.unsqueeze(0))
        print(f"  Carte de saillance: {saliency_results['saliency_map'].shape}")
        
        print("\n" + "="*80)
        print("✅ TOUS LES TESTS DE BASE PASSÉS AVEC SUCCÈS !")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_motion_components():
    """Test des composants de mouvement."""
    try:
        from neurogeomvision.cortex import MotionEnergyFilter, DirectionSelectivity
        
        print("\n" + "="*80)
        print("TEST DES COMPOSANTS DE MOUVEMENT")
        print("="*80)
        
        device = 'cpu'
        
        # Test 1: Filtre d'énergie de mouvement
        print("\n1. Test filtre d'énergie de mouvement...")
        motion_filter = MotionEnergyFilter(direction=0.0, device=device)
        
        # Créer une séquence vidéo simple (3 frames)
        test_video = torch.randn(3, 32, 32)
        motion_response = motion_filter(test_video.unsqueeze(0))
        print(f"  Vidéo: {test_video.shape} -> Réponse: {motion_response.shape}")
        
        # Test 2: Sélectivité directionnelle
        print("\n2. Test sélectivité directionnelle...")
        direction_model = DirectionSelectivity(n_directions=8, device=device)
        direction_results = direction_model(test_video.unsqueeze(0))
        print(f"  Carte de direction: {direction_results['direction_map'].shape}")
        print(f"  Vecteur de mouvement: {direction_results['motion_vector'].shape}")
        
        print("\n" + "="*80)
        print("✅ TESTS DE MOUVEMENT PASSÉS AVEC SUCCÈS !")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_advanced_models():
    """Test des modèles avancés."""
    try:
        from neurogeomvision.cortex import (
            HierarchicalVisionModel, WhatWherePathways,
            create_ventral_stream, create_dorsal_stream
        )
        
        print("\n" + "="*80)
        print("TEST DES MODÈLES AVANCÉS")
        print("="*80)
        
        device = 'cpu'
        input_shape = (64, 64)
        
        # Test 1: Modèle hiérarchique
        print("\n1. Test modèle hiérarchique...")
        hierarchical_model = HierarchicalVisionModel(
            input_shape=input_shape,
            use_color=True,
            use_motion=False,
            device=device
        )
        test_image = torch.randn(1, 64, 64)
        hierarchical_results = hierarchical_model(test_image)
        print(f"  Entrée: {test_image.shape}")
        print(f"  Features intégrés: {hierarchical_results['integrated_features'].shape}")
        print(f"  Classification: {hierarchical_results['classification'].shape}")
        
        # Test 2: Voies ventrale/dorsale
        print("\n2. Test voies ventrale/dorsale...")
        whatwhere_model = WhatWherePathways(
            input_shape=input_shape,
            device=device
        )
        whatwhere_results = whatwhere_model(test_image)
        print(f"  Voie ventrale features: {whatwhere_results['ventral']['features'].shape}")
        print(f"  Voie dorsale position: {whatwhere_results['dorsal']['position'].shape}")
        print(f"  Voie dorsale motion: {whatwhere_results['dorsal']['motion'].shape}")
        
        # Test 3: Flux ventral
        print("\n3. Test flux ventral...")
        ventral_stream = create_ventral_stream(input_channels=1, device=device)
        test_input = torch.randn(1, 1, 64, 64)
        ventral_output = ventral_stream(test_input)
        print(f"  Entrée: {test_input.shape} -> Sortie: {ventral_output.shape}")
        
        print("\n" + "="*80)
        print("✅ TESTS DES MODÈLES AVANCÉS PASSÉS AVEC SUCCÈS !")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_system():
    """Test du système intégré."""
    try:
        from neurogeomvision.cortex import BioInspiredCortex, IntegratedVisionSystem
        
        print("\n" + "="*80)
        print("TEST DU SYSTÈME INTÉGRÉ")
        print("="*80)
        
        device = 'cpu'
        input_shape = (64, 64)
        
        # Test 1: Cortex bio-inspiré (sans rétine)
        print("\n1. Test cortex bio-inspiré...")
        cortex_model = BioInspiredCortex(
            retinal_shape=input_shape,
            cortical_shape=input_shape,
            n_ganglion_cells=50,
            use_color=False,  # Désactiver couleur pour test rapide
            include_retinotopic_mapping=False,
            device=device
        )
        
        test_image = torch.randn(1, 64, 64)
        cortex_results = cortex_model(test_image, return_intermediate=False)
        print(f"  Entrée: {test_image.shape}")
        print(f"  Sortie finale: {cortex_results['final_output'].shape}")
        print(f"  Classification hiérarchique: {cortex_results['hierarchy_classification'].shape}")
        
        # Test 2: Système visuel intégré
        print("\n2. Test système visuel intégré...")
        vision_system = IntegratedVisionSystem(
            input_shape=input_shape,
            use_retina=False,  # Désactiver rétine pour test rapide
            use_cortex=True,
            device=device
        )
        
        system_results = vision_system(test_image)
        print(f"  Sortie système: {system_results['final_output'].shape}")
        
        # Afficher les informations du système
        system_info = vision_system.get_module_info()
        print(f"  Modules chargés: {system_info['modules_loaded']}")
        
        print("\n" + "="*80)
        print("✅ TESTS DU SYSTÈME INTÉGRÉ PASSÉS AVEC SUCCÈS !")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "="*80)
    print("LANCEMENT DES TESTS COMPLETS DU MODULE CORTEX")
    print("="*80)
    
    # Exécuter les tests dans l'ordre
    test_results = []
    
    print("\n>>> Test 1: Composants de base")
    test_results.append(("Composants de base", test_basic_components()))
    
    print("\n>>> Test 2: Composants de mouvement")
    test_results.append(("Composants de mouvement", test_motion_components()))
    
    print("\n>>> Test 3: Modèles avancés")
    test_results.append(("Modèles avancés", test_advanced_models()))
    
    print("\n>>> Test 4: Système intégré")
    test_results.append(("Système intégré", test_integrated_system()))
    
    # Résumé
    print("\n" + "="*80)
    print("RÉSUMÉ DES TESTS")
    print("="*80)
    
    all_passed = True
    for test_name, passed in test_results:
        status = "✅ PASSÉ" if passed else "❌ ÉCHOUÉ"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 TOUS LES TESTS DU MODULE CORTEX PASSÉS AVEC SUCCÈS !")
    else:
        print("⚠️  CERTAINS TESTS ONT ÉCHOUÉ")
    print("="*80)
