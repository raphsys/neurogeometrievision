"""
Script de profiling pour identifier les goulots d'étranglement.
"""

import torch
import time
import cProfile
import pstats
import io
import numpy as np
from functools import wraps
import sys
import os

sys.path.insert(0, os.path.abspath('.'))

def profile_func(func):
    """Décorateur pour profiler une fonction."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        profiler.disable()
        
        # Capture les stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 fonctions
        
        print(f"\n{'='*60}")
        print(f"PROFILING: {func.__name__}")
        print(f"Temps total: {elapsed:.3f} secondes")
        print(f"{'='*60}")
        print("Top 20 des fonctions les plus lentes:")
        print(s.getvalue())
        print(f"{'='*60}")
        
        return result
    return wrapper

def test_module(module_name, test_func_name, *args, **kwargs):
    """Teste et profile un module spécifique."""
    print(f"\n{'#'*80}")
    print(f"TEST ET PROFILING DU MODULE: {module_name}")
    print(f"{'#'*80}")
    
    try:
        # Import dynamique
        module = __import__(f'neurogeomvision.{module_name}', fromlist=['*'])
        
        if hasattr(module, test_func_name):
            func = getattr(module, test_func_name)
            
            # Applique le décorateur de profiling
            profiled_func = profile_func(func)
            
            # Exécute
            result = profiled_func(*args, **kwargs)
            
            print(f"✓ {module_name}.{test_func_name} terminé avec succès")
            return result
        else:
            print(f"✗ Fonction {test_func_name} non trouvée dans {module_name}")
            
    except Exception as e:
        print(f"✗ Erreur avec {module_name}: {e}")
        import traceback
        traceback.print_exc()
    
    return None

def test_all_modules():
    """Teste tous les modules principaux."""
    print("DÉBUT DU PROFILING COMPLET DE NEUROGEOMVISION")
    print("="*80)
    
    tests = [
        # (module_name, test_function, args, kwargs)
        ('retina_lgn.filters', 'apply_dog_filters', 
         (torch.randn(64, 64),), {}),
        
        ('v1_simple_cells.gabor_filters', 'apply_filters',
         (torch.randn(64, 64),), {}),
        
        ('association_field.field_models', '_create_local_field',
         (0.0,), {'field_size': 15}),
        
        ('entoptic_patterns.wilson_cowan', 'step',
         (), {'dt': 0.5}),
    ]
    
    results = {}
    for module_name, func_name, args, kwargs in tests:
        key = f"{module_name}.{func_name}"
        results[key] = test_module(module_name, func_name, *args, **kwargs)
    
    return results

def memory_usage_test():
    """Teste l'utilisation mémoire."""
    import psutil
    import os
    
    print(f"\n{'#'*80}")
    print("TEST D'UTILISATION MÉMOIRE")
    print(f"{'#'*80}")
    
    process = psutil.Process(os.getpid())
    
    # Test mémoire avant
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Crée quelques gros tenseurs
    print("Création de gros tenseurs...")
    big_tensors = []
    for i in range(5):
        tensor = torch.randn(256, 256, 8, device='cpu')
        big_tensors.append(tensor)
        mem_current = process.memory_info().rss / 1024 / 1024
        print(f"  Tenseur {i+1}: {tensor.numel():,} éléments, Mémoire: {mem_current:.1f} MB")
    
    # Test mémoire après
    mem_after = process.memory_info().rss / 1024 / 1024
    print(f"\nUtilisation mémoire:")
    print(f"  Avant: {mem_before:.1f} MB")
    print(f"  Après: {mem_after:.1f} MB")
    print(f"  Différence: {mem_after - mem_before:.1f} MB")
    
    # Libère la mémoire
    del big_tensors
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    mem_final = process.memory_info().rss / 1024 / 1024
    print(f"  Final: {mem_final:.1f} MB")

def speed_benchmark():
    """Benchmark de vitesse des opérations courantes."""
    print(f"\n{'#'*80}")
    print("BENCHMARK DE VITESSE")
    print(f"{'#'*80}")
    
    sizes = [(32, 32), (64, 64), (128, 128), (256, 256)]
    
    for h, w in sizes:
        print(f"\nTaille: {h}x{w} ({h*w:,} pixels)")
        
        # Test convolution vs boucle
        tensor = torch.randn(h, w)
        
        # Méthode boucle (lente)
        start = time.time()
        result_loop = torch.zeros_like(tensor)
        for y in range(1, h-1):
            for x in range(1, w-1):
                result_loop[y, x] = (tensor[y-1:y+2, x-1:x+2].mean())
        time_loop = time.time() - start
        
        # Méthode convolution (rapide)
        start = time.time()
        kernel = torch.ones(1, 1, 3, 3) / 9
        tensor_4d = tensor.unsqueeze(0).unsqueeze(0)
        result_conv = torch.nn.functional.conv2d(
            torch.nn.functional.pad(tensor_4d, (1, 1, 1, 1), mode='reflect'),
            kernel,
            padding=0
        ).squeeze()
        time_conv = time.time() - start
        
        # Vérifie que les résultats sont similaires
        diff = torch.abs(result_loop - result_conv).max().item()
        
        print(f"  Boucle: {time_loop:.4f}s")
        print(f"  Conv2D: {time_conv:.4f}s")
        print(f"  Speedup: {time_loop/time_conv if time_conv > 0 else 'inf':.1f}x")
        print(f"  Différence max: {diff:.6f}")

if __name__ == "__main__":
    print("PROFILING COMPLET DE NEUROGEOMVISION")
    print("Version: 1.0 - Optimisation des performances")
    print("="*80)
    
    # 1. Test de vitesse
    speed_benchmark()
    
    # 2. Test mémoire
    memory_usage_test()
    
    # 3. Profiling des modules
    print("\n" + "="*80)
    print("PROFILING INDIVIDUEL DES MODULES")
    print("="*80)
    
    results = test_all_modules()
    
    print("\n" + "="*80)
    print("RÉSUMÉ DU PROFILING")
    print("="*80)
    print("✓ Profiling terminé")
    print("✓ Vérifiez les goulots d'étranglement ci-dessus")
    print("✓ Les optimisations seront appliquées aux modules les plus lents")
