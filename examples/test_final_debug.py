"""
Debug final du module SNN.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

print("DEBUG FINAL DU MODULE SNN")
print("="*80)

def debug_snnlinear():
    """Debug SNNLinear."""
    print("\n1. DEBUG SNNLinear")
    print("-" * 60)
    
    try:
        from neurogeomvision.snn.layers import SNNLinear
        
        # Test avec input_size=10, output_size=5
        linear = SNNLinear(in_features=10, out_features=5)
        linear.reset_state()
        
        print("Test 1: Entrée 1D (10 features)")
        x1 = torch.randn(10, dtype=torch.float32)
        spikes1, voltages1 = linear(x1)
        print(f"  Input shape: {x1.shape}")
        print(f"  Spikes shape: {spikes1.shape}")
        print(f"  Expected: (5,)")
        print(f"  Spikes sum: {spikes1.sum().item()}")
        
        print("\nTest 2: Entrée 2D (batch_size=3, 10 features)")
        x2 = torch.randn(3, 10, dtype=torch.float32)
        linear.reset_state()
        spikes2, voltages2 = linear(x2)
        print(f"  Input shape: {x2.shape}")
        print(f"  Spikes shape: {spikes2.shape}")
        print(f"  Expected: (3, 5)")
        print(f"  Spikes sum: {spikes2.sum().item()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_snnclassifier():
    """Debug SNNClassifier."""
    print("\n2. DEBUG SNNClassifier")
    print("-" * 60)
    
    try:
        from neurogeomvision.snn.networks import SNNClassifier
        
        # Créer un classificateur simple
        classifier = SNNClassifier(
            input_size=10,
            hidden_sizes=[5],  # Une seule couche cachée de taille 5
            num_classes=3,
            n_timesteps=2
        )
        classifier.reset_state()
        
        print("Test 1: Entrée 1D")
        x1 = torch.randn(10, dtype=torch.float32)
        logits1, info1 = classifier(x1)
        print(f"  Input shape: {x1.shape}")
        print(f"  Logits shape: {logits1.shape}")
        print(f"  Expected: (3,)")
        print(f"  Info: {info1}")
        
        print("\nTest 2: Entrée 2D (batch_size=4)")
        x2 = torch.randn(4, 10, dtype=torch.float32)
        classifier.reset_state()
        logits2, info2 = classifier(x2)
        print(f"  Input shape: {x2.shape}")
        print(f"  Logits shape: {logits2.shape}")
        print(f"  Expected: (4, 3)")
        print(f"  Info: {info2}")
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_full_example():
    """Debug avec l'exemple complet."""
    print("\n3. DEBUG EXEMPLE COMPLET")
    print("-" * 60)
    
    try:
        from neurogeomvision.snn.networks import SNNClassifier
        
        # Exemple MNIST-like
        classifier = SNNClassifier(
            input_size=784,
            hidden_sizes=[128, 64],  # Deux couches cachées
            num_classes=10,
            n_timesteps=3
        )
        classifier.reset_state()
        
        print("Test avec entrée MNIST-like (batch_size=1)")
        x = torch.randn(1, 784, dtype=torch.float32)
        logits, info = classifier(x)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Expected: (1, 10)")
        print(f"  Classe prédite: {logits.argmax().item()}")
        print(f"  Info:")
        print(f"    - pooled_output_shape: {info['pooled_output_shape']}")
        print(f"    - last_hidden_size: {classifier.last_hidden_size}")
        print(f"    - batch_size: {info['batch_size']}")
        
        # Vérifier que les dimensions sont correctes
        assert logits.shape == (1, 10), f"Shape incorrect: {logits.shape}"
        assert info['pooled_output_shape'][1] == classifier.last_hidden_size, \
            f"Pooled shape mismatch: {info['pooled_output_shape'][1]} != {classifier.last_hidden_size}"
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Exécute le debug."""
    print("\n" + "="*80)
    print("DÉMARRAGE DU DEBUG")
    print("="*80)
    
    results = []
    
    results.append(("SNNLinear", debug_snnlinear()))
    results.append(("SNNClassifier", debug_snnclassifier()))
    results.append(("Exemple complet", debug_full_example()))
    
    print("\n" + "="*80)
    print("RÉSUMÉ DU DEBUG")
    print("="*80)
    
    all_passed = True
    for test_name, success in results:
        status = "✓ PASSÉ" if success else "✗ ÉCHOUÉ"
        print(f"  {test_name:<20} {status}")
        if not success:
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 TOUS LES TESTS DE DEBUG RÉUSSIS!")
        print("Le module SNN est maintenant fonctionnel.")
        
        # Test final rapide
        print("\nTest final rapide:")
        test_code = """
import torch
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from neurogeomvision.snn import SNNClassifier

# Créer et tester
model = SNNClassifier(input_size=784, hidden_sizes=[128, 64], num_classes=10)
model.reset_state()

# Test avec une image MNIST-like
x = torch.randn(1, 784, dtype=torch.float32)
logits, info = model(x)

print(f"Test réussi!")
print(f"Input shape: {x.shape}")
print(f"Logits shape: {logits.shape}")
print(f"Classe prédite: {logits.argmax().item()}")
        """
        
        exec(test_code)
    else:
        print("⚠ Certains tests de debug ont échoué.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
