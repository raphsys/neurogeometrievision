"""
Test corrigé du module SNN.
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

def test_simple():
    """Test simple et robuste."""
    print("\n" + "="*60)
    print("TEST SNN SIMPLIFIÉ")
    print("="*60)
    
    # Test d'import
    print("1. Test d'import...")
    try:
        from neurogeomvision.snn.neurons import LIFNeuron, LIFLayer
        from neurogeomvision.snn.layers import SNNLinear, TemporalPooling
        print("✓ Import réussi")
    except Exception as e:
        print(f"✗ Erreur import: {e}")
        return False
    
    # Test LIFNeuron
    print("\n2. Test LIFNeuron...")
    try:
        lif = LIFNeuron(tau_m=20.0, v_thresh=-50.0)
        
        # Simulation simple
        current = torch.tensor(2.0)  # 2 nA
        spike, voltage = lif(current)
        
        print(f"  ✓ LIFNeuron fonctionnel")
        print(f"  Spike: {spike.item()}, Voltage: {voltage.item():.2f} mV")
    except Exception as e:
        print(f"  ✗ Erreur LIFNeuron: {e}")
        return False
    
    # Test LIFLayer
    print("\n3. Test LIFLayer...")
    try:
        layer = LIFLayer(n_neurons=5)
        
        # Courants d'entrée
        currents = torch.randn(5) * 2.0
        spikes, voltages = layer(currents)
        
        print(f"  ✓ LIFLayer fonctionnel")
        print(f"  Shape spikes: {spikes.shape}")
        print(f"  Spikes: {spikes.sum().item()}/{len(spikes)}")
    except Exception as e:
        print(f"  ✗ Erreur LIFLayer: {e}")
        return False
    
    # Test SNNLinear
    print("\n4. Test SNNLinear...")
    try:
        linear = SNNLinear(in_features=10, out_features=5)
        
        # Passe avant
        x = torch.randn(10)
        spikes, voltages = linear(x)
        
        print(f"  ✓ SNNLinear fonctionnel")
        print(f"  Input: {x.shape} -> Output: {spikes.shape}")
    except Exception as e:
        print(f"  ✗ Erreur SNNLinear: {e}")
        return False
    
    # Test TemporalPooling
    print("\n5. Test TemporalPooling...")
    try:
        pooling = TemporalPooling(window_size=3)
        
        # Série temporelle
        for i in range(5):
            x = torch.randn(4)
            pooled = pooling(x)
            if i >= 2:  # Après avoir rempli la fenêtre
                print(f"  Step {i}: pooled shape {pooled.shape}")
        
        print(f"  ✓ TemporalPooling fonctionnel")
    except Exception as e:
        print(f"  ✗ Erreur TemporalPooling: {e}")
        return False
    
    print("\n" + "="*60)
    print("✅ TESTS SNN SIMPLIFIÉS RÉUSSIS!")
    return True

def test_networks():
    """Test des réseaux."""
    print("\n" + "="*60)
    print("TEST RÉSEAUX SNN")
    print("="*60)
    
    try:
        from neurogeomvision.snn.networks import SNNClassifier, SNNVisualEncoder
        
        # Test SNNClassifier
        print("1. Test SNNClassifier...")
        classifier = SNNClassifier(
            input_size=784,
            hidden_sizes=[128, 64],
            num_classes=10,
            n_timesteps=5
        )
        
        # Données d'entrée
        x = torch.randn(1, 784)
        logits, info = classifier(x)
        
        print(f"  ✓ SNNClassifier fonctionnel")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Classe prédite: {logits.argmax().item()}")
        
        # Test SNNVisualEncoder
        print("\n2. Test SNNVisualEncoder...")
        encoder = SNNVisualEncoder(
            image_size=(32, 32),
            n_orientations=8,
            n_timesteps=3
        )
        
        # Image d'entrée
        image = torch.randn(32, 32)
        encoding, info_enc = encoder(image)
        
        print(f"  ✓ SNNVisualEncoder fonctionnel")
        print(f"  Encoding shape: {encoding.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur réseaux: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test principal."""
    print("\n" + "="*80)
    print("NEUROGEOMVISION - TESTS SNN CORRIGÉS")
    print("="*80)
    
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Test simple
    result1 = test_simple()
    
    # Test réseaux
    result2 = test_networks()
    
    # Résumé
    print("\n" + "="*80)
    print("RÉSUMÉ DES TESTS")
    print("="*80)
    
    tests = [
        ("Composants de base", result1),
        ("Architectures réseau", result2)
    ]
    
    for test_name, success in tests:
        status = "✓ PASSÉ" if success else "✗ ÉCHOUÉ"
        print(f"  {test_name:<25} {status}")
    
    n_passed = sum(1 for _, s in tests if s)
    n_total = len(tests)
    
    print(f"\nTotal: {n_passed}/{n_total} tests réussis")
    
    if n_passed == n_total:
        print("\n🎉 MODULE SNN FONCTIONNEL!")
    else:
        print(f"\n⚠ {n_total - n_passed} tests ont échoué.")
    
    return n_passed == n_total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
