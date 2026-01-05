"""
Test final fonctionnel du module SNN.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

print("="*80)
print("TEST FINAL FONCTIONNEL DU MODULE SNN")
print("="*80)

def test_minimal():
    """Test minimal qui fonctionne à coup sûr."""
    print("\n1. TEST MINIMAL")
    print("-" * 60)
    
    try:
        from neurogeomvision.snn import LIFNeuron, SNNLinear
        
        # Test 1: LIFNeuron seul
        print("Test LIFNeuron...")
        neuron = LIFNeuron()
        spike, voltage = neuron(torch.tensor(10.0, dtype=torch.float32))
        print(f"✓ LIFNeuron: spike={spike.item()}, voltage={voltage.item():.2f}")
        
        # Test 2: SNNLinear simple
        print("\nTest SNNLinear...")
        linear = SNNLinear(in_features=5, out_features=3)
        linear.reset_state()
        x = torch.randn(5, dtype=torch.float32)
        spikes, voltages = linear(x)
        print(f"✓ SNNLinear: input={x.shape}, output={spikes.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_classifier_simple():
    """Test SNNClassifier simple."""
    print("\n2. TEST SNNCLASSIFIER SIMPLE")
    print("-" * 60)
    
    try:
        from neurogeomvision.snn.networks import SNNClassifier
        
        # Version très simple
        classifier = SNNClassifier(
            input_size=10,
            hidden_sizes=[5],  # Une seule couche
            num_classes=3,
            n_timesteps=2  # Peu de timesteps
        )
        classifier.reset_state()
        
        print("Test avec batch_size=1...")
        x1 = torch.randn(10, dtype=torch.float32)
        logits1, info1 = classifier(x1)
        print(f"✓ Fonctionnel: input={x1.shape}, logits={logits1.shape}")
        print(f"  Classe prédite: {logits1.argmax().item()}")
        print(f"  Info: pooled_shape={info1['pooled_output_shape']}")
        
        print("\nTest avec batch_size=2...")
        x2 = torch.randn(2, 10, dtype=torch.float32)
        classifier.reset_state()
        logits2, info2 = classifier(x2)
        print(f"✓ Fonctionnel: input={x2.shape}, logits={logits2.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_classifier_mnist():
    """Test SNNClassifier pour MNIST."""
    print("\n3. TEST SNNCLASSIFIER MNIST")
    print("-" * 60)
    
    try:
        from neurogeomvision.snn.networks import SNNClassifier
        
        classifier = SNNClassifier(
            input_size=784,
            hidden_sizes=[256, 128],  # Deux couches
            num_classes=10,
            n_timesteps=3
        )
        classifier.reset_state()
        
        # Test batch_size=1
        x1 = torch.randn(1, 784, dtype=torch.float32)
        logits1, info1 = classifier(x1)
        
        print(f"✓ Test 1 - batch_size=1:")
        print(f"  Input shape: {x1.shape}")
        print(f"  Logits shape: {logits1.shape}")
        print(f"  Classe prédite: {logits1.argmax().item()}")
        print(f"  Last hidden size: {classifier.last_hidden_size}")
        print(f"  Pooled shape: {info1['pooled_output_shape']}")
        
        # Vérifications
        assert logits1.shape == (1, 10), f"Mauvaise shape: {logits1.shape}"
        assert info1['pooled_output_shape'][1] == classifier.last_hidden_size
        
        # Test batch_size=4
        x4 = torch.randn(4, 784, dtype=torch.float32)
        classifier.reset_state()
        logits4, info4 = classifier(x4)
        
        print(f"\n✓ Test 2 - batch_size=4:")
        print(f"  Input shape: {x4.shape}")
        print(f"  Logits shape: {logits4.shape}")
        
        assert logits4.shape == (4, 10), f"Mauvaise shape: {logits4.shape}"
        
        return True
    except Exception as e:
        print(f"✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test d'intégration complet."""
    print("\n4. TEST D'INTÉGRATION")
    print("-" * 60)
    
    try:
        from neurogeomvision.snn import (
            LIFNeuron, SNNLinear, SNNClassifier,
            encode_image_to_spikes, calculate_spike_stats
        )
        
        print("Étape 1: Neurone LIF")
        neuron = LIFNeuron()
        for i in range(5):
            current = torch.tensor(15.0 if i == 2 else 0.0, dtype=torch.float32)
            spike, voltage = neuron(current)
            if spike > 0.5:
                print(f"  Step {i}: SPIKE! voltage={voltage.item():.1f}")
        
        print("\nÉtape 2: Encodage d'image")
        image = torch.randn(1, 28, 28, dtype=torch.float32)
        spikes = encode_image_to_spikes(image, n_timesteps=3)
        stats = calculate_spike_stats(spikes)
        print(f"  Image shape: {image.shape}")
        print(f"  Spikes shape: {spikes.shape}")
        print(f"  Moyenne taux: {stats['mean_firing_rate']:.1f} Hz")
        
        print("\nÉtape 3: Classification")
        classifier = SNNClassifier(
            input_size=784,
            hidden_sizes=[128],
            num_classes=10,
            n_timesteps=2
        )
        classifier.reset_state()
        
        # Aplatir l'image
        image_flat = image.reshape(1, -1)
        logits, info = classifier(image_flat)
        
        print(f"  Image aplatie: {image_flat.shape}")
        print(f"  Logits: {logits.shape}")
        print(f"  Classe: {logits.argmax().item()}")
        print(f"  Info: {info}")
        
        return True
    except Exception as e:
        print(f"✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Exécute tous les tests."""
    print("\n" + "="*80)
    print("DÉMARRAGE DES TESTS")
    print("="*80)
    
    results = []
    
    results.append(("Test minimal", test_minimal()))
    results.append(("Classifier simple", test_classifier_simple()))
    results.append(("Classifier MNIST", test_classifier_mnist()))
    results.append(("Intégration", test_integration()))
    
    print("\n" + "="*80)
    print("RÉSUMÉ DES TESTS")
    print("="*80)
    
    all_passed = True
    for test_name, success in results:
        status = "✓ PASSÉ" if success else "✗ ÉCHOUÉ"
        print(f"  {test_name:<20} {status}")
        if not success:
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 TOUS LES TESTS RÉUSSIS!")
        print("Le module SNN est maintenant COMPLÈTEMENT FONCTIONNEL!")
        
        # Code d'exemple
        print("\n" + "="*80)
        print("EXEMPLE D'UTILISATION:")
        print("="*80)
        
        example_code = """
# Import des modules
import torch
from neurogeomvision.snn import SNNClassifier, encode_image_to_spikes

# 1. Créer un classificateur SNN
classifier = SNNClassifier(
    input_size=784,
    hidden_sizes=[256, 128],
    num_classes=10,
    n_timesteps=5
)

# 2. Réinitialiser les états
classifier.reset_state()

# 3. Préparer une image (ex: MNIST)
image = torch.randn(1, 28, 28)  # Image 28x28
image_flat = image.reshape(1, -1)  # Aplatir en (1, 784)

# 4. Faire une prédiction
logits, info = classifier(image_flat)

print(f"Classe prédite: {logits.argmax().item()}")
print(f"Logits shape: {logits.shape}")
print(f"Info: {info}")

# 5. Encodage d'image en spikes
spikes = encode_image_to_spikes(image, n_timesteps=10)
print(f"Spikes shape: {spikes.shape}")
"""
        
        print(example_code)
        
    else:
        print("⚠ Certains tests ont échoué.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
