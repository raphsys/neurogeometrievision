"""
Test complet du module SNN.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

print("="*80)
print("TEST COMPLET DU MODULE SNN")
print("="*80)

def test_imports():
    """Test les imports."""
    print("\n1. TEST DES IMPORTS")
    print("-" * 60)
    
    try:
        import neurogeomvision.snn
        print("✓ Module SNN importé")
        
        from neurogeomvision.snn import (
            LIFNeuron, IzhikevichNeuron, LIFLayer,
            SNNLinear, SNNConv2d, TemporalPooling,
            SNNClassifier, SNNVisualEncoder,
            encode_image_to_spikes, calculate_spike_stats
        )
        
        print("✓ Toutes les classes importées")
        return True
        
    except ImportError as e:
        print(f"✗ Erreur d'import: {e}")
        return False

def test_neurons():
    """Test les neurones."""
    print("\n2. TEST DES NEURONES")
    print("-" * 60)
    
    try:
        from neurogeomvision.snn import LIFNeuron, IzhikevichNeuron, LIFLayer
        
        # LIFNeuron
        neuron = LIFNeuron()
        spike, voltage = neuron(torch.tensor(15.0, dtype=torch.float32))
        print(f"✓ LIFNeuron: spike={spike.item()}, voltage={voltage.item():.2f}")
        
        # IzhikevichNeuron
        izh = IzhikevichNeuron()
        spike, voltage = izh(torch.tensor(10.0, dtype=torch.float32))
        print(f"✓ IzhikevichNeuron: spike={spike.item()}, voltage={voltage.item():.2f}")
        
        # LIFLayer
        layer = LIFLayer(n_neurons=10)
        spikes, voltages = layer(torch.randn(10, dtype=torch.float32))
        print(f"✓ LIFLayer: spikes shape={spikes.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return False

def test_classifier():
    """Test le classificateur."""
    print("\n3. TEST SNNCLASSIFIER")
    print("-" * 60)
    
    try:
        from neurogeomvision.snn import SNNClassifier
        
        # Créer le classificateur
        classifier = SNNClassifier(
            input_size=784,
            hidden_sizes=[256, 128],
            num_classes=10,
            n_timesteps=3
        )
        
        # Réinitialiser les états
        classifier.reset_state()
        
        # Test batch_size=1
        x1 = torch.randn(784, dtype=torch.float32)
        logits1, info1 = classifier(x1)
        
        print(f"✓ Test 1D: input={x1.shape}, logits={logits1.shape}")
        print(f"  Classe prédite: {logits1.argmax().item()}")
        
        # Test batch_size=4
        x4 = torch.randn(4, 784, dtype=torch.float32)
        classifier.reset_state()
        logits4, info4 = classifier(x4)
        
        print(f"✓ Test batch: input={x4.shape}, logits={logits4.shape}")
        
        # Vérifications
        assert logits1.shape == (10,), f"Mauvaise shape: {logits1.shape}"
        assert logits4.shape == (4, 10), f"Mauvaise shape: {logits4.shape}"
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return False

def test_utils():
    """Test les utilitaires."""
    print("\n4. TEST DES UTILITAIRES")
    print("-" * 60)
    
    try:
        from neurogeomvision.snn import encode_image_to_spikes, calculate_spike_stats
        
        # Encodage d'image
        image = torch.randn(1, 28, 28, dtype=torch.float32)
        spikes = encode_image_to_spikes(image, n_timesteps=5)
        
        print(f"✓ Encodage: image={image.shape}, spikes={spikes.shape}")
        
        # Statistiques
        stats = calculate_spike_stats(spikes)
        print(f"✓ Statistiques: mean rate={stats['mean_firing_rate']:.1f} Hz")
        print(f"  Total spikes: {stats['total_spikes']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return False

def main():
    """Exécute tous les tests."""
    print("\n" + "="*80)
    print("DÉMARRAGE DES TESTS")
    print("="*80)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Neurones", test_neurons()))
    results.append(("Classifier", test_classifier()))
    results.append(("Utilitaires", test_utils()))
    
    print("\n" + "="*80)
    print("RÉSUMÉ")
    print("="*80)
    
    all_passed = True
    for test_name, success in results:
        status = "✓ PASSÉ" if success else "✗ ÉCHOUÉ"
        print(f"  {test_name:<15} {status}")
        if not success:
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 TOUS LES TESTS RÉUSSIS !")
        print("Le module SNN est complètement fonctionnel.")
        
        # Afficher un exemple d'utilisation
        print("\n" + "="*80)
        print("EXEMPLE D'UTILISATION:")
        print("="*80)
        print("""
# Import
from neurogeomvision.snn import SNNClassifier, encode_image_to_spikes

# Créer un classificateur
classifier = SNNClassifier(
    input_size=784,
    hidden_sizes=[256, 128],
    num_classes=10,
    n_timesteps=5
)

# Réinitialiser les états
classifier.reset_state()

# Préparer les données
image = torch.randn(28, 28)
spikes = encode_image_to_spikes(image, n_timesteps=10)

# Classification
image_flat = image.reshape(1, -1)
logits, info = classifier(image_flat)

print(f"Classe prédite: {logits.argmax().item()}")
print(f"Info: {info}")
""")
    else:
        print("⚠ Certains tests ont échoué.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
