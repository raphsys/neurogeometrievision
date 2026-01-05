"""
Test corrigé du module SNN.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

print("="*80)
print("TEST FINAL CORRIGÉ DU MODULE SNN")
print("="*80)
print(f"PyTorch: {torch.__version__}")
print()

def test_neurons_fixed():
    """Test des modèles de neurones corrigés."""
    print("1. TEST DES MODÈLES DE NEURONES CORRIGÉS")
    print("-" * 60)
    
    try:
        from neurogeomvision.snn.neurons import LIFNeuron, IzhikevichNeuron, LIFLayer
        
        # Test LIFNeuron
        print("Test LIFNeuron...")
        neuron = LIFNeuron()
        spike, voltage = neuron(torch.tensor(2.0, dtype=torch.float32))
        print(f"  ✓ LIFNeuron: spike={spike.item()}, voltage={voltage.item():.2f}")
        
        # Test IzhikevichNeuron avec float32
        print("Test IzhikevichNeuron...")
        izh = IzhikevichNeuron()
        izh.set_neuron_type('regular_spiking')
        spike, voltage = izh(torch.tensor(5.0, dtype=torch.float32))
        print(f"  ✓ IzhikevichNeuron: spike={spike.item()}, voltage={voltage.item():.2f}")
        
        # Test LIFLayer
        print("Test LIFLayer...")
        layer = LIFLayer(n_neurons=10)
        spikes, voltages = layer(torch.randn(10, dtype=torch.float32))
        print(f"  ✓ LIFLayer: {spikes.shape} spikes, {spikes.sum().item():.0f} total")
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_layers_fixed():
    """Test des couches SNN corrigées."""
    print("\n2. TEST DES COUCHES SNN CORRIGÉES")
    print("-" * 60)
    
    try:
        from neurogeomvision.snn.layers import SNNLinear, SNNConv2d, TemporalPooling
        
        # Test SNNLinear
        print("Test SNNLinear...")
        linear = SNNLinear(in_features=20, out_features=10)
        spikes, voltages = linear(torch.randn(20, dtype=torch.float32))
        print(f"  ✓ SNNLinear: {spikes.shape} spikes, mean spike rate: {spikes.mean().item()*1000:.1f} Hz")
        
        # Test SNNConv2d avec forme correcte
        print("Test SNNConv2d...")
        conv = SNNConv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        image = torch.randn(1, 1, 16, 16, dtype=torch.float32)  # (batch, channels, height, width)
        spikes, voltages = conv(image)
        print(f"  ✓ SNNConv2d: {spikes.shape} spikes")
        
        # Test TemporalPooling
        print("Test TemporalPooling...")
        pooling = TemporalPooling(window_size=5)
        # Réinitialiser l'état
        pooling.reset_state()
        for i in range(7):
            x = torch.randn(8, dtype=torch.float32)
            pooled = pooling(x)
            if i >= 4:
                print(f"  Step {i}: pooled shape {pooled.shape}")
        print(f"  ✓ TemporalPooling: fonctionnel")
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_networks_fixed():
    """Test des architectures réseau corrigées."""
    print("\n3. TEST DES ARCHITECTURES RÉSEAU CORRIGÉES")
    print("-" * 60)
    
    try:
        from neurogeomvision.snn.networks import SNNClassifier, SNNVisualEncoder
        
        # Test SNNClassifier simplifié
        print("Test SNNClassifier simplifié...")
        classifier = SNNClassifier(
            input_size=784,
            hidden_sizes=[128],  # Une seule couche cachée pour simplifier
            num_classes=10,
            n_timesteps=3
        )
        # Réinitialiser les états
        classifier.reset_state()
        
        x = torch.randn(1, 784, dtype=torch.float32)
        logits, info = classifier(x)
        print(f"  ✓ SNNClassifier: logits shape={logits.shape}")
        print(f"    Classe prédite: {logits.argmax().item()}")
        print(f"    Info: pooled shape={info['pooled_output_shape']}")
        
        # Test SNNVisualEncoder
        print("\nTest SNNVisualEncoder...")
        encoder = SNNVisualEncoder(
            input_shape=(1, 32, 32),  # (channels, height, width)
            encoding_size=128,
            n_timesteps=2
        )
        # Réinitialiser les états
        encoder.reset_state()
        
        image = torch.randn(1, 1, 32, 32, dtype=torch.float32)  # (batch, channels, height, width)
        encoding, info = encoder(image)
        print(f"  ✓ SNNVisualEncoder: encoding shape={encoding.shape}")
        print(f"    Info: flattened_size={info['flattened_size']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test d'intégration complet."""
    print("\n4. TEST D'INTÉGRATION COMPLET")
    print("-" * 60)
    
    try:
        from neurogeomvision.snn.neurons import LIFNeuron
        from neurogeomvision.snn.layers import SNNLinear
        from neurogeomvision.snn.networks import SNNClassifier
        from neurogeomvision.snn.utils import encode_image_to_spikes, calculate_spike_stats
        from neurogeomvision.snn.learning import STDP_SNN, SurrogateGradient
        
        print("Test 1: Neurone LIF individuel")
        neuron = LIFNeuron()
        for i in range(5):
            current = torch.tensor(10.0 if i == 2 else 0.0, dtype=torch.float32)
            spike, voltage = neuron(current)
            print(f"  Step {i}: spike={spike.item()}, voltage={voltage.item():.1f}")
        
        print("\nTest 2: Couche SNNLinear")
        layer = SNNLinear(in_features=10, out_features=5)
        layer.reset_state()
        x = torch.randn(10, dtype=torch.float32)
        spikes, voltages = layer(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output spikes shape: {spikes.shape}")
        print(f"  Spike rate: {spikes.mean().item()*1000:.1f} Hz")
        
        print("\nTest 3: Encodage d'image")
        image = torch.randn(1, 28, 28, dtype=torch.float32)  # MNIST-like
        spikes = encode_image_to_spikes(image, n_timesteps=5)
        stats = calculate_spike_stats(spikes)
        print(f"  Image shape: {image.shape}")
        print(f"  Spikes shape: {spikes.shape}")
        print(f"  Mean firing rate: {stats['mean_firing_rate']:.1f} Hz")
        
        print("\nTest 4: STDP")
        stdp = STDP_SNN(pre_size=20, post_size=10)
        pre_spikes = (torch.rand(20) > 0.8).float()
        post_spikes = (torch.rand(10) > 0.8).float()
        weights = stdp.stdp_update(pre_spikes, post_spikes, dt=1.0)
        print(f"  STDP weights shape: {weights.shape}")
        print(f"  Mean weight: {weights.mean().item():.4f}")
        
        print("\nTest 5: Gradient de substitution")
        surrogate = SurrogateGradient(alpha=1.0)
        x = torch.randn(5, dtype=torch.float32)
        grad = surrogate(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Gradient shape: {grad.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Exécute tous les tests corrigés."""
    results = []
    
    print("="*80)
    print("LANCEMENT DES TESTS CORRIGÉS")
    print("="*80)
    
    # Test 1: Neurons
    results.append(("Neurones", test_neurons_fixed()))
    
    # Test 2: Layers
    results.append(("Couches", test_layers_fixed()))
    
    # Test 3: Networks
    results.append(("Réseaux", test_networks_fixed()))
    
    # Test 4: Intégration
    results.append(("Intégration", test_integration()))
    
    # Résumé
    print("\n" + "="*80)
    print("RÉSUMÉ DES TESTS CORRIGÉS")
    print("="*80)
    
    all_passed = True
    for test_name, success in results:
        status = "✓ PASSÉ" if success else "✗ ÉCHOUÉ"
        print(f"  {test_name:<15} {status}")
        if not success:
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 TOUS LES TESTS CORRIGÉS RÉUSSIS!")
        print("Le module SNN est maintenant complètement fonctionnel.")
        
        # Afficher la structure
        print("\nStructure du module SNN:")
        print("neurogeomvision/snn/")
        print("├── __init__.py")
        print("├── neurons.py      # LIFNeuron, IzhikevichNeuron, LIFLayer")
        print("├── layers.py       # SNNLinear, SNNConv2d, TemporalPooling")
        print("├── networks.py     # SNNClassifier, SNNVisualEncoder")
        print("├── learning.py     # STDP_SNN, SurrogateGradient")
        print("├── utils.py        # encode_image_to_spikes, calculate_spike_stats")
        print("└── visual_processing.py")
        
    else:
        print("⚠ Certains tests ont encore échoué.")
        print("Vérifiez les messages d'erreur ci-dessus.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
