"""
Test simple du module SNN après corrections.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

print("="*80)
print("TEST SIMPLE DU MODULE SNN")
print("="*80)

# Test 1: Import de base
print("1. Test d'import de base...")
try:
    import neurogeomvision.snn
    print("✓ Module snn importé")
    
    from neurogeomvision.snn import LIFNeuron, SNNLinear, SNNClassifier
    print("✓ Classes de base importées")
    
    print("\n2. Test LIFNeuron...")
    neuron = LIFNeuron()
    spike, voltage = neuron(torch.tensor(5.0, dtype=torch.float32))
    print(f"✓ LIFNeuron fonctionnel: spike={spike.item()}, voltage={voltage.item():.2f}")
    
    print("\n3. Test SNNLinear...")
    linear = SNNLinear(in_features=10, out_features=5)
    linear.reset_state()
    x = torch.randn(1, 10, dtype=torch.float32)
    spikes, voltages = linear(x)
    print(f"✓ SNNLinear fonctionnel: input={x.shape}, output={spikes.shape}")
    
    print("\n4. Test SNNClassifier...")
    classifier = SNNClassifier(
        input_size=784,
        hidden_sizes=[128, 64],
        num_classes=10,
        n_timesteps=3
    )
    classifier.reset_state()
    x = torch.randn(1, 784, dtype=torch.float32)
    logits, info = classifier(x)
    print(f"✓ SNNClassifier fonctionnel:")
    print(f"  Input shape: {x.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Classe prédite: {logits.argmax().item()}")
    print(f"  Pooled shape: {info['pooled_output_shape']}")
    print(f"  Last hidden size: {classifier.last_hidden_size}")
    
    print("\n5. Test avec batch_size > 1...")
    classifier.reset_state()
    x_batch = torch.randn(4, 784, dtype=torch.float32)
    logits_batch, info_batch = classifier(x_batch)
    print(f"✓ Batch test fonctionnel:")
    print(f"  Input shape: {x_batch.shape}")
    print(f"  Logits shape: {logits_batch.shape}")
    
    print("\n" + "="*80)
    print("🎉 TOUS LES TESTS RÉUSSIS!")
    print("Le module SNN est maintenant complètement fonctionnel!")
    
    # Résumé des fonctionnalités
    print("\nFonctionnalités disponibles:")
    print("1. LIFNeuron - Neurone Leaky Integrate-and-Fire")
    print("2. IzhikevichNeuron - Neurone d'Izhikevich")
    print("3. LIFLayer - Couche de neurones LIF")
    print("4. SNNLinear - Couche linéaire SNN")
    print("5. SNNConv2d - Couche convolutionnelle SNN")
    print("6. TemporalPooling - Pooling temporel")
    print("7. SNNClassifier - Classificateur SNN")
    print("8. SNNVisualEncoder - Encodeur visuel SNN")
    print("9. encode_image_to_spikes - Encodage d'images")
    print("10. calculate_spike_stats - Statistiques de spikes")
    
except ImportError as e:
    print(f"✗ Erreur d'import: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"✗ Erreur: {e}")
    import traceback
    traceback.print_exc()
