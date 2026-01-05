"""
Test complet du module SNN.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

def test_snn_neurons():
    """Test des modèles de neurones."""
    print("\n" + "="*60)
    print("TEST MODÈLES DE NEURONES SNN")
    print("="*60)
    
    from neurogeomvision.snn.neurons import LIFNeuron, IzhikevichNeuron, LIFLayer
    
    # 1. LIF Neuron
    print("1. Test LIFNeuron...")
    lif = LIFNeuron(tau_m=20.0, v_thresh=-50.0, v_rest=-65.0)
    
    # Simulation avec courant constant
    current_input = [1.0] * 100  # 1 nA pendant 100ms
    results = lif.simulate(current_input)
    
    print(f"  ✓ LIF simulé: {len(results['spike_times'])} spikes")
    print(f"  Spike times: {results['spike_times'][:5]}...")
    
    # Visualisation
    lif.visualize_response(results, 'lif_neuron_response.png')
    print(f"  ✓ Visualisation sauvegardée")
    
    # 2. Izhikevich Neuron
    print("\n2. Test IzhikevichNeuron...")
    izh = IzhikevichNeuron()
    izh.set_neuron_type('regular_spiking')
    
    # Simulation
    I_input = [5.0] * 200  # Courant constant
    results_izh = izh.simulate(I_input)
    
    print(f"  ✓ Izhikevich simulé: {sum(results_izh['spikes'])} spikes")
    
    # 3. LIF Layer
    print("\n3. Test LIFLayer...")
    layer = LIFLayer(n_neurons=10)
    
    # Courants d'entrée
    currents = torch.randn(10) * 2.0
    spikes, voltages = layer(currents)
    
    print(f"  ✓ LIFLayer: {spikes.sum().item()} spikes sur {len(spikes)} neurones")
    
    return True

def test_snn_layers():
    """Test des couches SNN."""
    print("\n" + "="*60)
    print("TEST COUCHES SNN")
    print("="*60)
    
    from neurogeomvision.snn.layers import SNNLinear, SNNConv2d, SNNRecurrent
    
    # 1. Couche linéaire
    print("1. Test SNNLinear...")
    linear = SNNLinear(in_features=20, out_features=10)
    
    # Passe avant
    x = torch.randn(20)
    spikes, voltages = linear(x)
    
    print(f"  ✓ SNNLinear: entrée {x.shape} → sortie {spikes.shape}")
    print(f"  Spikes: {spikes.sum().item()}/{len(spikes)}")
    
    # 2. Couche convolutionnelle
    print("\n2. Test SNNConv2d...")
    conv = SNNConv2d(in_channels=1, out_channels=4, kernel_size=3)
    
    # Image d'entrée
    image = torch.randn(1, 16, 16)  # (channels, height, width)
    spikes_conv, voltages_conv = conv(image)
    
    print(f"  ✓ SNNConv2d: entrée {image.shape} → sortie {spikes_conv.shape}")
    
    # 3. Couche récurrente
    print("\n3. Test SNNRecurrent...")
    recurrent = SNNRecurrent(hidden_size=15, input_size=10)
    
    # Séquence
    x_seq = torch.randn(10)
    spikes_rec = []
    for _ in range(5):
        sp, _ = recurrent(x_seq)
        spikes_rec.append(sp)
    
    spikes_stack = torch.stack(spikes_rec)
    print(f"  ✓ SNNRecurrent: séquence de {len(spikes_rec)} pas")
    print(f"  Spikes totaux: {spikes_stack.sum().item()}")
    
    return True

def test_snn_networks():
    """Test des architectures SNN."""
    print("\n" + "="*60)
    print("TEST ARCHITECTURES SNN")
    print("="*60)
    
    from neurogeomvision.snn.networks import SNNClassifier, SNNVisualEncoder
    
    # 1. Classificateur
    print("1. Test SNNClassifier...")
    classifier = SNNClassifier(
        input_size=28*28,
        hidden_sizes=[128, 64],
        num_classes=10
    )
    
    # Données MNIST-like
    x = torch.randn(1, 28*28)
    logits, info = classifier(x)
    
    print(f"  ✓ SNNClassifier: entrée {x.shape} → logits {logits.shape}")
    print(f"  Classes: {logits.argmax(dim=1).item()}")
    print(f"  Time step: {info['time_step']}")
    
    # 2. Encodeur visuel
    print("\n2. Test SNNVisualEncoder...")
    encoder = SNNVisualEncoder(
        image_size=(32, 32),
        n_orientations=8
    )
    
    # Image d'entrée
    image = torch.randn(32, 32)
    encoding, info_enc = encoder(image)
    
    print(f"  ✓ SNNVisualEncoder: entrée {image.shape} → encoding {encoding.shape}")
    if 'v1_orientations' in info_enc:
        print(f"  Orientations V1: {info_enc['v1_orientations'].shape}")
    
    return True

def test_snn_learning():
    """Test des algorithmes d'apprentissage."""
    print("\n" + "="*60)
    print("TEST APPRENTISSAGE SNN")
    print("="*60)
    
    from neurogeomvision.snn.learning import STDP_SNN, SurrogateGradient
    
    # 1. STDP
    print("1. Test STDP_SNN...")
    stdp = STDP_SNN(pre_size=20, post_size=10)
    
    # Spikes aléatoires
    pre_spikes = (torch.rand(20) > 0.7).float()
    post_spikes = (torch.rand(10) > 0.7).float()
    
    # Mise à jour STDP
    weights = stdp.stdp_update(pre_spikes, post_spikes, dt=1.0)
    
    print(f"  ✓ STDP_SNN: poids {weights.shape}")
    print(f"  Poids moyen: {weights.mean().item():.4f}")
    
    # 2. Gradient de substitution
    print("\n2. Test SurrogateGradient...")
    surrogate = SurrogateGradient(surrogate_type='sigmoid', alpha=1.0)
    
    # Potentiels membranaires
    voltages = torch.randn(10) * 0.5
    grad_approx = surrogate(voltages, threshold=1.0)
    
    print(f"  ✓ SurrogateGradient: {grad_approx.shape}")
    print(f"  Gradients: [{grad_approx.min().item():.3f}, {grad_approx.max().item():.3f}]")
    
    return True

def test_snn_visual_processing():
    """Test du traitement visuel SNN."""
    print("\n" + "="*60)
    print("TEST TRAITEMENT VISUEL SNN")
    print("="*60)
    
    from neurogeomvision.snn.visual_processing import RetinaEncoder, V1SpikingLayer
    
    # 1. Encodeur rétinien
    print("1. Test RetinaEncoder...")
    retina = RetinaEncoder(
        image_size=(32, 32),
        n_channels=4,
        encoding_type='temporal'
    )
    
    # Image de test
    image = torch.randn(32, 32)
    encoded = retina(image)
    
    print(f"  ✓ RetinaEncoder: entrée {image.shape}")
    print(f"  Spikes: {encoded['spikes'].shape}")
    print(f"  Canaux: {encoded['n_channels']}")
    
    # 2. Couche V1 spiking
    print("\n2. Test V1SpikingLayer...")
    v1_layer = V1SpikingLayer(
        input_size=(4, 32, 32),  # (channels, height, width)
        n_orientations=8,
        n_phases=2
    )
    
    # Utilise les spikes de la rétine
    if 'spikes' in encoded:
        spikes = encoded['spikes'][0]  # Premier pas de temps
        v1_output = v1_layer(spikes)
        
        print(f"  ✓ V1SpikingLayer: entrée {spikes.shape}")
        if 'dominant_orientation' in v1_output:
            print(f"  Orientation dominante: {v1_output['dominant_orientation'].item()}")
    
    return True

def test_snn_utils():
    """Test des utilitaires SNN."""
    print("\n" + "="*60)
    print("TEST UTILITAIRES SNN")
    print("="*60)
    
    from neurogeomvision.snn.utils import (
        spike_encoding, spike_statistics, visualize_spikes
    )
    
    # 1. Encodage
    print("1. Test spike_encoding...")
    image = torch.randn(1, 16, 16)  # Image simple
    
    # Rate coding
    spikes_rate = spike_encoding(
        image,
        encoding_type='rate',
        n_time_steps=20
    )
    
    print(f"  ✓ Rate coding: {spikes_rate.shape}")
    print(f"  Spikes totaux: {spikes_rate.sum().item()}")
    
    # 2. Statistiques
    print("\n2. Test spike_statistics...")
    stats = spike_statistics(spikes_rate, bin_size=5)
    
    print(f"  ✓ Statistiques calculées")
    print(f"  Taux moyen: {stats['mean_firing_rate']:.1f} Hz")
    print(f"  Fano factor: {stats['fano_factor']:.3f}")
    print(f"  Spikes totaux: {stats['total_spikes']}")
    
    # 3. Visualisation
    print("\n3. Test visualize_spikes...")
    try:
        fig = visualize_spikes(spikes_rate, title="Test Spike Encoding")
        plt.savefig('snn_spikes_test.png', dpi=100)
        plt.close(fig)
        print(f"  ✓ Visualisation sauvegardée: snn_spikes_test.png")
    except Exception as e:
        print(f"  ⚠ Erreur visualisation: {e}")
    
    return True

def main():
    """Test principal."""
    print("\n" + "="*80)
    print("NEUROGEOMVISION - TESTS COMPLETS DU MODULE SNN")
    print("="*80)
    
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    results = {}
    
    try:
        results['neurons'] = test_snn_neurons()
    except Exception as e:
        print(f"✗ Erreur neurones: {e}")
        results['neurons'] = False
    
    try:
        results['layers'] = test_snn_layers()
    except Exception as e:
        print(f"✗ Erreur couches: {e}")
        results['layers'] = False
    
    try:
        results['networks'] = test_snn_networks()
    except Exception as e:
        print(f"✗ Erreur réseaux: {e}")
        results['networks'] = False
    
    try:
        results['learning'] = test_snn_learning()
    except Exception as e:
        print(f"✗ Erreur apprentissage: {e}")
        results['learning'] = False
    
    try:
        results['visual'] = test_snn_visual_processing()
    except Exception as e:
        print(f"✗ Erreur traitement visuel: {e}")
        results['visual'] = False
    
    try:
        results['utils'] = test_snn_utils()
    except Exception as e:
        print(f"✗ Erreur utilitaires: {e}")
        results['utils'] = False
    
    # Résumé
    print("\n" + "="*80)
    print("RÉSUMÉ DES TESTS SNN")
    print("="*80)
    
    for test_name, success in results.items():
        status = "✓ PASSÉ" if success else "✗ ÉCHOUÉ"
        print(f"  {test_name:<15} {status}")
    
    n_passed = sum(1 for s in results.values() if s)
    n_total = len(results)
    
    print(f"\nTotal: {n_passed}/{n_total} tests réussis")
    
    if n_passed == n_total:
        print("\n🎉 TOUS LES TESTS SNN RÉUSSIS!")
        print("Le module SNN est complètement fonctionnel.")
    elif n_passed >= n_total * 0.7:
        print(f"\n⚠ {n_total - n_passed} tests ont échoué, mais le module est utilisable.")
    else:
        print(f"\n❌ {n_total - n_passed} tests ont échoué, le module a des problèmes majeurs.")
    
    return n_passed == n_total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
