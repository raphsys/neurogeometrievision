"""
Validation du module SNN après corrections.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

print("VALIDATION DU MODULE SNN")
print("="*60)

# Test 1: Import du module
print("\n1. Import du module SNN...")
try:
    import neurogeomvision.snn
    print("✓ neurogeomvision.snn importé avec succès")
except Exception as e:
    print(f"✗ Erreur: {e}")
    sys.exit(1)

# Test 2: Import des classes principales
print("\n2. Import des classes principales...")
classes_to_test = [
    'LIFNeuron',
    'LIFLayer', 
    'SNNLinear',
    'SNNClassifier'
]

for class_name in classes_to_test:
    try:
        exec(f"from neurogeomvision.snn import {class_name}")
        print(f"✓ {class_name} importé")
    except ImportError as e:
        print(f"✗ {class_name}: {e}")

# Test 3: Instanciation et test basique
print("\n3. Tests d'instanciation...")

# Test LIFNeuron
try:
    from neurogeomvision.snn import LIFNeuron
    neuron = LIFNeuron()
    spike, voltage = neuron(torch.tensor(2.0))
    print(f"✓ LIFNeuron: spike={spike.item()}, voltage={voltage.item():.2f}")
except Exception as e:
    print(f"✗ LIFNeuron: {e}")

# Test LIFLayer
try:
    from neurogeomvision.snn import LIFLayer
    layer = LIFLayer(n_neurons=5)
    spikes, voltages = layer(torch.randn(5))
    print(f"✓ LIFLayer: {spikes.shape} spikes, {voltages.shape} voltages")
except Exception as e:
    print(f"✗ LIFLayer: {e}")

# Test SNNLinear
try:
    from neurogeomvision.snn import SNNLinear
    linear = SNNLinear(in_features=10, out_features=5)
    spikes, voltages = linear(torch.randn(10))
    print(f"✓ SNNLinear: {spikes.shape} spikes")
except Exception as e:
    print(f"✗ SNNLinear: {e}")

# Test SNNClassifier
try:
    from neurogeomvision.snn import SNNClassifier
    classifier = SNNClassifier(input_size=784, hidden_size=128, num_classes=10)
    logits, info = classifier(torch.randn(1, 784))
    print(f"✓ SNNClassifier: logits shape={logits.shape}")
except Exception as e:
    print(f"✗ SNNClassifier: {e}")

# Test 4: Vérification des attributs disponibles
print("\n4. Attributs disponibles dans neurogeomvision.snn...")
try:
    import neurogeomvision.snn as snn
    attrs = [attr for attr in dir(snn) if not attr.startswith('_')]
    print(f"Attributs ({len(attrs)}): {attrs}")
except Exception as e:
    print(f"✗ Impossible de lister les attributs: {e}")

print("\n" + "="*60)
print("RÉSUMÉ DE VALIDATION")
print("="*60)

print("Le module SNN devrait maintenant être fonctionnel!")
print("\nPour tester complètement:")
print("python test_snn_fixed.py")
print("python test_snn.py")
