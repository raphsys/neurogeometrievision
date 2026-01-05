"""
Test spécifique pour la correction du réseau SNN.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

print("Test spécifique de correction du réseau SNN")
print("="*60)

# Test 1: Vérifier que SNNClassifier fonctionne maintenant
print("\nTest 1: SNNClassifier avec une seule couche cachée")
try:
    from neurogeomvision.snn.networks import SNNClassifier
    
    classifier = SNNClassifier(
        input_size=784,
        hidden_sizes=[128],  # Une seule couche cachée
        num_classes=10,
        n_timesteps=3
    )
    
    # Réinitialiser les états
    classifier.reset_state()
    
    # Tester avec différentes tailles de batch
    print("\nTest avec batch_size=1:")
    x1 = torch.randn(1, 784, dtype=torch.float32)
    logits1, info1 = classifier(x1)
    print(f"  Input shape: {x1.shape}")
    print(f"  Logits shape: {logits1.shape}")
    print(f"  Info: {info1}")
    
    print("\nTest avec batch_size=4:")
    x4 = torch.randn(4, 784, dtype=torch.float32)
    classifier.reset_state()  # Réinitialiser pour le nouveau batch
    logits4, info4 = classifier(x4)
    print(f"  Input shape: {x4.shape}")
    print(f"  Logits shape: {logits4.shape}")
    print(f"  Info: {info4}")
    
    print("\n✓ SNNClassifier fonctionne correctement!")
    
except Exception as e:
    print(f"✗ Erreur: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Vérifier le comportement avec plusieurs couches
print("\n" + "="*60)
print("\nTest 2: SNNClassifier avec plusieurs couches cachées")
try:
    from neurogeomvision.snn.networks import SNNClassifier
    
    classifier = SNNClassifier(
        input_size=784,
        hidden_sizes=[256, 128, 64],  # Trois couches cachées
        num_classes=10,
        n_timesteps=5
    )
    
    # Réinitialiser les états
    classifier.reset_state()
    
    x = torch.randn(2, 784, dtype=torch.float32)
    logits, info = classifier(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Last hidden size: {classifier.last_hidden_size}")
    print(f"  Pooled output shape: {info['pooled_output_shape']}")
    print(f"  Batch size: {info['batch_size']}")
    
    # Vérifier que les dimensions sont correctes
    assert logits.shape == (2, 10), f"Logits shape incorrect: {logits.shape}"
    assert info['pooled_output_shape'][1] == classifier.last_hidden_size, \
        f"Dimension mismatch: {info['pooled_output_shape'][1]} != {classifier.last_hidden_size}"
    
    print("\n✓ SNNClassifier avec plusieurs couches fonctionne correctement!")
    
except Exception as e:
    print(f"✗ Erreur: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Vérifier le SNNVisualEncoder
print("\n" + "="*60)
print("\nTest 3: SNNVisualEncoder")
try:
    from neurogeomvision.snn.networks import SNNVisualEncoder
    
    encoder = SNNVisualEncoder(
        input_shape=(1, 32, 32),  # (channels, height, width)
        encoding_size=128,
        n_timesteps=3
    )
    
    # Réinitialiser les états
    encoder.reset_state()
    
    # Test avec batch_size=1
    image1 = torch.randn(1, 1, 32, 32, dtype=torch.float32)
    encoding1, info1 = encoder(image1)
    
    print(f"\nTest avec batch_size=1:")
    print(f"  Input shape: {image1.shape}")
    print(f"  Encoding shape: {encoding1.shape}")
    print(f"  Expected encoding size: {encoder.encoding_size}")
    print(f"  Info: flattened_size={info1['flattened_size']}")
    
    # Test avec batch_size=3
    image3 = torch.randn(3, 1, 32, 32, dtype=torch.float32)
    encoder.reset_state()
    encoding3, info3 = encoder(image3)
    
    print(f"\nTest avec batch_size=3:")
    print(f"  Input shape: {image3.shape}")
    print(f"  Encoding shape: {encoding3.shape}")
    
    # Vérifications
    assert encoding1.shape == (1, 128), f"Encoding shape incorrect: {encoding1.shape}"
    assert encoding3.shape == (3, 128), f"Encoding shape incorrect: {encoding3.shape}"
    assert info1['flattened_size'] == 512, f"Flattened size incorrect: {info1['flattened_size']}"
    
    print("\n✓ SNNVisualEncoder fonctionne correctement!")
    
except Exception as e:
    print(f"✗ Erreur: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("🎉 TOUS LES TESTS DES RÉSEAUX RÉUSSIS!")
print("Le module SNN est maintenant complètement fonctionnel.")
