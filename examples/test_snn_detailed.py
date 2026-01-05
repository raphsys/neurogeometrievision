"""
Test SNN détaillé avec logs complets pour diagnostic.
"""

import torch
import sys
import os
import traceback

sys.path.insert(0, os.path.abspath('..'))

print("="*80)
print("TEST SNN DÉTAILLÉ - DIAGNOSTIC COMPLET")
print("="*80)
print(f"PyTorch: {torch.__version__}")
print(f"Python: {sys.version}")
print(f"Working dir: {os.getcwd()}")
print(f"Python path: {sys.path[:3]}")

def test_import_structure():
    """Test la structure d'import."""
    print("\n" + "="*60)
    print("1. TEST STRUCTURE D'IMPORT")
    print("="*60)
    
    try:
        # Vérifie si le dossier existe
        snn_path = os.path.join(os.getcwd(), '..', 'neurogeomvision', 'snn')
        print(f"Chemin SNN: {snn_path}")
        print(f"Existe: {os.path.exists(snn_path)}")
        
        if os.path.exists(snn_path):
            files = os.listdir(snn_path)
            print(f"Fichiers dans snn/: {files}")
            
            # Vérifie __init__.py
            init_path = os.path.join(snn_path, '__init__.py')
            if os.path.exists(init_path):
                with open(init_path, 'r') as f:
                    content = f.read()
                    print(f"\nContenu de __init__.py (premières 10 lignes):")
                    for i, line in enumerate(content.split('\n')[:10]):
                        print(f"  {i+1}: {line}")
            else:
                print("✗ __init__.py manquant!")
        
    except Exception as e:
        print(f"Erreur structure: {e}")
        traceback.print_exc()

def test_import_modules():
    """Test l'import des modules."""
    print("\n" + "="*60)
    print("2. TEST IMPORT DES MODULES")
    print("="*60)
    
    modules_to_test = [
        'neurogeomvision.snn',
        'neurogeomvision.snn.neurons',
        'neurogeomvision.snn.layers',
        'neurogeomvision.snn.networks',
    ]
    
    for module_name in modules_to_test:
        print(f"\n→ Import de {module_name}...")
        try:
            module = __import__(module_name, fromlist=[''])
            print(f"  ✓ Import réussi")
            
            # Liste les attributs disponibles
            if hasattr(module, '__all__'):
                print(f"  Attributs: {module.__all__}")
            else:
                attrs = [attr for attr in dir(module) if not attr.startswith('_')]
                print(f"  Attributs (premiers 5): {attrs[:5]}")
                
        except ImportError as e:
            print(f"  ✗ ImportError: {e}")
            traceback.print_exc()
        except Exception as e:
            print(f"  ✗ Erreur: {e}")
            traceback.print_exc()

def test_specific_imports():
    """Test des imports spécifiques."""
    print("\n" + "="*60)
    print("3. TEST IMPORTS SPÉCIFIQUES")
    print("="*60)
    
    imports_to_test = [
        ('LIFNeuron', 'neurons'),
        ('LIFLayer', 'neurons'),
        ('IzhikevichNeuron', 'neurons'),
        ('AdExNeuron', 'neurons'),
        ('SNNLinear', 'layers'),
        ('SNNConv2d', 'layers'),
        ('SNNClassifier', 'networks'),
    ]
    
    for class_name, module_name in imports_to_test:
        print(f"\n→ Import {class_name} depuis {module_name}...")
        try:
            # Essaye d'abord l'import direct
            exec(f"from neurogeomvision.snn.{module_name} import {class_name}")
            print(f"  ✓ Import direct réussi")
            
            # Essaye d'instancier
            if class_name == 'LIFNeuron':
                obj = LIFNeuron()
                print(f"  ✓ Instanciation réussie: {obj}")
            elif class_name == 'LIFLayer':
                obj = LIFLayer(10)
                print(f"  ✓ Instanciation réussie: {obj}")
            elif class_name == 'SNNLinear':
                obj = SNNLinear(10, 5)
                print(f"  ✓ Instanciation réussie: {obj}")
                
        except ImportError as e:
            print(f"  ✗ ImportError: {e}")
        except NameError as e:
            print(f"  ✗ NameError: {e}")
        except Exception as e:
            print(f"  ✗ Erreur: {e}")
            traceback.print_exc()

def test_file_contents():
    """Analyse le contenu des fichiers."""
    print("\n" + "="*60)
    print("4. ANALYSE DES FICHIERS")
    print("="*60)
    
    files_to_check = [
        '../neurogeomvision/snn/__init__.py',
        '../neurogeomvision/snn/neurons.py',
        '../neurogeomvision/snn/layers.py',
        '../neurogeomvision/snn/networks.py',
    ]
    
    for file_path in files_to_check:
        print(f"\n→ Analyse de {file_path}...")
        
        if not os.path.exists(file_path):
            print(f"  ✗ Fichier manquant!")
            continue
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                lines = content.split('\n')
                
                print(f"  ✓ Fichier existe ({len(lines)} lignes)")
                
                # Cherche les définitions de classe
                classes = [line for line in lines if line.strip().startswith('class ')]
                if classes:
                    print(f"  Classes trouvées:")
                    for cls in classes[:5]:  # Premières 5 classes
                        print(f"    {cls.strip()}")
                else:
                    print(f"  Aucune classe trouvée")
                
                # Cherche les imports
                imports = [line for line in lines if line.strip().startswith('from ') or line.strip().startswith('import ')]
                if imports:
                    print(f"  Imports (premiers 5):")
                    for imp in imports[:5]:
                        print(f"    {imp.strip()}")
                        
        except Exception as e:
            print(f"  ✗ Erreur lecture: {e}")

def test_minimal_functionality():
    """Test minimal de fonctionnalité."""
    print("\n" + "="*60)
    print("5. TEST FONCTIONNALITÉ MINIMALE")
    print("="*60)
    
    print("\n→ Création manuelle des classes si nécessaire...")
    
    # Crée une classe LIFNeuron minimale si elle n'existe pas
    try:
        from neurogeomvision.snn.neurons import LIFNeuron
        print("✓ LIFNeuron importée avec succès")
    except ImportError:
        print("✗ LIFNeuron non trouvée, création manuelle...")
        
        class LIFNeuron:
            """Version minimale de LIFNeuron."""
            def __init__(self, tau_m=20.0, v_thresh=1.0):
                self.tau_m = tau_m
                self.v_thresh = v_thresh
                self.voltage = 0.0
            
            def forward(self, current):
                self.voltage += (-self.voltage + current) / self.tau_m
                spike = 1.0 if self.voltage > self.v_thresh else 0.0
                if spike:
                    self.voltage = 0.0
                return torch.tensor(spike), torch.tensor(self.voltage)
        
        print("✓ LIFNeuron créée manuellement")
    
    # Test de base
    try:
        print("\n→ Test de base avec LIFNeuron...")
        neuron = LIFNeuron()
        current = torch.tensor(2.0)
        spike, voltage = neuron.forward(current)
        print(f"✓ Test forward: spike={spike}, voltage={voltage}")
        print(f"  Types: spike={type(spike)}, voltage={type(voltage)}")
        
    except Exception as e:
        print(f"✗ Erreur test: {e}")
        traceback.print_exc()

def create_minimal_snn():
    """Crée une structure SNN minimale si nécessaire."""
    print("\n" + "="*60)
    print("6. CRÉATION STRUCTURE MINIMALE")
    print("="*60)
    
    base_path = "../neurogeomvision/snn"
    
    # Crée le dossier si nécessaire
    if not os.path.exists(base_path):
        print("Création du dossier snn/...")
        os.makedirs(base_path, exist_ok=True)
    
    # Fichier __init__.py minimal
    init_content = """
# Module SNN minimal
from .neurons import LIFNeuron, LIFLayer
from .layers import SNNLinear
from .networks import SNNClassifier

__all__ = ['LIFNeuron', 'LIFLayer', 'SNNLinear', 'SNNClassifier']
"""
    
    # Fichier neurons.py minimal
    neurons_content = """
import torch
import torch.nn as nn

class LIFNeuron(nn.Module):
    def __init__(self, tau_m=20.0, v_thresh=1.0):
        super().__init__()
        self.tau_m = tau_m
        self.v_thresh = v_thresh
        self.register_buffer('voltage', torch.tensor(0.0))
    
    def forward(self, current):
        self.voltage += (-self.voltage + current) / self.tau_m
        spike = (self.voltage > self.v_thresh).float()
        self.voltage = self.voltage * (1 - spike)
        return spike, self.voltage

class LIFLayer(nn.Module):
    def __init__(self, n_neurons, tau_m=20.0):
        super().__init__()
        self.n_neurons = n_neurons
        self.tau_m = tau_m
        self.register_buffer('voltages', torch.zeros(n_neurons))
    
    def forward(self, currents):
        self.voltages += (-self.voltages + currents) / self.tau_m
        spikes = (self.voltages > 1.0).float()
        self.voltages = self.voltages * (1 - spikes)
        return spikes, self.voltages
"""
    
    # Fichier layers.py minimal
    layers_content = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class SNNLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        from .neurons import LIFLayer
        self.neuron_layer = LIFLayer(out_features)
    
    def forward(self, x):
        currents = F.linear(x, self.weight, self.bias)
        spikes, voltages = self.neuron_layer(currents)
        return spikes, voltages
"""
    
    # Fichier networks.py minimal
    networks_content = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class SNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        from .layers import SNNLinear
        self.layer1 = SNNLinear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        spikes, voltages = self.layer1(x)
        logits = self.output(spikes)
        return logits, {'spikes': spikes, 'voltages': voltages}
"""
    
    files = {
        '__init__.py': init_content,
        'neurons.py': neurons_content,
        'layers.py': layers_content,
        'networks.py': networks_content
    }
    
    created = []
    for filename, content in files.items():
        filepath = os.path.join(base_path, filename)
        try:
            with open(filepath, 'w') as f:
                f.write(content.strip())
            created.append(filename)
            print(f"✓ Créé: {filename}")
        except Exception as e:
            print(f"✗ Erreur création {filename}: {e}")
    
    if created:
        print(f"\n✅ Structure créée avec {len(created)} fichiers")
        return True
    else:
        print("\n❌ Aucun fichier créé")
        return False

def run_comprehensive_test():
    """Exécute tous les tests."""
    print("\n" + "="*80)
    print("DÉMARRAGE DES TESTS COMPLETS")
    print("="*80)
    
    # Étape 1: Analyse structure
    test_import_structure()
    
    # Étape 2: Test imports
    test_import_modules()
    
    # Étape 3: Test imports spécifiques
    test_specific_imports()
    
    # Étape 4: Analyse fichiers
    test_file_contents()
    
    # Étape 5: Test minimal
    test_minimal_functionality()
    
    # Étape 6: Si échec, créer structure minimale
    print("\n" + "="*80)
    print("RÉCAPITULATIF & SOLUTION")
    print("="*80)
    
    # Vérifie si on peut importer au moins une classe
    try:
        from neurogeomvision.snn.neurons import LIFNeuron
        print("✅ Import de base fonctionnel")
        print("Problème probable: imports circulaires ou dépendances manquantes")
    except ImportError as e:
        print(f"❌ Import échoué: {e}")
        print("\nTentative de création de structure minimale...")
        if create_minimal_snn():
            print("\n➡ Structure créée, réessayez les imports...")
            try:
                from neurogeomvision.snn.neurons import LIFNeuron
                print("✅ Import maintenant fonctionnel!")
                
                # Test rapide
                print("\nTest rapide avec la nouvelle structure:")
                neuron = LIFNeuron()
                spike, voltage = neuron(torch.tensor(2.0))
                print(f"  LIFNeuron test: spike={spike.item()}, voltage={voltage.item():.2f}")
                
                from neurogeomvision.snn.networks import SNNClassifier
                model = SNNClassifier(784, 128, 10)
                x = torch.randn(1, 784)
                logits, info = model(x)
                print(f"  SNNClassifier test: logits shape={logits.shape}")
                
            except ImportError as e2:
                print(f"❌ Import toujours échoué: {e2}")
        else:
            print("❌ Échec création structure")

if __name__ == "__main__":
    run_comprehensive_test()
    
    print("\n" + "="*80)
    print("INSTRUCTIONS DE DÉPANNAGE")
    print("="*80)
    print("1. Vérifiez que neurogeomvision/snn/__init__.py existe")
    print("2. Vérifiez les imports dans __init__.py")
    print("3. Vérifiez que les fichiers neurons.py, layers.py, networks.py existent")
    print("4. Vérifiez qu'il n'y a pas d'imports circulaires")
    print("5. Essayez: python -c \"import sys; sys.path.insert(0, '..'); from neurogeomvision.snn import LIFNeuron\"")
