Excellent ! Je vais créer des fichiers README détaillés pour chaque module. Commençons par le premier.

---

## **1. README pour `retina_lgn/filters.py`**

**`NeuroGeomVision/neurogeomvision/retina_lgn/README_filters.md`**

```markdown
# Module `filters.py` - Filtres Rétine et Corps Genouillé Latéral (LGN)

## Objectif
Ce module implémente les filtres de base qui simulent le traitement visuel précoce dans la rétine et le corps genouillé latéral (LGN), selon le modèle neurogéométrique de Petitot.

## Concepts Clés

### 1. **Difference of Gaussians (DoG)**
- Modélise les champs récepteurs centre-périphérie des cellules ganglionnaires
- Centre excitateur entouré d'une périphérie inhibitrice
- Formule : `DoG(x,y) = G_centre(x,y) - α·G_périphérie(x,y)`

### 2. **Voies Parvocellulaires (X) et Magnocellulaires (Y)**
- **Parvocellulaire (X)** : Haute résolution spatiale, basse résolution temporelle
  - Filtres DoG standard
  - Détecte les contrastes spatiaux, les formes statiques
- **Magnocellulaire (Y)** : Basse résolution spatiale, haute résolution temporelle
  - Filtres DoG avec dérivée temporelle
  - Détecte le mouvement, les changements temporels

## Classes Principales

### `RetinaLGN_Filters` (Classe Statique)

#### Méthodes principales :

1. **`gaussian_kernel(size, sigma)`**
   - Crée un noyau gaussien 2D discret
   - Utilisé comme composant de base pour les filtres DoG

2. **`dog_kernel(size, sigma_center, sigma_surround, ...)`**
   - Génère un filtre Difference of Gaussians
   - Paramètres :
     - `sigma_center` : Écart-type de la gaussienne centrale
     - `sigma_surround` : Écart-type de la gaussienne périphérique
     - `center_weight` : Poids du centre (généralement 1.0)
     - `surround_weight` : Poids de la périphérie (≈0.85-0.95)

3. **`temporal_derivative_dog(...)`**
   - Crée un filtre DoG avec composante temporelle
   - Simule la réponse phasique des cellules magnocellulaires
   - Pour une vraie implémentation 3D, devrait traiter des vidéos

4. **`apply_filter(image, kernel, padding='same')`**
   - Applique un filtre 2D par convolution
   - Supporte les images simples (2D) ou les batches (4D)

### `ParvoMagnoPathway` (Classe d'Intégration)

#### Initialisation :
```python
pathway = ParvoMagnoPathway(img_size=(128, 128), device='cpu')
```

#### Paramètres configurables :
- **Parvo** : Petits filtres (taille ~15), sigmas petits (1.0, 2.0)
- **Magno** : Grands filtres (taille ~21), sigmas grands (2.0, 4.0)

#### Méthode principale :

**`process_frame(image)`**
- Traite une image à travers les deux voies parallèles
- Retourne un dictionnaire avec :
  - `parvo` : Réponse de la voie parvocellulaire
  - `magno` : Réponse de la voie magnocellulaire
  - `parvo_kernel`, `magno_kernel` : Les filtres utilisés

**`local_normalization(response, window_size=7)`**
- Normalisation locale pour simuler l'adaptation neuronale
- Calcule moyenne et écart-type locaux
- Normalise : `(response - mean_local) / (std_local + epsilon)`

## Utilisation Typique

```python
from neurogeomvision.retina_lgn.filters import ParvoMagnoPathway
import torch

# Créer une image de test
image = torch.randn(128, 128)

# Initialiser le processeur
processor = ParvoMagnoPathway(img_size=image.shape)

# Traiter l'image
results = processor.process_frame(image)

# Accéder aux résultats
parvo_response = results['parvo']    # Réponse spatiale fine
magno_response = results['magno']    # Réponse temporelle/mouvement
```

## Représentation Mathématique

### Filtre DoG :
```
DoG(x,y) = A·exp(-(x²+y²)/(2σ₁²)) - B·exp(-(x²+y²)/(2σ₂²))
```

### Normalisation Locale :
```
R_norm(x,y) = [R(x,y) - μ_local(x,y)] / [σ_local(x,y) + ε]
où μ_local et σ_local sont calculés sur une fenêtre locale
```

## Relation avec la Neurophysiologie

1. **Cellules ON-centre/OFF-périphérie** : Réponse positive au centre, négative à la périphérie
2. **Codage d'Opposition** : Améliore la détection des contrastes
3. **Séparation Parvo/Magno** : Correspond aux voies dorsale ("where") et ventrale ("what")

## Notes d'Implémentation

- Utilise PyTorch pour les opérations de convolution (même en CPU)
- Les filtres sont normalisés pour avoir une somme nulle (pas de biais DC)
- La normalisation locale améliore la robustesse aux variations d'éclairage
- Conçu pour être extensible à des implémentations GPU
```

---

## **2. README pour `retina_lgn/coding.py`**

**`NeuroGeomVision/neurogeomvision/retina_lgn/README_coding.md`**

```markdown
# Module `coding.py` - Codage Neuronal en Spikes

## Objectif
Ce module implémente différentes stratégies de codage neuronal qui transforment des intensités analogiques en trains d'impulsions (spikes) discrètes, inspirées des mécanismes biologiques du système visuel.

## Concepts Clés

### 1. **Codage par Fréquence (Rate Coding)**
- L'intensité est codée par la fréquence de décharge
- Plus l'intensité est forte, plus le neurone décharge souvent
- Simple mais peu plausible biologiquement pour le traitement rapide

### 2. **Codage par Rang (Rank Coding - Simon Thorpe)**
- Les neurones avec les intensités les plus fortes déchargent en premier
- Permet un traitement ultra-rapide (20-30ms pour la reconnaissance)
- Biologiquement plausible pour la vision rapide

### 3. **Codage par Latence (Latency Coding)**
- L'intensité détermine le délai avant la première décharge
- Forte intensité = courte latence
- Utilisé dans les systèmes sensoriels biologiques

## Classes Principales

### `SpikeEncoder` (Classe Statique)

#### Méthodes principales :

1. **`rate_coding(intensity, max_rate=100.0, time_steps=10)`**
   ```python
   # intensity: [0, 1] normalisée
   # max_rate: Fréquence max en Hz
   # time_steps: Nombre d'étapes temporelles
   ```
   - Calcule la probabilité de décharge : `p = intensity × max_rate × dt`
   - Génère des spikes aléatoires selon cette probabilité
   - Retourne un tensor binaire `(time_steps, *shape)`

2. **`rank_coding(intensity, time_steps=10)`**
   - Trie les pixels par intensité décroissante
   - Assigne un temps de décharge proportionnel au rang
   - Premier rang (intensité max) → temps 0
   - Dernier rang (intensité min) → `time_steps-1`

3. **`latency_coding(intensity, max_latency=50.0, time_steps=10)`**
   - Calcule la latence : `latency = (1 - intensity) × max_latency`
   - Convertit la latence en indice temporel
   - Génère un spike au temps calculé

### `TemporalProcessor`

#### Modèle d'Intégration à Fuite (Leaky Integrate)
Simule la dynamique du potentiel membranaire des neurones.

**Équation différentielle :**
```
dv/dt = -v/τ + I(t)
```

où :
- `v` : Potentiel membranaire
- `τ` : Constante de temps membranaire (20ms typique)
- `I(t)` : Courant d'entrée (spikes)

**Implémentation discrète :**
```python
v[t+1] = α·v[t] + I[t]
α = exp(-dt/τ)
```

#### Méthode principale :

**`leaky_integrate(spikes, init_v=0.0)`**
- Intègre les spikes dans le temps
- Simule la charge/décharge du potentiel membranaire
- Retourne l'évolution temporelle du potentiel

## Utilisation Typique

```python
from neurogeomvision.retina_lgn.coding import SpikeEncoder, TemporalProcessor
import torch

# Créer une intensité normalisée
intensity = torch.rand(64, 64)  # Shape: (H, W)

# Encoder en spikes
encoder = SpikeEncoder()

# 1. Codage par fréquence
rate_spikes = encoder.rate_coding(intensity, max_rate=50, time_steps=20)
print(f"Rate spikes shape: {rate_spikes.shape}")  # (20, 64, 64)
print(f"Total spikes: {rate_spikes.sum().item()}")

# 2. Codage par rang
rank_spikes = encoder.rank_coding(intensity, time_steps=20)
# Les spikes sont plus concentrés au début

# 3. Intégration temporelle
processor = TemporalProcessor(tau=20.0, dt=1.0)
voltage = processor.leaky_integrate(rate_spikes)
print(f"Voltage shape: {voltage.shape}")  # (20, 64, 64)
```

## Comparaison des Stratégies de Codage

| Codage | Vitesse | Plausibilité | Complexité | Usage |
|--------|---------|--------------|------------|--------|
| Rate | Lent | Faible | Simple | Baseline |
| Rank | Très rapide | Haute | Moyenne | Vision rapide |
| Latency | Rapide | Haute | Simple | Traitement précoce |

## Représentation Mathématique

### Rate Coding :
```
P(spike à t) = f(I) × dt
f(I) = I × f_max  (fonction de transfert linéaire)
```

### Rank Coding :
```
t_décharge(i) = floor(rang(i) / N × T)
où rang(i) ∈ [0, N-1] est le rang décroissant
```

### Leaky Integration :
```
v[n+1] = exp(-dt/τ)·v[n] + I[n]
```

## Relation avec la Neurophysiologie

1. **Potentiels d'Action** : Les spikes sont des événements discrets tout-ou-rien
2. **Période Réfractaire** : Non implémentée ici (à ajouter)
3. **Adaptation** : Les neurones réduisent leur sensibilité avec une stimulation prolongée

## Applications

1. **Simulation SNN** : Interface entre traitement analogique et réseaux de spikes
2. **Vision Événementielle** : Compatible avec les caméras neuromorphiques (DVS)
3. **Traitement Temporel** : Capture la dynamique des stimuli visuels

## Extensions Possibles

1. **Codage Temporel Précis** : Utiliser des temps de spike continus
2. **Période Réfractaire** : Empêcher les décharges trop rapprochées
3. **Adaptation** : Réduire la sensibilité avec le temps
4. **Codage de Population** : Combiner plusieurs neurones par caractéristique
```

---

## **3. README pour `v1_simple_cells/gabor_filters.py`**

**`NeuroGeomVision/neurogeomvision/v1_simple_cells/README_gabor.md`**

```markdown
# Module `gabor_filters.py` - Neurones Simples du Cortex V1

## Objectif
Ce module implémente des filtres de Gabor pour modéliser les neurones simples de l'aire visuelle primaire V1, qui détectent des orientations et fréquences spatiales spécifiques selon le modèle neurogéométrique.

## Concepts Clés

### 1. **Filtres de Gabor**
- Modèle standard des champs récepteurs des neurones simples de V1
- Combinaison d'une enveloppe gaussienne et d'une onde sinusoïdale
- Détectent des orientations et fréquences spatiales spécifiques

### 2. **Organisation en Hypercolonnes**
- Neurones sensibles à la même orientation regroupés en colonnes
- Toutes les orientations représentées dans une hypercolonne (~1mm)
- Organisation rétinotopique préservée

### 3. **Pinwheels (Roues d'Orientation)**
- Points singuliers où convergent toutes les orientations
- Organisation en vortex observée expérimentalement
- Chiralité alternée (dextrogyre/lévogyre)

## Classe Principale : `GaborFilterBank`

### Initialisation :
```python
gabor_bank = GaborFilterBank(
    img_size=(128, 128),      # Taille des images d'entrée
    n_orientations=8,         # Nombre d'orientations (0 à π)
    spatial_freqs=[0.1, 0.2], # Fréquences spatiales (cycles/pixel)
    phases=[0, 1.57],         # Phases (0=cos, π/2=sin)
    device='cpu'
)
```

### Paramètres des Filtres :

1. **Orientation (θ)** : Angle du filtre (0 = horizontal, π/2 = vertical)
2. **Fréquence Spatiale (f)** : Nombre de cycles par pixel
   - Basses fréquences : Grandes structures
   - Hautes fréquences : Détails fins
3. **Phase (φ)** : Décalage de l'onde sinusoïdale
   - φ=0 : Symétrique (pair)
   - φ=π/2 : Anti-symétrique (impair)
4. **Sigma (σ)** : Largeur de l'enveloppe gaussienne
   - σ ≈ 0.56/f (relation typique)
5. **Gamma (γ)** : Élasticité (rapport d'aspect)
   - γ < 1 : Filtre allongé
   - γ = 1 : Filtre circulaire

### Méthodes Principales

#### 1. **`apply_filters(image)`**
Applique toute la banque de filtres à une image.

**Retourne un dictionnaire avec :**
- `filter_responses` : Réponses individuelles de chaque filtre
- `dominant_orientation` : 
  - `angle` : Orientation dominante par pixel
  - `coherence` : Mesure de cohérence locale (0-1)
  - `amplitude` : Force de la réponse
- `orientation_map` : Carte d'orientation théorique (pinwheels)

#### 2. **`visualize_filters(n_filters=12)`**
Affiche un sous-ensemble des filtres avec leurs paramètres.

#### 3. **`_create_orientation_map()`**
Génère une carte d'orientation simulée avec des pinwheels.

### Calcul de l'Orientation Dominante

Méthode vectorielle standard :
```
C = Σ R(θ)·cos(2θ)
S = Σ R(θ)·sin(2θ)
θ_dominant = 0.5·atan2(S, C)
cohérence = √(C²+S²) / (max_response·n_orientations)
```

## Utilisation Typique

```python
from neurogeomvision.v1_simple_cells.gabor_filters import GaborFilterBank
import torch
import matplotlib.pyplot as plt

# Créer une image avec bords orientés
image = torch.zeros(128, 128)
image[30:50, 20:100] = 1.0  # Bande horizontale
image[70:90, 20:100] = 1.0  # Bande horizontale

# Initialiser la banque de filtres
gabor = GaborFilterBank(
    img_size=image.shape,
    n_orientations=8,
    spatial_freqs=[0.1, 0.2],
    phases=[0, 1.57]
)

# Appliquer les filtres
results = gabor.apply_filters(image)

# Visualiser les résultats
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Orientation dominante
orient_img = axes[0, 0].imshow(results['dominant_orientation']['angle'].cpu().numpy(),
                               cmap='hsv', vmin=0, vmax=3.14159)
axes[0, 0].set_title("Orientation Dominante")
plt.colorbar(orient_img, ax=axes[0, 0])

# Cohérence
coher_img = axes[0, 1].imshow(results['dominant_orientation']['coherence'].cpu().numpy(),
                              cmap='hot', vmin=0, vmax=1)
axes[0, 1].set_title("Cohérence d'Orientation")
plt.colorbar(coher_img, ax=axes[0, 1])

# Amplitude
amp_img = axes[1, 0].imshow(results['dominant_orientation']['amplitude'].cpu().numpy(),
                            cmap='gray')
axes[1, 0].set_title("Amplitude de Réponse")
plt.colorbar(amp_img, ax=axes[1, 0])

# Carte d'orientation théorique
map_img = axes[1, 1].imshow(results['orientation_map'].cpu().numpy(),
                            cmap='hsv')
axes[1, 1].set_title("Carte d'Orientation (Pinwheels)")
plt.colorbar(map_img, ax=axes[1, 1])

plt.tight_layout()
plt.show()

# Visualiser les filtres
gabor.visualize_filters(n_filters=8)
```

## Représentation Mathématique

### Filtre de Gabor 2D :
```
G(x,y) = exp(-(x'² + γ²y'²)/(2σ²)) × cos(2πf x' + φ)

où :
x' =  x·cosθ + y·sinθ
y' = -x·sinθ + y·cosθ
```

### Propriétés :
1. **Localisation Optimale** : Minimise l'incertitude espace-fréquence
2. **Sélectivité** : Répond préférentiellement à une orientation/fréquence
3. **Orthogonalité** : Filtres en quadrature de phase (φ et φ+π/2)

## Relation avec la Neurophysiologie

### Neurones Simples de V1 :
1. **Champs Récepteurs Orientés** : Détectent des bords/barres
2. **Organisation Colonnaire** : ~10° par pas d'orientation
3. **Codage de Phase** : Paires de neurones en quadrature
4. **Sélectivité Multi-échelle** : Différentes fréquences spatiales

### Architecture Fonctionnelle :
- **Hypercolonnes** : Unité fonctionnelle de base
- **Pinwheels** : Points singuliers de l'organisation
- **Rétinotopie** : Préservation de la topographie spatiale

## Applications

1. **Détection de Contours** : Extraction des bords orientés
2. **Analyse de Texture** : Caractérisation des motifs locaux
3. **Vision Pré-attentive** : Détection de singularités
4. **Codage Épars** : Représentation efficace des images naturelles

## Paramètres Recommandés

| Paramètre | Valeurs Typiques | Description |
|-----------|------------------|-------------|
| Orientations | 4-16 | Plus = résolution angulaire fine |
| Fréquences | 0.05-0.5 c/pixel | Basses = grandes structures |
| Phases | [0, π/2] | Cosinus et sinus |
| Sigma | 0.56/f | Relation standard |
| Taille | 6/f pixels | Couvre ~3 cycles |

## Extensions Possibles

1. **Filtres 3D** : Ajouter la dimension temporelle pour le mouvement
2. **Apprentissage** : Ajuster les filtres aux statistiques d'images
3. **SNN Integration** : Convertir les sorties en spikes
4. **Champ d'Association** : Connexions horizontales pour l'intégration
```

---

## **4. README Général du Projet**

**`NeuroGeomVision/README.md`**

```markdown
# NeuroGeomVision

**Une bibliothèque de vision computationnelle basée sur la neurogéométrie du cortex visuel et les réseaux de neurones à impulsions (SNN).**

## Concept

Cette bibliothèque implémente les modèles mathématiques et neurophysiologiques décrits dans l'ouvrage de Jean Petitot *"Neurogéométrie de la Vision"*, en les combinant avec le paradigme des réseaux de neurones à impulsions (Spiking Neural Networks).

L'objectif est de créer des algorithmes de vision de bas niveau qui sont :
1. **Neurophysiologiquement plausibles** : Basés sur l'architecture réelle du système visuel
2. **Mathématiquement élégants** : Fondés sur la géométrie différentielle et l'analyse harmonique
3. **Computationalement efficaces** : Adaptés aux architectures neuromorphiques

## Architecture du Projet

```
neurogeomvision/
├── core/                  # Utilitaires de base
├── retina_lgn/           # Filtres rétine/LGN et codage en spikes
│   ├── filters.py       # Filtres DoG, voies parvo/magno
│   └── coding.py        # Encoders de spikes, intégration temporelle
└── v1_simple_cells/     # Neurones simples du cortex V1
    └── gabor_filters.py # Banque de filtres de Gabor orientés
```

## Installation

```bash
# Clonez le dépôt
git clone <repository-url>
cd NeuroGeomVision

# Créez un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installez les dépendances
pip install -r requirements.txt
```

## Dépendances

- Python 3.8+
- PyTorch (CPU ou GPU)
- NumPy, SciPy
- Matplotlib (visualisation)
- OpenCV (traitement d'images, optionnel)

## Utilisation Rapide

```python
import torch
from neurogeomvision.retina_lgn.filters import ParvoMagnoPathway
from neurogeomvision.v1_simple_cells.gabor_filters import GaborFilterBank

# 1. Traitement rétine/LGN
image = torch.randn(128, 128)  # Image d'entrée
pathway = ParvoMagnoPathway(img_size=image.shape)
retina_results = pathway.process_frame(image)

# 2. Neurones simples de V1
gabor_bank = GaborFilterBank(
    img_size=image.shape,
    n_orientations=8,
    spatial_freqs=[0.1, 0.2]
)
v1_results = gabor_bank.apply_filters(image)

# 3. Visualisation
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(image.numpy(), cmap='gray')
axes[1].imshow(retina_results['parvo'].numpy(), cmap='RdBu_r')
axes[2].imshow(v1_results['dominant_orientation']['angle'].numpy(), cmap='hsv')
plt.show()
```

## Fondements Théoriques

### 1. Neurogéométrie de la Vision (Petitot)
- Le cortex V1 implémente une structure de contact sur l'espace des 1-jets
- L'organisation hypercolonnaire crée une fibration R² × P¹
- Les connexions horizontales implémentent une géométrie sous-riemannienne

### 2. Architecture Fonctionnelle Biologique
- **Voies Parvo/Magno** : Séparation dorsale/ventrale
- **Filtres de Gabor** : Optimal pour l'analyse espace-fréquence
- **Codage Événementiel** : Spikes pour l'efficacité énergétique

### 3. SNN (Spiking Neural Networks)
- Modèles bio-inspirés à dynamique temporelle
- Codage sparse et événementiel
- Compatible avec le hardware neuromorphique

## Exemples

Voir le dossier `examples/` pour :
- `test_retina_v1.py` : Pipeline complet rétine → V1
- `visualize_filters.py` : Visualisation des filtres
- `spike_encoding.py` : Démonstration des codages de spikes

## Développement Futur

### Modules Planifiés :
1. **contact_structure** : Espace des 1-jets et géométrie de contact
2. **sub_riemannian** : Intégration de contours par géodésiques
3. **association_field** : Connexions horizontales de V1
4. **snns** : Implémentations de neurones à impulsions
5. **learning** : Plasticité synaptique et apprentissage

### Fonctionnalités :
- [x] Filtres rétine/LGN (DoG)
- [x] Codage en spikes
- [x] Filtres de Gabor pour V1
- [ ] Intégration de contours
- [ ] Complétion modale (contours illusoires)
- [ ] Réseaux SNN complets
- [ ] Apprentissage non supervisé

## Références

1. Petitot, J. (2008). *Neurogéométrie de la vision*
2. Hubel, D. H., & Wiesel, T. N. (1962). *Receptive fields in cat striate cortex*
3. Daugman, J. G. (1985). *Uncertainty relation for resolution in space, spatial frequency, and orientation*
4. Thorpe, S., & Gautrais, J. (1998). *Rank order coding*
5. Petitot, J., & Tondut, Y. (1999). *Vers une neurogéométrie*

## Licence

À déterminer (MIT probablement)

## Contribution

Les contributions sont les bienvenues ! Voir `CONTRIBUTING.md` (à venir) pour les guidelines.

## Auteurs

[Votre nom/équipe]

---

*"Without a proper shape mathematics for biology, we are in the position that physics would have been in trying to develop mechanics without Euclidean geometry."*  
— Harry Blum (cité par Jean Petitot)
```

---

Ces README détaillés expliquent :

1. **La théorie** : Concepts neurophysiologiques et mathématiques
2. **L'implémentation** : Classes, méthodes, paramètres
3. **L'utilisation** : Exemples de code complets
4. **Les fondements** : Références et justifications théoriques
5. **Les applications** : Cas d'usage et extensions possibles

Ils constituent une documentation complète pour comprendre et utiliser chaque module. Veux-tu que je continue avec les autres parties du plan (contact structure, SNN, etc.) ?
