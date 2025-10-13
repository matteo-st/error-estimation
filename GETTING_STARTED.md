# Getting Started Guide - Datasets & Models

Ce guide t'explique comment obtenir les datasets et modèles nécessaires pour faire tourner le code.

---

## 📊 Datasets Supportés

Le code supporte plusieurs datasets standards:

### 1. CIFAR-10 / CIFAR-100 (Téléchargement Automatique)

**Le plus simple pour commencer!**

```python
# CIFAR-10: 60k images 32x32, 10 classes
# CIFAR-100: 60k images 32x32, 100 classes
# Téléchargement automatique via torchvision
```

**Pas besoin de télécharger manuellement** - le code le fait automatiquement:

```bash
# Exemple: Lance le code avec CIFAR-10
cd /Users/ulyssetrin/Desktop/matteo/error-estimation
source venv/bin/activate

python -m code.detection_clean \
    --dataset cifar10 \
    --model resnet34 \
    --n_classes 10
```

Les données seront téléchargées dans `./data/cifar-10-batches-py/` automatiquement.

---

### 2. ImageNet (Téléchargement Manuel Requis)

**Plus complexe, gros dataset**

ImageNet nécessite un téléchargement manuel depuis le site officiel.

#### Option A: ImageNet via Kaggle (Plus facile)

1. **Crée un compte Kaggle**: https://www.kaggle.com/
2. **Va sur**: https://www.kaggle.com/c/imagenet-object-localization-challenge
3. **Télécharge**: `ILSVRC.tar.gz` (~150 GB)
4. **Extraction**:
   ```bash
   mkdir -p ./data/imagenet
   tar -xzf ILSVRC.tar.gz -C ./data/imagenet/
   ```

5. **Structure attendue**:
   ```
   ./data/imagenet/ILSVRC/
   ├── imagenet_class_index.json
   ├── ILSVRC2012_val_labels.json
   └── Data/CLS-LOC/
       ├── train/
       │   ├── n01440764/  # classe 1
       │   │   ├── n01440764_0.JPEG
       │   │   └── ...
       │   └── ...
       └── val/
           ├── ILSVRC2012_val_00000001.JPEG
           └── ...
   ```

#### Option B: ImageNet via timm (ViT pré-entraînés)

**Beaucoup plus simple - pas besoin du dataset complet!**

Les modèles ViT de `timm` sont pré-entraînés, donc tu peux tester l'**algorithme** sans le dataset:

```python
# Génère des données synthétiques pour tester
import torch
from code.utils.models import get_model

model = get_model(
    model_name="timm_vit_tiny16",
    dataset_name="imagenet",
    n_classes=1000,
    input_dim=(3, 224, 224),
    model_seed=1,
    checkpoint_dir="checkpoints/ce"
)

# Teste avec des données aléatoires
dummy_images = torch.randn(10, 3, 224, 224)
logits = model(dummy_images)
print(logits.shape)  # torch.Size([10, 1000])
```

---

### 3. Gaussian Mixture (Synthétique)

**Idéal pour tester rapidement!**

Données synthétiques générées à partir de mélanges de gaussiennes.

#### Génération Rapide

```python
from code.utils.datasets import get_synthetic_dataset

# Génère un dataset synthétique
dataset = get_synthetic_dataset(
    data_name="gaussian_mixture",
    n_samples=10000,
    dim=3072,  # dimension (3x32x32 pour image-like)
    n_classes=10,
    seed=42
)
```

**Problème**: Nécessite des données pré-générées dans `./data/gaussian_mixture/`.

#### Génère les données:

```python
import os
import torch
import numpy as np
from torch.distributions import MultivariateNormal

# Configuration
dim = 3072
n_classes = 10
n_samples = 10000
seed = 42

# Crée le dossier
data_dir = f"./data/gaussian_mixture/dim-{dim}_classes-{n_classes}-seed-{seed}"
os.makedirs(data_dir, exist_ok=True)

# Génère les paramètres
torch.manual_seed(seed)
means = torch.randn(n_classes, dim) * 5  # Moyennes séparées
covs = torch.stack([torch.eye(dim) for _ in range(n_classes)])  # Covariances identité

# Génère les samples
weights = torch.ones(n_classes) / n_classes
gen = torch.Generator().manual_seed(seed)
components = torch.multinomial(weights, n_samples, replacement=True, generator=gen)

samples = []
labels = []
for i in range(n_classes):
    mask = (components == i)
    n_i = mask.sum().item()
    if n_i > 0:
        samples_i = torch.randn(n_i, dim) + means[i]
        samples.append(samples_i)
        labels.append(torch.full((n_i,), i, dtype=torch.long))

all_samples = torch.cat(samples, dim=0)
all_labels = torch.cat(labels, dim=0)

# Permute pour mélanger
perm = torch.randperm(all_samples.size(0), generator=gen)
all_samples = all_samples[perm]
all_labels = all_labels[perm]

# Sauvegarde
torch.save(all_samples, os.path.join(data_dir, "samples.pt"))
torch.save(all_labels, os.path.join(data_dir, "labels.pt"))

print(f"✓ Dataset synthétique créé: {data_dir}")
print(f"  - Samples: {all_samples.shape}")
print(f"  - Labels: {all_labels.shape}")
```

---

## 🤖 Modèles Supportés

### 1. Modèles CIFAR (Nécessitent Checkpoints)

**Modèles disponibles**:
- `resnet34_cifar10` - ResNet-34 entraîné sur CIFAR-10
- `resnet34_cifar100` - ResNet-34 entraîné sur CIFAR-100
- `densenet121_cifar10` - DenseNet-121 entraîné sur CIFAR-10
- `densenet121_cifar100` - DenseNet-121 entraîné sur CIFAR-100

**Où télécharger** (mentionné dans README original):
```bash
wget https://github.com/edadaltocg/relative-uncertainty/releases/download/checkpoints/resnet34_cifar10.pth
```

**Structure attendue**:
```
checkpoints/ce/
├── resnet34_cifar10/
│   └── 1/
│       └── best.pth
├── resnet34_cifar100/
│   └── 1/
│       └── best.pth
└── ...
```

---

### 2. Modèles ImageNet (Via timm - Téléchargement Automatique)

**Le plus simple! Pas besoin de checkpoints manuels.**

Les modèles suivants sont téléchargés automatiquement via `timm` ou `transformers`:

- `timm_vit_tiny16_imagenet` - ViT-Tiny/16 pré-entraîné
- `timm_vit_base16_imagenet` - ViT-Base/16 pré-entraîné
- `vit_base16_imagenet` - ViT-Base/16 (transformers)
- `vit_large16_imagenet` - ViT-Large/16 (transformers)
- `vit_huge14_imagenet` - ViT-Huge/14 (transformers)

**Exemple d'utilisation**:

```python
from code.utils.models import get_model

# Téléchargement automatique!
model = get_model(
    model_name="timm_vit_tiny16",
    dataset_name="imagenet",
    n_classes=1000,
    input_dim=(3, 224, 224),
    model_seed=1,
    checkpoint_dir="checkpoints/ce"
)
```

---

## 🚀 Scénarios de Démarrage

### Scénario 1: Test Rapide avec ViT (Recommandé)

**Aucun téléchargement manuel requis!**

```bash
cd /Users/ulyssetrin/Desktop/matteo/error-estimation
source venv/bin/activate

# Installe timm et transformers si nécessaire
pip install timm transformers

# Crée un script de test rapide
cat > quick_test.py << 'EOF'
import torch
from code.utils.models import get_model
from code.utils.sample_splitting import split_for_partition_detector
from torch.utils.data import TensorDataset, DataLoader

# 1. Charge le modèle (téléchargement automatique)
print("📦 Chargement du modèle ViT-Tiny...")
model = get_model(
    model_name="timm_vit_tiny16",
    dataset_name="imagenet",
    n_classes=1000,
    input_dim=(3, 224, 224),
    model_seed=1,
    checkpoint_dir="checkpoints/ce"
)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 2. Génère des données synthétiques pour tester
print("🔧 Génération de données synthétiques...")
n_samples = 5000
fake_images = torch.randn(n_samples, 3, 224, 224)
fake_labels = torch.randint(0, 1000, (n_samples,))
dataset = TensorDataset(fake_images, fake_labels)

# 3. Test sample splitting (Phase 1 fix!)
print("✂️  Test du sample splitting...")
result = split_for_partition_detector(
    dataset,
    min_samples=5000,
    resolution_ratio=0.05
)

if result['can_split']:
    print(f"✅ Split réussi!")
    print(f"   - D_resolution: {len(result['D_res'])} samples")
    print(f"   - D_calibration: {len(result['D_cal'])} samples")

    # 4. Crée les dataloaders
    res_loader = DataLoader(result['D_res'], batch_size=128)
    cal_loader = DataLoader(result['D_cal'], batch_size=128)

    # 5. Test d'inférence
    print("🔮 Test d'inférence...")
    batch = next(iter(res_loader))
    images, labels = batch
    with torch.no_grad():
        logits = model(images.to(device))
    print(f"✅ Logits shape: {logits.shape}")
    print(f"✅ Predictions: {logits.argmax(dim=1)[:10]}")

print("\n🎉 Tout fonctionne!")
EOF

# Lance le test
python quick_test.py
```

---

### Scénario 2: CIFAR-10 avec ResNet (Nécessite Checkpoint)

```bash
# 1. Télécharge le checkpoint
mkdir -p checkpoints/ce/resnet34_cifar10/1
wget -O checkpoints/ce/resnet34_cifar10/1/best.pth \
    https://github.com/edadaltocg/relative-uncertainty/releases/download/checkpoints/resnet34_cifar10.pth

# 2. Lance le code
python -m code.detection_clean
```

Le script `detection_clean.py` est configuré par défaut pour ImageNet + ViT, mais tu peux modifier le `base_config` à la ligne 267:

```python
base_config = {
    "data": {
        "name": "cifar10",  # Change ici
        "n_classes": 10,
        ...
    },
    "model": {
        "name": "resnet34",  # Change ici
        ...
    },
    ...
}
```

---

### Scénario 3: Gaussian Mixture (Synthétique)

```bash
# 1. Génère le dataset
python << 'EOF'
import os
import torch

dim = 3072
n_classes = 10
n_samples = 10000
seed = 42

data_dir = f"./data/gaussian_mixture/dim-{dim}_classes-{n_classes}-seed-{seed}"
os.makedirs(data_dir, exist_ok=True)

torch.manual_seed(seed)
means = torch.randn(n_classes, dim) * 5
weights = torch.ones(n_classes) / n_classes
gen = torch.Generator().manual_seed(seed)
components = torch.multinomial(weights, n_samples, replacement=True, generator=gen)

samples = []
labels = []
for i in range(n_classes):
    mask = (components == i)
    n_i = mask.sum().item()
    if n_i > 0:
        samples_i = torch.randn(n_i, dim) + means[i]
        samples.append(samples_i)
        labels.append(torch.full((n_i,), i, dtype=torch.long))

all_samples = torch.cat(samples, dim=0)
all_labels = torch.cat(labels, dim=0)
perm = torch.randperm(all_samples.size(0), generator=gen)

torch.save(all_samples[perm], os.path.join(data_dir, "samples.pt"))
torch.save(all_labels[perm], os.path.join(data_dir, "labels.pt"))

print(f"✓ Dataset créé: {data_dir}")
EOF

# 2. Modifie base_config dans detection_clean.py ligne 267:
#    "data": {"name": "gaussian_mixture", ...}
```

---

## 📁 Structure Recommandée

```
error-estimation/
├── data/
│   ├── cifar-10-batches-py/       # Auto-téléchargé
│   ├── cifar-100-python/          # Auto-téléchargé
│   ├── imagenet/                  # Manuel (optionnel)
│   │   └── ILSVRC/...
│   └── gaussian_mixture/          # Généré par script
│       └── dim-3072_classes-10-seed-42/
│           ├── samples.pt
│           └── labels.pt
├── checkpoints/
│   └── ce/
│       ├── resnet34_cifar10/      # Manuel
│       │   └── 1/best.pth
│       └── ...
├── venv/                          # Environnement virtuel
└── ...
```

---

## ✅ Checklist de Démarrage

### Pour un test rapide (5 minutes):
- [ ] Environnement virtuel activé (`source venv/bin/activate`)
- [ ] `timm` et `transformers` installés
- [ ] Lance `quick_test.py` (Scénario 1)
- [ ] Tout devrait fonctionner sans téléchargement manuel!

### Pour reproduire les expériences du papier:
- [ ] Télécharge ImageNet (~150 GB)
- [ ] Télécharge les checkpoints ResNet/DenseNet
- [ ] Configure `base_config` dans `detection_clean.py`
- [ ] Lance les expériences

---

## 🆘 Troubleshooting

### Erreur: "Checkpoint file not found"
→ Tu as besoin du checkpoint pré-entraîné. Soit:
1. Télécharge-le manuellement (voir liens ci-dessus)
2. Ou utilise un modèle `timm_vit_*` qui se télécharge auto

### Erreur: "Pre-generated samples not found"
→ Pour Gaussian Mixture, génère le dataset avec le script du Scénario 3

### Erreur: "No module named 'timm'" ou "transformers"
```bash
pip install timm transformers
```

### Le code est lent
→ Utilise un GPU si disponible:
```bash
# Vérifie CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 🎯 Recommandation Finale

**Pour commencer immédiatement** sans rien télécharger:

1. ✅ Utilise `quick_test.py` (Scénario 1)
2. ✅ Modèle ViT-Tiny téléchargé automatiquement
3. ✅ Données synthétiques générées à la volée
4. ✅ Test du sample splitting (Phase 1)
5. ✅ Tout fonctionne en < 5 minutes!

**Pour les expériences complètes**:
- ImageNet + checkpoints (plusieurs heures de setup)
- CIFAR + checkpoints (30 minutes de setup)

---

**Auteur**: Claude Code
**Date**: 2025-10-13
**Branch**: `fix/theoretical-guarantees-p0`
