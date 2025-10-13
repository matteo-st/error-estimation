#!/usr/bin/env python3
"""
Complete Setup Script - Télécharge datasets et modèles

Ce script télécharge:
1. CIFAR-10 dataset (auto via torchvision)
2. CIFAR-100 dataset (auto via torchvision)
3. ResNet34 checkpoints pour CIFAR-10 et CIFAR-100
4. Vérifie que tout fonctionne

Temps estimé: 5-10 minutes
Espace disque: ~500 MB
"""

import os
import sys
import torch
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100
from pathlib import Path
import urllib.request
import hashlib

print("=" * 70)
print("🚀 Complete Setup - Downloading Everything")
print("=" * 70)

# Configuration
DATA_DIR = "./data"
CHECKPOINTS_DIR = "./checkpoints/ce"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# Checkpoint URLs (from relative-uncertainty releases)
CHECKPOINT_URLS = {
    "resnet34_cifar10": "https://github.com/edadaltocg/relative-uncertainty/releases/download/checkpoints/resnet34_cifar10_1.pt",
    "resnet34_cifar100": "https://github.com/edadaltocg/relative-uncertainty/releases/download/checkpoints/resnet34_cifar100_1.pt",
    "densenet121_cifar10": "https://github.com/edadaltocg/relative-uncertainty/releases/download/checkpoints/densenet121_cifar10_1.pt",
    "densenet121_cifar100": "https://github.com/edadaltocg/relative-uncertainty/releases/download/checkpoints/densenet121_cifar100_1.pt",
}

def download_with_progress(url, destination):
    """Download file with progress bar"""
    print(f"   Downloading from {url}")

    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r   Progress: {percent}% ")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, destination, reporthook)
        print()  # New line after progress
        return True
    except Exception as e:
        print(f"\n   ❌ Error: {e}")
        return False

# Step 1: Download CIFAR-10
print("\n" + "=" * 70)
print("📦 Step 1/5: Downloading CIFAR-10 dataset...")
print("=" * 70)
print("   Size: ~170 MB")

try:
    train_dataset = CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=None
    )
    test_dataset = CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=None
    )
    print(f"✅ CIFAR-10 downloaded successfully!")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Location: {DATA_DIR}/cifar-10-batches-py/")
except Exception as e:
    print(f"❌ Error downloading CIFAR-10: {e}")
    sys.exit(1)

# Step 2: Download CIFAR-100
print("\n" + "=" * 70)
print("📦 Step 2/5: Downloading CIFAR-100 dataset...")
print("=" * 70)
print("   Size: ~170 MB")

try:
    train_dataset = CIFAR100(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=None
    )
    test_dataset = CIFAR100(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=None
    )
    print(f"✅ CIFAR-100 downloaded successfully!")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Location: {DATA_DIR}/cifar-100-python/")
except Exception as e:
    print(f"❌ Error downloading CIFAR-100: {e}")
    sys.exit(1)

# Step 3: Download ResNet34 CIFAR-10 checkpoint
print("\n" + "=" * 70)
print("📦 Step 3/5: Downloading ResNet34-CIFAR10 checkpoint...")
print("=" * 70)

checkpoint_dir = os.path.join(CHECKPOINTS_DIR, "resnet34_cifar10", "1")
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "best.pth")

if os.path.exists(checkpoint_path):
    print(f"✅ Checkpoint already exists: {checkpoint_path}")
else:
    print(f"   Downloading to: {checkpoint_path}")
    url = CHECKPOINT_URLS["resnet34_cifar10"]

    if download_with_progress(url, checkpoint_path):
        print(f"✅ ResNet34-CIFAR10 checkpoint downloaded!")
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"   Size: {file_size:.1f} MB")
    else:
        print(f"⚠️  Failed to download checkpoint (network error?)")
        print(f"   You can download manually from:")
        print(f"   {url}")

# Step 4: Download ResNet34 CIFAR-100 checkpoint
print("\n" + "=" * 70)
print("📦 Step 4/5: Downloading ResNet34-CIFAR100 checkpoint...")
print("=" * 70)

checkpoint_dir = os.path.join(CHECKPOINTS_DIR, "resnet34_cifar100", "1")
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "best.pth")

if os.path.exists(checkpoint_path):
    print(f"✅ Checkpoint already exists: {checkpoint_path}")
else:
    print(f"   Downloading to: {checkpoint_path}")
    url = CHECKPOINT_URLS["resnet34_cifar100"]

    if download_with_progress(url, checkpoint_path):
        print(f"✅ ResNet34-CIFAR100 checkpoint downloaded!")
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"   Size: {file_size:.1f} MB")
    else:
        print(f"⚠️  Failed to download checkpoint (network error?)")
        print(f"   You can download manually from:")
        print(f"   {url}")

# Step 5: Verify everything works
print("\n" + "=" * 70)
print("🔍 Step 5/5: Verifying setup...")
print("=" * 70)

errors = []

# Check datasets
if not os.path.exists(os.path.join(DATA_DIR, "cifar-10-batches-py")):
    errors.append("CIFAR-10 dataset not found")
if not os.path.exists(os.path.join(DATA_DIR, "cifar-100-python")):
    errors.append("CIFAR-100 dataset not found")

# Check checkpoints
resnet10_path = os.path.join(CHECKPOINTS_DIR, "resnet34_cifar10", "1", "best.pth")
resnet100_path = os.path.join(CHECKPOINTS_DIR, "resnet34_cifar100", "1", "best.pth")

if not os.path.exists(resnet10_path):
    errors.append("ResNet34-CIFAR10 checkpoint not found")
if not os.path.exists(resnet100_path):
    errors.append("ResNet34-CIFAR100 checkpoint not found")

# Test model loading
print("\n   Testing model loading...")
try:
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from code.utils.models import get_model

    model = get_model(
        model_name="resnet34",
        dataset_name="cifar10",
        n_classes=10,
        input_dim=(3, 32, 32),
        model_seed=1,
        checkpoint_dir=CHECKPOINTS_DIR
    )
    print("   ✅ Model loads successfully!")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    errors.append(f"Model loading failed: {e}")
    print(f"   ❌ Model loading failed: {e}")

# Final summary
print("\n" + "=" * 70)
if errors:
    print("⚠️  Setup completed with warnings:")
    print("=" * 70)
    for error in errors:
        print(f"   - {error}")
    print("\nSome components may not work. Check errors above.")
else:
    print("✅ Setup completed successfully!")
    print("=" * 70)

print("""
What was downloaded:
--------------------
✓ CIFAR-10 dataset (~170 MB)
  - 50,000 training images (32x32x3)
  - 10,000 test images
  - 10 classes

✓ CIFAR-100 dataset (~170 MB)
  - 50,000 training images (32x32x3)
  - 10,000 test images
  - 100 classes

✓ ResNet34 checkpoints (~85 MB total)
  - Pre-trained on CIFAR-10
  - Pre-trained on CIFAR-100

Total disk space used: ~425 MB

Next steps:
-----------
1. Run quick test:
   python quick_test.py

2. Run detection on CIFAR-10:
   python -m code.detection_clean

3. Run with custom config:
   Edit base_config in detection_clean.py line 267

4. Run tests:
   PYTHONPATH=$(pwd) pytest tests/test_sample_splitting.py -v

Happy experimenting! 🎉
""")

print("=" * 70)
