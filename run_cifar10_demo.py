#!/usr/bin/env python3
"""
Demo: Misclassification Detection sur CIFAR-10

Ce script montre comment utiliser le code avec CIFAR-10 (déjà téléchargé!)
et un modèle ViT-Tiny pour détecter les misclassifications.
"""

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from code.utils.models import get_model_essentials
from code.utils.sample_splitting import split_for_partition_detector

print("=" * 70)
print("🎯 Misclassification Detection Demo - CIFAR-10")
print("=" * 70)

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n📱 Device: {device}")

# Step 1: Load CIFAR-10
print("\n" + "=" * 70)
print("📦 Step 1/5: Loading CIFAR-10 dataset...")
print("=" * 70)

try:
    dataset = CIFAR10(
        root="./data",
        train=True,
        download=False,  # Already downloaded!
        transform=None
    )
    print(f"✅ CIFAR-10 loaded successfully!")
    print(f"   Train samples: {len(dataset)}")
    print(f"   Classes: {dataset.classes}")
except Exception as e:
    print(f"❌ Error loading CIFAR-10: {e}")
    print("\nRun setup_complete.py first to download CIFAR-10")
    sys.exit(1)

# Step 2: Sample Splitting (Phase 1 Fix!)
print("\n" + "=" * 70)
print("✂️  Step 2/5: Sample splitting (Phase 1 - Theorem 3.1)...")
print("=" * 70)

result = split_for_partition_detector(
    dataset,
    min_samples=5000,
    resolution_ratio=0.05,
    seed=42
)

if not result['can_split']:
    print(f"⚠️  {result['warning']}")
    print("   Continuing anyway for demo purposes...")
    # Use all data
    from torch.utils.data import Subset
    n = len(dataset)
    n_res = int(n * 0.05)
    result['D_res'] = Subset(dataset, range(n_res))
    result['D_cal'] = Subset(dataset, range(n_res, n))

print(f"✅ Split successful!")
print(f"   D_resolution: {len(result['D_res'])} samples (for GMM clustering)")
print(f"   D_calibration: {len(result['D_cal'])} samples (for intervals)")

# Step 3: Load ViT Model
print("\n" + "=" * 70)
print("🤖 Step 3/5: Loading ViT-Tiny model...")
print("=" * 70)

try:
    essentials = get_model_essentials("timm_vit_tiny16", "imagenet")
    model = essentials["model"]
    transform = essentials["test_transforms"]

    model.eval()
    model.to(device)

    print(f"✅ Model loaded successfully!")
    print(f"   Architecture: ViT-Tiny/16")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Input size: 224x224")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\nMake sure timm is installed:")
    print("  pip install timm")
    sys.exit(1)

# Step 4: Apply transforms and create dataloaders
print("\n" + "=" * 70)
print("🔧 Step 4/5: Preparing dataloaders...")
print("=" * 70)

# Apply transforms to datasets
result['D_res'].dataset.transform = transform
result['D_cal'].dataset.transform = transform

# Create dataloaders
batch_size = 64  # Smaller batch for demo
res_loader = DataLoader(result['D_res'], batch_size=batch_size, shuffle=False)
cal_loader = DataLoader(result['D_cal'], batch_size=batch_size, shuffle=False)

print(f"✅ Dataloaders created!")
print(f"   Resolution loader: {len(res_loader)} batches")
print(f"   Calibration loader: {len(cal_loader)} batches")
print(f"   Batch size: {batch_size}")

# Step 5: Compute logits and predictions
print("\n" + "=" * 70)
print("🔮 Step 5/5: Computing model predictions...")
print("=" * 70)

print("\n   Processing calibration data...")
all_logits = []
all_labels = []
all_correct = []

with torch.no_grad():
    for i, (images, labels) in enumerate(cal_loader):
        if i % 10 == 0:
            print(f"   Batch {i}/{len(cal_loader)}...", end='\r')

        images = images.to(device)
        logits = model(images)
        predictions = logits.argmax(dim=1).cpu()

        # Note: ViT is trained on ImageNet (1000 classes)
        # CIFAR-10 has only 10 classes, so predictions won't match
        # This is just for demonstration purposes

        all_logits.append(logits.cpu())
        all_labels.append(labels)
        all_correct.append((predictions == labels).float())

print()  # New line after progress

all_logits = torch.cat(all_logits, dim=0)
all_labels = torch.cat(all_labels, dim=0)
all_correct = torch.cat(all_correct, dim=0)

print(f"\n✅ Predictions computed!")
print(f"   Total samples: {len(all_labels)}")
print(f"   Logits shape: {all_logits.shape}")

# Step 6: Analyze predictions
print("\n" + "=" * 70)
print("📊 Analysis:")
print("=" * 70)

# Compute confidence
probs = torch.softmax(all_logits, dim=1)
max_probs, pred_classes = probs.max(dim=1)

print(f"\nPrediction Statistics:")
print(f"   Mean confidence: {max_probs.mean():.3f}")
print(f"   Median confidence: {max_probs.median():.3f}")
print(f"   Min confidence: {max_probs.min():.3f}")
print(f"   Max confidence: {max_probs.max():.3f}")

# Note about domain shift
print(f"\n⚠️  Note: Domain Shift!")
print(f"   ViT is trained on ImageNet (1000 classes, natural images)")
print(f"   CIFAR-10 has 10 classes (tiny 32x32 images)")
print(f"   Accuracy will be low due to domain mismatch!")
print(f"   For real use, train a model on CIFAR-10 first.")

# What you can do next
print("\n" + "=" * 70)
print("🎯 Next Steps:")
print("=" * 70)

print("""
1. Use MegaPartitionDetector to create confidence intervals:

   from code.utils.detection.methods import MegaPartitionDetector

   detector = MegaPartitionDetector(
       model=model,
       list_n_cluster=[100],
       alpha=0.05,
       name="soft-kmeans_torch",
       space="probits",
       bound="bernstein"
   )

   # Fit on calibration data
   detector.fit(all_logits, all_correct, cal_loader, fit_clustering=True)

   # Predict on test data
   test_loader = DataLoader(test_dataset, batch_size=64)
   upper_bounds = []
   for images, _ in test_loader:
       images = images.to(device)
       logits = model(images)
       bounds = detector(logits=logits)
       upper_bounds.append(bounds)

2. Validate theoretical guarantees:

   from code.utils.validation import TheoreticalGuaranteesValidator

   validator = TheoreticalGuaranteesValidator()
   coverage = validator.compute_empirical_coverage(
       detector, test_loader, alpha=0.05
   )
   print(f"Coverage: {coverage['empirical_coverage']:.3f}")

3. Train your own model on CIFAR-10 for better results

4. Read the documentation:
   - PHASE1_FIXES.md for theoretical details
   - docs/CONCENTRATION_BOUNDS.md for Hoeffding vs Bernstein
""")

print("=" * 70)
print("✅ Demo completed successfully!")
print("=" * 70)
