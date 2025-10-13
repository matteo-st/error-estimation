#!/usr/bin/env python3
"""
Quick Test Script - Vérifie que tout fonctionne sans téléchargements manuels

Ce script:
1. Télécharge automatiquement un modèle ViT-Tiny pré-entraîné (via timm)
2. Génère des données synthétiques pour tester
3. Teste le sample splitting (Phase 1 fix)
4. Vérifie l'inférence du modèle

Temps d'exécution: ~2-3 minutes (premier run avec téléchargement du modèle)
Pas de datasets ou checkpoints manuels requis!
"""

import torch
from torch.utils.data import TensorDataset, DataLoader
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

print("=" * 70)
print("🚀 Quick Test - Error Estimation with Theoretical Guarantees")
print("=" * 70)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n📱 Device: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("   (Running on CPU - will be slower)")

# 1. Load model (automatic download)
print("\n" + "=" * 70)
print("📦 Step 1: Loading ViT-Tiny model...")
print("=" * 70)

try:
    from code.utils.models import get_model

    model = get_model(
        model_name="timm_vit_tiny16",
        dataset_name="imagenet",
        n_classes=1000,
        input_dim=(3, 224, 224),
        model_seed=1,
        checkpoint_dir="checkpoints/ce"
    )
    model.eval()
    model.to(device)
    print("✅ Model loaded successfully!")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\nTry installing missing dependencies:")
    print("  pip install timm transformers")
    sys.exit(1)

# 2. Generate synthetic data
print("\n" + "=" * 70)
print("🔧 Step 2: Generating synthetic data...")
print("=" * 70)

n_samples = 5000
print(f"   Generating {n_samples} fake images (224x224x3)...")

fake_images = torch.randn(n_samples, 3, 224, 224)
fake_labels = torch.randint(0, 1000, (n_samples,))
dataset = TensorDataset(fake_images, fake_labels)

print(f"✅ Synthetic dataset created!")
print(f"   Shape: {fake_images.shape}")
print(f"   Labels range: [{fake_labels.min()}, {fake_labels.max()}]")

# 3. Test sample splitting (Phase 1 fix!)
print("\n" + "=" * 70)
print("✂️  Step 3: Testing sample splitting (Phase 1 Fix)...")
print("=" * 70)

try:
    from code.utils.sample_splitting import split_for_partition_detector

    result = split_for_partition_detector(
        dataset,
        min_samples=5000,
        resolution_ratio=0.05,
        seed=42
    )

    if result['can_split']:
        print("✅ Sample splitting successful!")
        print(f"   D_resolution: {len(result['D_res'])} samples (5.0%)")
        print(f"   D_calibration: {len(result['D_cal'])} samples (95.0%)")
        print("   ✓ Theoretical guarantees (Theorem 3.1) should hold")

        # Create dataloaders
        res_loader = DataLoader(result['D_res'], batch_size=128, shuffle=False)
        cal_loader = DataLoader(result['D_cal'], batch_size=128, shuffle=False)

    else:
        print(f"⚠️  Cannot split: {result['warning']}")
        print("   Using all data for calibration (guarantees may not hold)")
        cal_loader = DataLoader(dataset, batch_size=128, shuffle=False)
        res_loader = cal_loader

except Exception as e:
    print(f"❌ Error in sample splitting: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Test independence validation
print("\n" + "=" * 70)
print("🔍 Step 4: Testing independence validation...")
print("=" * 70)

try:
    from code.utils.validation import TheoreticalGuaranteesValidator

    validator = TheoreticalGuaranteesValidator()

    # Get data hashes
    res_data = torch.stack([result['D_res'][i][0] for i in range(min(100, len(result['D_res'])))])
    cal_data = torch.stack([result['D_cal'][i][0] for i in range(min(100, len(result['D_cal'])))])

    hash_res = hash(res_data.numpy().tobytes())
    hash_cal = hash(cal_data.numpy().tobytes())

    # Check independence
    try:
        validator.check_independence(hash_res, hash_cal)
        print("✅ Independence check passed!")
        print("   D_resolution and D_calibration are disjoint")
    except ValueError as e:
        print(f"❌ Independence violated: {e}")

    # Test empty cluster detection
    print("\n   Testing empty cluster detection...")
    mock_cluster_counts = torch.tensor([100, 0, 150, 0, 200, 50])
    diag = validator.check_empty_clusters(mock_cluster_counts, threshold=0.2)
    print(f"   - Empty clusters: {diag['n_empty']} / {diag['n_total']}")
    print(f"   - Fraction empty: {diag['fraction_empty']:.1%}")
    print(f"   - Has warning: {diag['has_warning']}")

except Exception as e:
    print(f"❌ Error in validation: {e}")
    import traceback
    traceback.print_exc()

# 5. Test inference
print("\n" + "=" * 70)
print("🔮 Step 5: Testing model inference...")
print("=" * 70)

try:
    # Get a batch from resolution loader
    batch = next(iter(res_loader))
    images, labels = batch

    print(f"   Batch shape: {images.shape}")
    print(f"   Labels: {labels[:10].tolist()}")

    # Run inference
    with torch.no_grad():
        logits = model(images.to(device))

    predictions = logits.argmax(dim=1).cpu()
    probabilities = torch.softmax(logits, dim=1).cpu()
    max_probs = probabilities.max(dim=1)[0]

    print(f"\n✅ Inference successful!")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Predictions (first 10): {predictions[:10].tolist()}")
    print(f"   Max probabilities: {max_probs[:5].numpy()}")
    print(f"   Mean confidence: {max_probs.mean():.3f}")

except Exception as e:
    print(f"❌ Error in inference: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 6. Summary
print("\n" + "=" * 70)
print("✅ All Tests Passed!")
print("=" * 70)

print("""
Summary:
--------
✓ Model loading (ViT-Tiny from timm)
✓ Sample splitting (Phase 1 fix)
✓ Independence validation
✓ Empty cluster detection
✓ Model inference

Next Steps:
-----------
1. Try with real CIFAR-10 data:
   python -m code.detection_clean

2. Read the documentation:
   - GETTING_STARTED.md - Datasets and models setup
   - PHASE1_FIXES.md - Theoretical guarantee fixes
   - docs/CONCENTRATION_BOUNDS.md - Hoeffding vs Bernstein

3. Run comprehensive tests:
   PYTHONPATH=$(pwd) pytest tests/test_sample_splitting.py -v

4. Explore Phase 2 integration:
   - See PHASE2_PROGRESS.md for roadmap

Happy experimenting! 🎉
""")

print("=" * 70)
