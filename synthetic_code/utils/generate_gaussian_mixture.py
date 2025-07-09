import os
import json
import time

import torch
import numpy as np
from torch.utils.data import Dataset

# ──────────────────────────────────────────────────────────────────────────────
# 1) Generation script
# ──────────────────────────────────────────────────────────────────────────────
def main(cfg):
    torch.manual_seed(cfg["seed"])          # seeds CPU _and_ CUDA default
    torch.cuda.manual_seed_all(cfg["seed"]) # redundant but explicit

    out_dir = os.path.join(cfg["out_dir"], 
                           f'dim-{cfg["dim"]}_classes-{cfg["n_classes"]}-seed-{cfg["seed"]}')
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    # --- Load true mixture parameters from .npz (CPU) ---
    params = np.load(cfg["params_path"])
    means_np   = params["means"]    # [n_classes, dim]
    covs_np    = params["covs"]     # [n_classes, dim, dim]
    weights_np = params["weights"]  # [n_classes]
    t1 = time.time()
    print(f"[1] Loaded params in {t1 - t0:.2f}s")

    # --- Move to GPU & factorize covariances ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    means   = torch.from_numpy(means_np).float().to(device)      # [n_classes, dim]
    covs    = torch.from_numpy(covs_np).float().to(device)       # [n_classes, dim, dim]
    weights = torch.from_numpy(weights_np).float().to(device)    # [n_classes]

    t2 = time.time()

    assert cfg["dim"] == means.shape[1], \
        f"Expected means.shape[1] == {cfg['dim']}, got {means.shape[1]}"
    assert cfg["n_classes"] == means.shape[0], \
        f"Expected means.shape[0] == {cfg['n_classes']}, got {means.shape[0]}"

    # Cholesky on GPU: one shot per component
    L_chols = torch.linalg.cholesky(covs)                         # [n_classes, dim, dim]
    t3 = time.time()
    print(f"[2] Moved to GPU in {t2 - t1:.2f}s, Cholesky in {t3 - t2:.2f}s")

    # --- Pick component assignment once on CPU for full reproducibility ---
    # gen = torch.Generator(device="cpu").manual_seed(cfg["seed"])
    n_samples = cfg["n_samples"]
    # Draw n_samples indices in [0, n_classes)
    comps = torch.multinomial(weights.cpu(), n_samples, replacement=True)  # [n_samples]
    t4 = time.time()
    print(f"[3] Sampled {n_samples} component indices in {t4 - t3:.2f}s")

    # --- Allocate storage on CPU ---
    dim = means.shape[1]
    samples = torch.empty(n_samples, dim, dtype=torch.float32)
    labels  = comps.clone()  # [n_samples]
    t5 = time.time()

    # --- Generate per-class batches on GPU, copy back to CPU ---
    for i in range(means.shape[0]):
        idx = (comps == i).nonzero(as_tuple=True)[0]  # positions for class i
        if idx.numel() == 0:
            continue

        # sample std‐normal noise for this class
        z = torch.randn(idx.numel(), dim, device=device)
        # compute samples_i = mean_i + L_chols[i] @ z.T, then transpose
        block = (L_chols[i] @ z.transpose(0, 1)).transpose(0, 1) + means[i]
        # move to CPU and write into the big samples tensor
        samples[idx] = block.cpu()

    t6 = time.time()
    print(f"[4] Generated all samples in {t6 - t5:.2f}s")
    print(n_samples, "samples generated in", samples.shape, "with means", means.shape,)
    # --- Save outputs ---


    samples_path = os.path.join(out_dir, "samples.pt")
    labels_path  = os.path.join(out_dir, "labels.pt")
    torch.save(samples, samples_path)
    torch.save(labels,  labels_path)
    t7 = time.time()

    # --- Save a simple config.json for traceability ---
    config_out = {
        "model_name": cfg["model_name"],
        "n_samples":  cfg["n_samples"],
        "seed":       cfg["seed"],
        "dim":        dim,
        "n_classes":  means.shape[0],
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_out, f, indent=2)
    t8 = time.time()

    print(f"[5] Saved data in {t7 - t6:.2f}s, wrote config in {t8 - t7:.2f}s")
    print(f"[Total] {t8 - t0:.2f}s — outputs in {cfg['out_dir']}")

       # --- Sanity checks ---
    print("\n[Sanity checks]")

    n_classes = means.shape[0]
    # Move everything to CPU for checking
    samples_cpu = samples
    labels_cpu  = labels
    means_cpu   = torch.from_numpy(means_np).float()   # true means
    covs_cpu    = torch.from_numpy(covs_np).float()    # true covariances
    weights_cpu = torch.from_numpy(weights_np).float() # true weights

    # 1) Class counts vs. expected
    counts    = torch.bincount(labels_cpu, minlength=n_classes).float()
    expected  = weights_cpu * cfg["n_samples"]
    rel_err   = (counts - expected) / expected.clamp(min=1)
    print(" class | count  | expected | rel_err")
    for i in range(n_classes):
        print(f"  {i:>2d}   | {counts[i]:>6.0f} | {expected[i]:>8.1f} | {rel_err[i]:> .2%}")

    # 2) Mean error per class
    print("\n mean-error per class (||sample_mean – true_mean||₂):")
    for i in range(n_classes):
        idx = (labels_cpu == i).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            print(f"  class {i}: NO SAMPLES!")
            continue
        sample_mean = samples_cpu[idx].mean(dim=0)
        err = torch.norm(sample_mean - means_cpu[i]).item()
        print(f"  class {i}: {err:.4f}")

    # 3) Variance error per class
    print("\n variance-error per class (mean(emp_var) vs mean(true_var)):")
    for i in range(n_classes):
        idx = (labels_cpu == i).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            print(f"  class {i}: NO SAMPLES!")
            continue
        # Empirical variance along each dimension, then average
        emp_var = samples_cpu[idx].var(dim=0, unbiased=False).mean().item()
        # True variance is the diagonal of the true covariance matrix
        true_var = covs_cpu[i].diagonal().mean().item()
        rel_err_var = (emp_var - true_var) / true_var if true_var != 0 else float('nan')
        print(f"  class {i}: emp_var={emp_var:.3f}, true_var={true_var:.3f}, rel_err={rel_err_var: .2%}")

    print("[End sanity checks]\n")
# ──────────────────────────────────────────────────────────────────────────────



if __name__ == "__main__":
    # Simple dict for all params
    config = {
        "out_dir":      "data/gaussian_mixture",
        "n_samples":    10000,
        "dim":          3072,  # e.g., 32x32x3 for CIFAR-10
        "n_classes":    10,    # e.g., CIFAR-10 has 10 classes
        "seed":         1,
        "model_name":   "resnet34",
        "params_path":  "checkpoints/ce/resnet34_synth_dim-3072_classes-10/data_parameters.npz",
    }
    main(config)
