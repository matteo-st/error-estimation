# Minimal parity check between your Torch KMeans and scikit-learn's KMeans.
# - Samples n points in R^d with torch
# - Compares only inertia and the first few cluster centers (no alignment, no labels)
#
# Usage:
# 1) Put your class in ./kmeans.py (same folder as the notebook)
# 2) Ensure its imports resolve (utils/distances, etc.), or adjust the import below.
# 3) Make sure scikit-learn is installed: pip install scikit-learn
#
# Feel free to edit n, d, k, seed, and init_method.

import os, sys, importlib
import numpy as np
import torch
from sklearn.cluster import KMeans as SKKMeans
from .kmeans import KMeans  as TorchKMeans  # Adjust if your class is named differently
import time
from .gaussian_mixture import GaussianMixture as GMTorch
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_info
print(threadpool_info())
# -------- params you may tweak --------
# n = 25000
# d = 1000
k = 200
seed = 0
init_method = "random"   # or "rnd" to match your class
num_init = 1
max_iter = 300
# tol = 1
first_centers_to_show = 3
# -------------------------------------
device = torch.device("cuda")
torch.manual_seed(seed)


temperature = 1
data_name = "imagenet"
model_name = "timm_vit_base16"
r=2
n_epochs = 1
seed_split = 9
transform = "test"
root = f"storage_latent/{data_name}_{model_name}_r-{r}_seed-split-{seed_split}/"
latent_path = root + f"logits_train_n-epochs{n_epochs}_transform-{transform}.pt"

pkg = torch.load(latent_path, map_location="cpu")
X = pkg["logits"].to(torch.float32)[:300]
X = torch.softmax(X / temperature, dim=1)
X = X.sort(dim=1, descending=True)[0]
Xt = X.float().unsqueeze(0)  # (1, n, d)


if True:

    ## Scikit
    gmskt = GaussianMixture(n_components=k, 
                            random_state=seed, 
                            covariance_type="diag", 
                            init_params=init_method,
                          n_init=num_init, 
                          verbose=0, 
                            # reg_covar=1e-3
                            )
    t0 = time.time()
    gmskt.fit(X.numpy())
    t1 = time.time()
    print(f"sklearn GMM took {t1-t0:.3f} seconds")
    print("sklearn GMM converged:", gmskt.converged_)
    print("n_iter sklearn GMM:", gmskt.n_iter_)
    print("sklearn GMM lower bound:", gmskt.lower_bound_)


    gmtorch = GMTorch(n_components=k, 
                    covariance_type="diag", 
                    n_features = X.shape[1],
                    # tol=1e-3, 
                    # max_iter=max_iter, 
                    # n_init=num_init, 
                    init_params=init_method,
                    random_state=seed,
                    # verbose=0
                    )
    t0 = time.time()
    gmtorch.fit(Xt.to(device))
    t1 = time.time()
    print(f"Torch GMM took {t1-t0:.3f} seconds")
    print("n_iter Torch GMM:", gmtorch.n_iter_)
    print("Torch GMM lower bound:", gmtorch.log_likheood)
    



else:
    # Prepare outputs
    results = {}

    # ---- Torch KMeans ----

    # Your API expects (BS, N, D) with BS=1


    t0 = time.time()
    tkm = TorchKMeans(
        init_method=init_method if init_method in ["k-means++", "rnd"] else "rnd",
        num_init=num_init,
        max_iter=max_iter,
        p_norm=2,
        # tol=tol,
        normalize=None,
        n_clusters=k,
        verbose=False,
        seed=seed,
    )
    res = tkm(Xt.to(device), k=k)
    print("type of results:", res.inertia)
    t1 = time.time()
    print(f"Torch KMeans took {t1-t0:.3f} seconds")
    # inertia_torch = float(tkm._result.inertia[0].cpu().numpy())
    # centers_torch = tkm._result.centers[0].cpu().numpy()
    # results["torch"] = (inertia_torch, centers_torch)
    # print(f"[Torch] inertia = {inertia_torch:.6f}")
    # print(f"[Torch] first {first_centers_to_show} centers:\n", centers_torch[:first_centers_to_show])


    # ---- sklearn KMeans ----
    t0 = time.time()
    sk = SKKMeans(
        n_clusters=k,
        init="k-means++" if init_method == "k-means++" else "random",
        n_init=num_init,
        max_iter=max_iter,
        # tol=tol,
        # algorithm="lloyd",
        random_state=seed,
    )
    print("dtype", X.numpy().dtype)
    print(X.numpy().flags.c_contiguous)
    sk.fit(X.numpy())
    inertia_sk = float(sk.inertia_)
    centers_sk = sk.cluster_centers_
    results["sklearn"] = (inertia_sk, centers_sk)
    print(f"[sklearn] inertia = {inertia_sk:.6f}")
    t1 = time.time()
    print(f"sklearn KMeans took {t1-t0:.3f} seconds")

    print("n_iter sklearn:", sk.n_iter_)

    # ## Bregman
    # from .models import BregmanHard
    # t0 = time.time()
    # breg = BregmanHard(
    #                 n_clusters=k,
    #                 n_init=num_init,
    #                 initializer=init_method,
    #                 random_state=seed,
    #                 tol=tol,
    #                 max_iter=max_iter,
    #                 )
    # clusters = breg.fit_predict(X.numpy())
    # print("inertia bregman:", breg.inertia_)
    # t1 = time.time()
    # print(f"Bregman KMeans took {t1-t0:.3f} seconds")
