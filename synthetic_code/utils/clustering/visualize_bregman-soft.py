#!/usr/bin/env python

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Adjust this if your package root differs:
ROOT_PATH = "/home/lamsade/msammut/error_detection/error-estimation/synthetic_code"
sys.path.append(ROOT_PATH)

out_dir = os.path.join(ROOT_PATH, "utils", "clustering", "visualizations")
os.makedirs(out_dir, exist_ok=True)

def main():
    # 1) Prepare divergences

    # 2) Generate data
    seed = 42
    rng = np.random.default_rng(seed)
    n_per = 1000
    alpha1 = [2, 5, 1]
    alpha2 = [5, 1, 2]
    X1 = rng.dirichlet(alpha1, size=n_per)
    X2 = rng.dirichlet(alpha2, size=n_per)
    X = np.vstack([X1, X2])



    n_clusters = 2
    seed = 42
    initializer = "random"  # or "kmeans", "kmedoids", etc.
    max_iter = 1000  # number of iterations for the clustering algorithm
    n_init = 10  # number of iterations for the initializer
    covariance_type = "spherical"  # "full", "tied", "diag", "spherical"

    rng = np.random.default_rng(seed)
    seed_2 = 1234

    # 2. Sample from two Dirichlet distributions
    n_per = 1000
    alpha1 = [2, 5, 1]    # first mixture component
    alpha2 = [5, 1, 2]    # second mixture component
    X1 = rng.dirichlet(alpha1, size=n_per)
    X2 = rng.dirichlet(alpha2, size=n_per)
    X  = np.vstack([X1, X2])    

    # True labels
    y_true = np.array([0] * n_per + [1] * n_per)

    # Clustering
    mixture = GaussianMixture(
        n_components=n_clusters,
        init_params=initializer,
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed_2,
        verbose=0
    )
    y_pred = mixture.fit_predict(X)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, s=5)
    axes[0].set_title('Original Labels')
    axes[0].set_xlabel('X[:, 0]')
    axes[0].set_ylabel('X[:, 1]')

    axes[1].scatter(X[:, 0], X[:, 1], c=y_pred, s=5)
    axes[1].set_title('Predicted Labels')
    axes[1].set_xlabel('X[:, 0]')
    axes[1].set_ylabel('X[:, 1]')
    plt.tight_layout()
    fname = f"cluster_GaussianMixtures.png"
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()
    print(f"Saved {fname}")


if __name__ == "__main__":
    main()
