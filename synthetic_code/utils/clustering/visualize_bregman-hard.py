#!/usr/bin/env python

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Adjust this if your package root differs:
ROOT_PATH = "/home/lamsade/msammut/error_detection/error-estimation/synthetic_code"
sys.path.append(ROOT_PATH)

from utils.clustering.models import BregmanHard
from utils.clustering.divergences import (
    euclidean,
    kullback_leibler,
    itakura_saito,
    alpha_divergence_factory,
)

def main():
    # 1) Prepare divergences
    divergences = {
        "Euclidean": euclidean,
        "KL": kullback_leibler,
        "Itakura_Saito": itakura_saito,
        "Alpha0.5": alpha_divergence_factory(0.5),
    }

    # 2) Generate data
    seed = 42
    rng = np.random.default_rng(seed)
    n_per = 1000
    alpha1 = [2, 5, 1]
    alpha2 = [5, 1, 2]
    X1 = rng.dirichlet(alpha1, size=n_per)
    X2 = rng.dirichlet(alpha2, size=n_per)
    X = np.vstack([X1, X2])

    out_dir = os.getcwd()

    # 3) Cluster & plot each divergence
    for name, div in divergences.items():
        bh = BregmanHard(
            n_clusters=3,
            divergence=div,
            max_iter=10,
            tol=1e-4,
            initializer="random",
            n_init=2,
            random_state=1234,
            verbose=False,
        )
        bh.fit(X)
        labels = bh.predict(X)

        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=labels, s=5)
        plt.title(f"BregmanHard ({name})")
        fname = f"clusters_{name}.png"
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()
        print(f"Saved {fname}")

    # 4) Baseline Euclidean KMeans
    km = KMeans(
        n_clusters=3,
        init="random",
        n_init=2,
        max_iter=10,
        tol=1e-4,
        random_state=1234,
        verbose=False,
    )
    labels_km = km.fit_predict(X)
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels_km, s=5)
    plt.title("KMeans (Euclidean)")
    km_fname = "clusters_KMeans_Euclidean.png"
    plt.savefig(os.path.join(out_dir, km_fname))
    plt.close()
    print(f"Saved {km_fname}")

if __name__ == "__main__":
    main()
