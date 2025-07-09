import os
import sys
from sklearn.mixture import GaussianMixture
import numpy as np
ROOT_PATH = "/home/lamsade/msammut/error_detection/error-estimation/synthetic_code"
sys.path.append(ROOT_PATH)

from utils.clustering.models import BregmanSoft

n_clusters = 2
divergence = "mahalanobis"  # "kullback_leibler", "itakura_saito", "mahalanobis"
seed = 42
initializer = "random"  # or "kmeans", "kmedoids", etc.
max_iter = 10  # number of iterations for the clustering algorithm
n_init = 1  # number of iterations for the initializer
covariance_type = "diag"  # "full", "tied", "diag", "spherical"

rng = np.random.default_rng(seed)
seed_2 = 1234

# 2. Sample from two Dirichlet distributions
n_per = 1000
alpha1 = [2, 5, 1]    # first mixture component
alpha2 = [5, 1, 2]    # second mixture component
X1 = rng.dirichlet(alpha1, size=n_per)
X2 = rng.dirichlet(alpha2, size=n_per)
X  = np.vstack([X1, X2])    

mixture = GaussianMixture(
    n_components=n_clusters,
    init_params=initializer,
    n_init=n_init,
    max_iter=max_iter,
    random_state=seed_2,
    verbose=1
    )

labels = mixture.fit_predict(X)
print(np.unique(labels, return_counts=True))
# print("KMeans Inertia:", mixture.inertia_)

bregmanhard = BregmanSoft(
    n_clusters=n_clusters,
    divergence=divergence,
    max_iter=max_iter,
    initializer=initializer,
    n_init=n_init,
    random_state=seed_2,
    verbose=True)

bregmanhard.fit(X)
predict_bregamn = bregmanhard.predict(X)
# print("Bregman Inertia:", bregmanhard.inertia_)
print(np.unique(predict_bregamn, return_counts=True))

