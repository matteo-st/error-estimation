import os
import sys
from sklearn.cluster import KMeans
import numpy as np
ROOT_PATH = "/home/lamsade/msammut/error_detection/error-estimation/synthetic_code"
sys.path.append(ROOT_PATH)

from utils.clustering.models import BregmanHard

n_clusters = 3
seed = 42
initializer = "random"  # or "kmeans", "kmedoids", etc.
max_iter = 10  # number of iterations for the clustering algorithm
n_init = 2  # number of iterations for the initializer

rng = np.random.default_rng(seed)
seed_2 = 1234

# 2. Sample from two Dirichlet distributions
n_per = 1000
alpha1 = [2, 5, 1]    # first mixture component
alpha2 = [5, 1, 2]    # second mixture component
X1 = rng.dirichlet(alpha1, size=n_per)
X2 = rng.dirichlet(alpha2, size=n_per)
X  = np.vstack([X1, X2])    

kmeans = KMeans(n_clusters=n_clusters,
                    init=initializer,
                    n_init=n_init,
                    max_iter=max_iter,
                 random_state=seed_2,
                 verbose=1)
labels = kmeans.fit_predict(X)
print(np.unique(labels, return_counts=True))
print("KMeans Inertia:", kmeans.inertia_)
bregmanhard = BregmanHard(
    n_clusters=n_clusters,
    max_iter=max_iter,
    initializer=initializer,
    n_init=n_init,
    random_state=seed_2,
    verbose=True)

bregmanhard.fit(X)
predict_bregamn = bregmanhard.predict(X)
print("Bregman Inertia:", bregmanhard.inertia_)
print(np.unique(predict_bregamn, return_counts=True))

