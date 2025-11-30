from .my_soft_kmeans import SoftKMeans as TorchSoftKMeans
from .kmeans import KMeans as TorchKMeans

quantizers = {
    "soft-kmeans_torch": TorchSoftKMeans,
    "kmeans_torch": TorchKMeans,
}
