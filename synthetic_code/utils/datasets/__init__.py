import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from torch.distributions import MultivariateNormal
import time
from tqdm import tqdm
DATA_DIR = os.environ.get("DATA_DIR", "./data")
CHECKPOINTS_DIR_BASE = os.environ.get("CHECKPOINTS_DIR", "checkpoints/")


# -------------------------------
# 1. Data Generation: Gaussian Mixture Dataset
# -------------------------------



# class GaussianMixtureDataset(Dataset):
#     """
#     Dataset for a mixture of Gaussians with chunked pre-generation.

#     Args:
#         n_samples (int): Total number of samples to generate.
#         means (Tensor): Tensor of shape [n_components, dim].
#         covs (Tensor): Tensor of shape [n_components, dim, dim].
#         weights (Tensor): Tensor of shape [n_components].
#         seed (int, optional): Random seed for reproducibility. Default: None.
#         block_size (int, optional): Number of samples to generate per block to
#             control peak memory usage. Default: 5000.
#     """
#     def __init__(self, n_samples, means, covs, weights, seed=None, block_size=5000):
#         super().__init__()
#         self.n_samples = n_samples
#         self.block_size = block_size

#         # Keep parameters on CPU
#         self.means = means.cpu()
#         self.covs = covs.cpu()
#         self.weights = weights.cpu()
#         self.cov_chols   = torch.linalg.cholesky(covs.cpu())  # [n_classes, dim, dim]
#         # Reproducible RNG on CPU
#         # self.rng = torch.Generator(device='cpu')
#         # if seed is not None:
#         #     self.rng.manual_seed(seed)

#         # Sample component indices for all samples
#         components = torch.multinomial(
#             self.weights,
#             self.n_samples,
#             replacement=True
#             # generator=self.rng
#         )

#         # Pre-generate samples and labels in blocks
#         samples_list = []
#         labels_list = []
#         for start in tqdm(range(0, self.n_samples, self.block_size), desc="Generating samples"):
#         # for start in range(0, self.n_samples, self.block_size):
#             end         = min(start + self.block_size, self.n_samples)
#             comps_block = components[start:end]  # [block_size]
#             B           = comps_block.size(0)

#             # Index means and covs for this block
#             means_block = self.means[comps_block]  # [block_size, dim]
#             # covs_block = self.covs[comps_block]    # [block_size, dim, dim]
#             L_block     = self.cov_chols[comps_block]

#             # Sample from the block of MVNs

#             # sample standard normals and apply transform
#             z           = torch.randn(B, self.means.size(1))
#             samples_block  = means_block + torch.bmm(L_block, z.unsqueeze(-1)).squeeze(-1)
#             # mvn = MultivariateNormal(
#             #     loc=means_block,
#             #     covariance_matrix=covs_block,
#             #     validate_args=False
#             # )
#             # samples_block = mvn.sample(sample_shape=())  # [block_size, dim] #, generator=self.rng

#             samples_list.append(samples_block)
#             labels_list.append(comps_block)

#         # Concatenate all blocks into final buffers
#         self.samples = torch.cat(samples_list, dim=0)  # [n_samples, dim]
#         self.labels = torch.cat(labels_list, dim=0)    # [n_samples]

#     def __len__(self):
#         return self.n_samples

#     def __getitem__(self, idx):
#         # Return a single sample and its label
#         return self.samples[idx], self.labels[idx]

import os
import torch
from torch.utils.data import Dataset

class GaussianMixtureDataset(Dataset):
    """
    Dataset for a mixture of Gaussians with fully pre-generated samples for
    speed and reproducibility.

    Either:
      - If `samples_path` and `labels_path` exist under `data_dir`, loads them from disk.
      - Otherwise, generates `n_samples` on the fly (deterministic, single-shot).

    Args:
        data_dir (str): Directory where pre-generated `samples.pt` and `labels.pt` live.
                        If files are missing, generation will occur in-memory.
        n_samples (int): Number of samples to generate if pre-generated files are absent.
        means (Tensor): [n_classes, dim] tensor of mixture means.
        covs (Tensor):  [n_classes, dim, dim] tensor of mixture covariances.
        weights (Tensor): [n_classes] tensor of class weights (sum to 1).
        seed (int, optional): RNG seed for reproducibility. Default: None.
    """
    def __init__(self, data_dir = None, 
                 n_samples = None,
                #   means=None, covs=None, weights=None, 
                #   seed=None,
                #  device="cpu"
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.n_samples = n_samples
        self.device = torch.device('cpu')

        samples_path = os.path.join(data_dir, 'samples.pt')
        labels_path  = os.path.join(data_dir, 'labels.pt')
        if os.path.exists(samples_path) and os.path.exists(labels_path):
            # Load pre-generated data
            self.samples = torch.load(samples_path)
            self.labels  = torch.load(labels_path)
        else:
            raise FileNotFoundError(
                f"Pre-generated samples not found at {samples_path} or {labels_path}. "
                "Please generate the dataset first."
            )
        # else:
        #     # Pre-generate all samples in one shot
        #     dim = means.size(1)
        #     n_classes = means.size(0)

        #     means_cpu   = means.to(device)
        #     covs_cpu    = covs.to(device)
        #     weights_cpu = weights.to(device)
        #     L_chols     = torch.linalg.cholesky(covs_cpu)

        #     # Deterministic CPU RNG for component draws
        #     gen = torch.Generator(device=device)
        #     if seed is not None:
        #         gen.manual_seed(seed)
        #     comps = torch.multinomial(weights_cpu, self.n_samples, replacement=True, generator=gen)

        #     # Allocate buffers
        #     samples = torch.empty(self.n_samples, dim, dtype=torch.float32)
        #     labels  = comps.clone()

        #     # Batch-by-class sampling
        #     for i in range(n_classes):
        #         idx = (comps == i).nonzero(as_tuple=True)[0]
        #         if idx.numel() == 0:
        #             continue
        #         # GPU-backed noise generation for speed
        #         z = torch.randn(idx.numel(), dim, device=device)
        #         block = (L_chols[i] @ z.T).T + means[i]
        #         samples[idx] = block.cpu()

        #     self.samples = samples
        #     self.labels  = labels
        #     # Optionally: save for future runs
        #     os.makedirs(data_dir, exist_ok=True)
        #     torch.save(self.samples, samples_path)
        #     torch.save(self.labels, labels_path)

        # Final shapes
        print(f"self.n_samples", self.n_samples)
        print("self.samples.size(0)", self.samples.size(0))
        assert self.samples.size(0) == self.n_samples 

        assert self.labels.size(0)  == self.n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # reshape if needed (e.g., for images):
        return self.samples[idx].view(3,32,32), self.labels[idx]


# class GaussianMixtureDataset(Dataset):
#     """
#     Dataset for a mixture of Gaussians with chunked pre-generation.

#     Args:
#         n_samples (int): Total number of samples to generate.
#         means (Tensor): Tensor of shape [n_components, dim].
#         covs (Tensor): Tensor of shape [n_components, dim, dim].
#         weights (Tensor): Tensor of shape [n_components].
#         seed (int, optional): Random seed for reproducibility. Default: None.
#         block_size (int, optional): Number of samples to generate per block to
#             control peak memory usage. Default: 5000.
#     """
#     def __init__(self, n_samples, means, covs, weights, seed=None):
#         super().__init__()
#         self.n_samples = n_samples

#         # Keep parameters on CPU
#         self.dim = means.size(1)  # Dimension of each sample
#         self.means = means.cpu()
#         self.covs = covs.cpu()
#         self.weights = weights.cpu()
#         self.cov_chols   = torch.linalg.cholesky(covs.cpu())  # [n_classes, dim, dim]
#         # Reproducible RNG on CPU
#         self.gen = torch.Generator(device='cpu')
#         if seed is not None:
#             self.gen.manual_seed(seed)

#         # Sample component indices for all samples
#         self.components = torch.multinomial(
#             self.weights,
#             self.n_samples,
#             replacement=True
#             # generator=self.rng
#         )

#     def __len__(self):
#         return self.n_samples

#     def __getitem__(self, idx):
#         comp = self.components[idx]
#         mean = self.means[comp]  # [block_size, dim]
#         L     = self.cov_chols[comp]  # [block_size, dim, dim]

#         # sample standard normals and apply transform
#         z       = torch.randn(self.dim)
#         sample  = mean + L @ z
#         # Return a single sample and its label
#         sample = sample.view(3, 32, 32)
#         return sample, comp



# class GaussianMixtureDataset(Dataset):
#     def __init__(self, n_samples, means, stds, weights, seed=None, device="cpu"):
#         """
#         Generates samples from a Gaussian mixture model.
        
#         Args:
#             n_samples (int): Number of samples to generate.
#             means (torch.Tensor): Tensor of shape [n_classes, dim]
#                                   e.g., [7, 10]
#             stds (torch.Tensor): Tensor of shape [n_classes, dim]
#                                  e.g., [7, 10]
#             weights (torch.Tensor): Tensor of shape [n_classes]
#                                     e.g., [7]
#         """
#         # if seed is not None:
#         #     torch.manual_seed(seed)
#         #     torch.cuda.manual_seed_all(seed)  # For GPU support if needed
#         #     torch.backends.cudnn.deterministic = True
#         # gen = torch.Generator(device=device)
#         # gen.manual_seed(seed)
        
#         self.n_samples = n_samples
#         self.means = means      # [n_classes, dim]
#         self.stds = stds        # [n_classes, dim]
#         self.weights = weights  # [n_classes]
#         self.n_classes, self.dim = means.shape

#         # Sample component indices using multinomial sampling
#         # components: [n_samples]
#         self.components = torch.multinomial(self.weights, n_samples, replacement=True
#                                             # , generator=gen
#                                             )
#         # Select means and stds for each sampled component
#         # chosen_means: [n_samples, dim]
#         chosen_means = self.means[self.components]
#         # chosen_stds: [n_samples, dim]
#         chosen_stds = self.stds[self.components]
#         # Sample from the normal distribution (elementwise).
#         # samples: [n_samples, dim]
#         self.samples = torch.normal(mean=chosen_means, std=chosen_stds
#                                     # , generator=gen
#                                     ).cpu()
#         # Labels are the indices of the components (classes)
#         # labels: [n_samples]
#         self.labels = self.components.cpu()

#     def __len__(self):
#         return self.n_samples

#     def __getitem__(self, idx):
#         # Returns:
#         #  sample: [dim] and label: scalar
#         return self.samples[idx], self.labels[idx]


def load_data_config(data_folder):
    """
    Loads the dataset configuration from the config.json file in data_folder.
    """
    config_path = os.path.join(data_folder, "config.json")
    with open(config_path, "r") as f:
        data_config = json.load(f)
    return data_config

def load_generating_params(data_folder):
    """
    Loads the generating means and standard deviation from dataset_stats.csv.
    
    Assumes that dataset_stats.csv contains one row per class for a given dataset.
    We look for rows corresponding to the test dataset (if not found, fallback to "train").
    Each row's "mean" and "std" are stored as semicolon–separated numbers.
    
    Returns:
        means (ndarray): Array of shape (n_classes, dim) containing the generating means.
        std (float): The standard deviation (assumed to be constant across classes).
    """
    stats_path = os.path.join(data_folder, "dataset_stats.csv")
    df = pd.read_csv(stats_path)
    
    # Use rows for "test_dataset" if available; otherwise fall back to "train"
    df_subset = df[df["dataset"] == "test_dataset"]
    if df_subset.empty:
        df_subset = df[df["dataset"] == "train"]
    
    df_subset = df_subset.sort_values(by="class")
    means_list = []
    std_list = []
    for _, row in df_subset.iterrows():
        # Strip the square brackets and split on any whitespace.
        mean_str = row["mean"].strip("[]")
        # Split by whitespace and convert each piece to float.
        mean_vals = np.array([float(v) for v in mean_str.split()])
        std_val = float(row["std"])
        means_list.append(mean_vals)
        std_list.append(std_val)
    
    means = np.stack(means_list, axis=0)
    # Assume the standard deviation is identical across classes.
    std = std_list[0]
    return means, std

def bayes_proba(X_input, means_input, std_input):
    """
    For each input x (rows of X_input), returns the probability distribution over classes,
    computed via:
    
        P(y=i | x) = exp(-||x - mu_i||^2/(2 std_input^2)) / sum_j exp(-||x - mu_j||^2/(2 std_input^2))
    
    Parameters:
        X_input (ndarray): Shape (n_samples, dim)
        means_input (ndarray): Shape (n_classes, dim)
        std_input (float): Standard deviation.
        
    Returns:
        probs (ndarray): Shape (n_samples, n_classes)
    """
    # Compute squared Euclidean distances.
    dists = np.sum((X_input[:, None, :] - means_input[None, :, :]) ** 2, axis=2)
    logits = -dists / (2 * std_input**2)
    # For numerical stability, subtract the maximum logit per sample.
    logits -= np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probs



from typing import Any, Dict, Type
import numpy as np	
import os
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageNet

from synthetic_code.utils.models import get_model_essentials


datasets_registry: Dict[str, Any] = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "svhn": SVHN,
    "imagenet": ImageNet,
}


# def get_synthetic_dataset(model_name: str, # to get the test transform
#                  checkpoint_dir: str = os.path.join(CHECKPOINTS_DIR_BASE, "ce"),
#                  n_samples_train = 1000,
#                 n_samples_test = 1000,
#                  seed_train: int = 0,
#                  seed_test: int = 1,
#                  device: str = "cpu",
#                  **kwargs) -> Dataset:
    
#     checkpoints_dir = os.path.join(checkpoints_dir, model_name)
#     config_model_path = os.path.join(checkpoints_dir, "config.json")

#     if not os.path.exists(config_model_path):
#         raise FileNotFoundError(f"Configuration file not found at {config_model_path}")
#     # Load the configuration file
#     with open(config_model_path, "r") as f:
#         config_model = json.load(f)

#     means = torch.tensor(config_model["means"]).to(device)
#     stds = torch.tensor(config_model["stds"]).to(device)
#     weights = torch.tensor(config_model["weights"]).to(device)

#     # Generate Dataset
#         # Create training and validation datasets.
#     train_dataset = GaussianMixtureDataset(n_samples_train, means, stds, weights, seed=seed_train)
#     val_dataset = GaussianMixtureDataset(n_samples_test, means, stds, weights, seed=seed_test)

#     return train_dataset, val_dataset

def get_synthetic_dataset(
        # model_name: str, # to get the test transform
                          data_name: str = "gaussian_mixture",
                #  checkpoint_dir: str = os.path.join(CHECKPOINTS_DIR_BASE, "ce"),
                 n_samples = 1000,
                 dim: int = 10,
                 n_classes: int = 7,
                 seed: int = 0,
                #  device: str = "cpu"
                 ) -> Dataset:
    


    # checkpoint_dir = os.path.join(checkpoint_dir, 
    #                                 model_name + f"_synth_dim-{input_dim}_classes-{n_classes}")
    
    # data_parameters = np.load(os.path.join(checkpoint_dir, "data_parameters.npz"))
    # means =  torch.from_numpy(data_parameters["means"].astype(np.float32))
    # covs =  torch.from_numpy(data_parameters["covs"].astype(np.float32))
    # weights =  torch.from_numpy(data_parameters["weights"].astype(np.float32))
    data_dir = os.path.join(DATA_DIR, data_name, 
                        f"dim-{dim}_classes-{n_classes}-seed-{seed}")
    return GaussianMixtureDataset(data_dir=data_dir, n_samples=n_samples)

    # return GaussianMixtureDataset(n_samples, means, covs, weights, seed=seed)

# def get_synthetic_dataset_old(model_name: str, # to get the test transform
#                  checkpoint_dir: str = os.path.join(CHECKPOINTS_DIR_BASE, "ce"),
#                  n_samples = 1000,
#                  seed: int = 0,
#                  device: str = "cpu") -> Dataset:
    
#     checkpoint_dir = os.path.join(checkpoint_dir, model_name)
#     config_model_path = os.path.join(checkpoint_dir, "config.json")

#     if not os.path.exists(config_model_path):
#         raise FileNotFoundError(f"Configuration file not found at {config_model_path}")
#     # Load the configuration file
#     with open(config_model_path, "r") as f:
#         config_model = json.load(f)

#     means = torch.tensor(config_model["means"]).to(device)
#     # stds = torch.tensor(config_model["stds"]).to(device)
#     covs = torch.tensor(config_model["covs"]).to(device)
#     weights = torch.tensor(config_model["weights"]).to(device)

#     return GaussianMixtureDataset(n_samples, means, covs, weights, 
#                                 #   seed=seed, device=device
#                                   )



def get_dataset(dataset_name: str, 
                 model_name: str, # to get the test transform
                 root: str,
                 shuffle: bool = False,
                 random_state: int = 0,
                 **kwargs) -> Dataset:

    if dataset_name not in datasets_registry.keys():
        raise ValueError(f"Dataset {dataset_name} not found")

    model_essentials = get_model_essentials(model_name, dataset_name)
    test_transform = model_essentials["test_transforms"]
    if not shuffle:
        return datasets_registry[dataset_name](
            root, train=False, 
            transform=test_transform, 
            download=True) 
    

    dataset = datasets_registry[dataset_name](
            root, train=False, 
            transform=test_transform, 
            download=True) 
    # reproducible permutation
    gen = torch.Generator()
    gen.manual_seed(random_state)
    perm = torch.randperm(len(dataset), generator=gen).tolist()

    return Subset(dataset, perm)

        