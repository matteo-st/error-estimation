import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# -------------------------------
# 1. Data Generation: Gaussian Mixture Dataset
# -------------------------------
class GaussianMixtureDataset(Dataset):
    def __init__(self, n_samples, means, stds, weights, seed=None):
        """
        Generates samples from a Gaussian mixture model.
        
        Args:
            n_samples (int): Number of samples to generate.
            means (torch.Tensor): Tensor of shape [n_classes, dim]
                                  e.g., [7, 10]
            stds (torch.Tensor): Tensor of shape [n_classes, dim]
                                 e.g., [7, 10]
            weights (torch.Tensor): Tensor of shape [n_classes]
                                    e.g., [7]
        """
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For GPU support if needed
            torch.backends.cudnn.deterministic = True
        
        self.n_samples = n_samples
        self.means = means      # [n_classes, dim]
        self.stds = stds        # [n_classes, dim]
        self.weights = weights  # [n_classes]
        self.n_classes, self.dim = means.shape

        # Sample component indices using multinomial sampling
        # components: [n_samples]
        self.components = torch.multinomial(self.weights, n_samples, replacement=True)
        # Select means and stds for each sampled component
        # chosen_means: [n_samples, dim]
        chosen_means = self.means[self.components]
        # chosen_stds: [n_samples, dim]
        chosen_stds = self.stds[self.components]
        # Sample from the normal distribution (elementwise).
        # samples: [n_samples, dim]
        self.samples = torch.normal(mean=chosen_means, std=chosen_stds)
        # Labels are the indices of the components (classes)
        # labels: [n_samples]
        self.labels = self.components

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Returns:
        #  sample: [dim] and label: scalar
        return self.samples[idx], self.labels[idx]


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
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageFolder, ImageNet

datasets_registry: Dict[str, Any] = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "svhn": SVHN,
    "imagenet": ImageNet,
}


def get_dataset(dataset_name: str, root: str, **kwargs) -> Dataset:

    if dataset_name["name"] == "synthetic":
        # base_dir = os.path.join(root, f"synthetic/dim-{dataset_name['dim']}_classes-{dataset_name['classes']}")

        train_path = os.path.join(root, "train_detector_dataset.npz")
        concentration_path = os.path.join(root, "concentration_dataset.npz")
        test_path = os.path.join(root, "test_dataset.npz")

        train_npz = np.load(train_path)
        concentration_npz = np.load(concentration_path)
        test_npz = np.load(test_path)

        X_train = torch.tensor(train_npz["X"], dtype=torch.float32)
        y_train = torch.tensor(train_npz["y"], dtype=torch.float32)
        X_concentration = torch.tensor(concentration_npz["X"], dtype=torch.float32)
        y_concentration = torch.tensor(concentration_npz["y"], dtype=torch.float32)
        X_test = torch.tensor(test_npz["X"], dtype=torch.float32)
        y_test = torch.tensor(test_npz["y"], dtype=torch.float32)

        train_dataset = TensorDataset(X_train, y_train)
        concentration_dataset = TensorDataset(X_concentration, y_concentration)
        test_dataset = TensorDataset(X_test, y_test)

        return train_dataset, concentration_dataset, test_dataset


    elif dataset_name is not None:
        return datasets_registry[dataset_name](root, **kwargs)
    else:
        try:
            return ImageFolder(root, **kwargs)
        except:
            raise ValueError(f"Dataset {root} not found")