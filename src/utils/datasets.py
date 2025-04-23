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
        


def get_dataset_cls(dataset_name: str) -> Type[Dataset]:
    try:
        return datasets_registry[dataset_name]
    except:
        return ImageFolder


def get_datasets_names():
    return list(datasets_registry.keys())
