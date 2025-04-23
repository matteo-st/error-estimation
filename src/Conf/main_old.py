import os
import argparse
from argparse import Namespace
from dataclasses import dataclass
import itertools
import json
import random
import numpy as np
from torch.autograd import Variable
import torch
import torch.utils.data
from torchvision.models.feature_extraction import get_graph_node_names
from time import strftime, localtime
from typing import Any, Tuple, Dict, List

from tqdm import tqdm
from src.RelU.methods import add_dropout_layer, enable_dropout, get_method
from src.utils.datasets import get_dataset
from src.utils.helpers import append_results_to_file, create_experiment_folder
from src.utils.models import _get_openmix_cifar100_transforms, _get_openmix_cifar10_transforms, get_model_essentials
import torchvision
import timm
import timm.data
from src.utils.eval import get_classification_metrics, evaluate_classification

# Global directories
DATA_DIR = os.environ.get("DATA_DIR", "data/")
CHECKPOINTS_DIR_BASE = os.environ.get("CHECKPOINTS_DIR", "checkpoints/")


# @dataclass
# class Config:
#     model: str
#     dataset: str
#     seed: int
#     style: str
#     split_ratio: int
#     batch_size_train: int
#     batch_size_test: int
#     method: str
#     results_folder: Optional[str] = None
#     # Additional parameters for clustering and other methods
#     n_clusters: int = 10
#     alpha: float = 0.05
#     cluster_space: Optional[str] = None
#     # You can add more parameters here (e.g., learning rates, dropout probabilities, etc.)


def get_model_and_dataset(config) -> Tuple[torch.nn.Module, torch.utils.data.Dataset]:

    CHECKPOINTS_DIR = os.environ.get("CHECKPOINTS_DIR", "checkpoints/")
    CHECKPOINTS_DIR = os.path.join(CHECKPOINTS_DIR, config.style)

    model_essentials = get_model_essentials(config.model_name, config.dataset)
    model = model_essentials["model"]
    test_transform = model_essentials["test_transforms"]

    try:
        w = torch.load(
            os.path.join(CHECKPOINTS_DIR, "_".join([config.model_name, config.dataset]), str(config.seed), "best.pth"), map_location="cpu"
        )
    except:
        w = torch.load(
            os.path.join(CHECKPOINTS_DIR, "_".join([config.model_name, config.dataset]), "last.pt"), map_location="cpu")
    w = {k.replace("module.", ""): v for k, v in w.items()}

    model.load_state_dict(w)
    # load data
    dataset = get_dataset(
        dataset_name=config.dataset, root=DATA_DIR, train=False, transform=test_transform, download=True
    )

    return model, dataset



def main(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    # model = get_model(args.model_name, args.seed)

    model, dataset = get_model_and_dataset(config)
    model = model.to(device)
    model.eval()

    # randomly permutate dataset
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    dataset = torch.utils.data.Subset(dataset, indices)

    # Data Preparation
    config.num_classes = {"cifar10": 10, "svhn": 10, "cifar100": 100, "imagenet": 1000}[config.dataset]
    n = len(dataset)
    num_train_samples = int(n // config.split_ratio)

    train_dataset = torch.utils.data.Subset(dataset, range(0, num_train_samples))
    test_dataset = torch.utils.data.Subset(dataset, range(num_train_samples, n))
    val_dataset = torch.utils.data.Subset(test_dataset, range(0, len(test_dataset) // 5))
    test_dataset = torch.utils.data.Subset(test_dataset, range(len(test_dataset) // 5, len(test_dataset)))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size_train, shuffle=False, pin_memory=True, num_workers=6, prefetch_factor=2
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size_test, shuffle=False, pin_memory=True, num_workers=6, prefetch_factor=2
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size_test, shuffle=False, pin_memory=True, num_workers=6, prefetch_factor=2
    )

    # Method
    method = get_method(config.method, model, **vars(config))

    print(3 * "---", "Training method", 3 * "---")
    train_preds, train_targets, train_scores = method.fit(train_dataloader, val_dataloader)
    evaluate_classification(train_preds, train_targets, train_scores, config.results_folder, name_save_file = "train")

    print(3 * "---", "Testing method", 3 * "---")
    test_preds, test_targets, test_scores = method.evaluate(test_dataloader)
    evaluate_classification(test_preds, test_targets, test_scores, config.results_folder, name_save_file = "test")

    print("End of main()")
    return 


if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config.json")
    experiment_folder = create_experiment_folder(config_path)

    with open(config_path, "r") as f:
        config = json.load(f)

    config["results_folder"] = experiment_folder
    # config = Config(**config_dict)
    config = Namespace(**config)
    main(config)
