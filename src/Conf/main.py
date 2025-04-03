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
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

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
# Create k folds
    full_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size_train, shuffle=False, pin_memory=True, num_workers=6, prefetch_factor=2
        )
    
    list_error_labels = []
    for inputs, targets in tqdm(full_dataloader, total=len(full_dataloader), desc=f"Building Stratify"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        with torch.no_grad():      
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            error_labels = preds != targets

        list_error_labels.append(error_labels.cpu().numpy())

    error_labels = np.concat(list_error_labels, axis=0)
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.arange(len(dataset)), error_labels)):
        print(f"Processing fold {fold+1} ...")
        if False:
            fold_error_labels = error_labels[test_idx]
            misclassified = np.sum(fold_error_labels)
            correctly_classified = len(test_idx) - misclassified
            print(f"Fold {fold+1}: Misclassified samples: {misclassified}, Correctly classified samples: {correctly_classified}")
            continue
        # Create training and test subsets.
        fold_train_dataset = torch.utils.data.Subset(dataset, train_idx)
        fold_test_dataset = torch.utils.data.Subset(dataset, test_idx)

        train_dataloader = torch.utils.data.DataLoader(
            fold_train_dataset, batch_size=config.batch_size_train, shuffle=False,
            pin_memory=True, num_workers=6, prefetch_factor=2
        )
        test_dataloader = torch.utils.data.DataLoader(
            fold_test_dataset, batch_size=config.batch_size_test, shuffle=False,
            pin_memory=True, num_workers=6, prefetch_factor=2
        )
        val_dataloader = None
        method = get_method(config.method, model, **vars(config))

        if hasattr(method.method, "fit"):
            print(3 * "---", "Training method", 3 * "---")
            train_preds, train_targets, train_scores = method.fit(train_dataloader, val_dataloader, name_save_file = f"train_fold{fold}")
            evaluate_classification(train_preds, train_targets, train_scores, config.results_folder, name_save_file = f"train_fold{fold}")   
        

        print(3 * "---", "Testing method", 3 * "---")
        test_preds, test_targets, test_scores = method.evaluate(test_dataloader, name_save_file = f"test_fold{fold}")
        evaluate_classification(test_preds, test_targets, test_scores, config.results_folder, name_save_file = f"test_fold{fold}")

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
