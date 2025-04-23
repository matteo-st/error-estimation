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
from synthetic_code.utils.model import MLP

# Global directories
DATA_DIR = "data/synthetic/dim-2_classes-3"
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
    CHECKPOINTS_DIR = os.path.join(CHECKPOINTS_DIR, config.style, config.model_name ,"best_mlp.pth")
    # Load the primary classifier.
  
    model = MLP(input_dim=2, hidden_dims=[64, 32], num_classes=3)
    model.load_state_dict(torch.load(CHECKPOINTS_DIR, map_location="cpu"))

    # load data
    train_dataset, concentration_dataset, test_dataset = get_dataset(
        dataset_name=config.dataset, root=DATA_DIR
    )

    return model, train_dataset, concentration_dataset, test_dataset



def main(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    # model = get_model(args.model_name, args.seed)

    model, train_dataset, concentration_dataset, test_dataset = get_model_and_dataset(config)
    model = model.to(device)
    model.eval()


    # Data Preparation

    train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.batch_size_train, shuffle=False, pin_memory=True, num_workers=6, prefetch_factor=2
        )
    
    concentration_dataloader = torch.utils.data.DataLoader(
            concentration_dataset, batch_size=config.batch_size_train, shuffle=False, pin_memory=True, num_workers=6, prefetch_factor=2
        )
    
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=config.batch_size_train, shuffle=False, pin_memory=True, num_workers=6, prefetch_factor=2
        )
    
    print('Samples Train :', len(train_dataloader.dataset))
    print('Samples Concentration :', len(concentration_dataloader.dataset))
    print('Samples Test :', len(test_dataloader.dataset))

    method = get_method(config.method, model, **vars(config))

    if hasattr(method.method, "fit"):
            print(3 * "---", "Training method", 3 * "---")
            train_preds, train_targets, train_scores = method.fit(train_dataloader, concentration_dataloader, name_save_file = f"train")
            evaluate_classification(train_preds, train_targets, train_scores, config.results_folder, name_save_file = f"train")   
        

    print(3 * "---", "Testing method", 3 * "---")
    test_preds, test_targets, test_scores = method.evaluate(test_dataloader, name_save_file = f"test")
    evaluate_classification(test_preds, test_targets, test_scores, config.results_folder, name_save_file = f"test")


    print("End of main()")
    return 


if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config.json")
    experiment_folder = create_experiment_folder(config_path, results_dir="synth_results")
    

    with open(config_path, "r") as f:
        config = json.load(f)
    config["results_folder"] = experiment_folder
    config = Namespace(**config)

    print(f"Dataset : {config.dataset['name']} - Model : {config.model_name} - Method : {config.method}")
    main(config)
