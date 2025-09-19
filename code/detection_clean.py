import os
import json
import torch
from torch.utils.data import  DataLoader, Subset
from torch.distributions import MultivariateNormal, Categorical
from code.utils.models import BayesClassifier, MLPClassifier, get_model
from code.utils.detection.factory import get_detector
from code.utils.datasets import GaussianMixtureDataset, get_dataset, get_synthetic_dataset
from code.utils.eval import MultiDetectorEvaluator
from code.utils.models import get_model_essentials
import numpy as np

import pandas as pd
import joblib
import random
from typing import Dict, Any, List, Tuple
import warnings
from copy import deepcopy
from code.utils.detection.methods import MultiDetectors, HyperparameterSearch
from code.utils.helper import make_config_list, _prepare_config_for_results

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*force_all_finite.*"
)

GPU_ID = 0
# N_THREADS = 8
# os.environ["OMP_NUM_THREADS"]   = f"{N_THREADS}"
# os.environ["MKL_NUM_THREADS"]   = f"{N_THREADS}"

# torch.set_num_threads(N_THREADS)
# torch.set_num_interop_threads(N_THREADS)

# # 4. Verify settings
# print("OMP_NUM_THREADS =", os.getenv("OMP_NUM_THREADS"))
# print("MKL_NUM_THREADS =", os.getenv("MKL_NUM_THREADS"))
print("torch.get_num_threads() =", torch.get_num_threads())
print("torch.get_num_interop_threads() =", torch.get_num_interop_threads())

CHECKPOINTS_DIR_BASE = os.environ.get("CHECKPOINTS_DIR", "checkpoints/")
DATA_DIR = os.environ.get("DATA_DIR", "./data")




def create_experiment_folder(config, results_dir = "synth_results/checking_reproducibility_split"):
    """
    Create a folder for the experiment results.
    
    Args:
        config (dict): Configuration dictionary containing parameters for the experiment.
        
    Returns:
        str: Path to the created experiment folder.
    """
    print("Creating experiment folder...", "results_dir:", results_dir)
    os.makedirs(results_dir, exist_ok=True)


    # Liste des dossiers d'expériences existants
    existing = [
        f for f in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, f)) and f.startswith("experiment_")
    ]
    # Extraction des numéros d'expérience
    numbers = []
    for folder in existing:
        try:
            number = int(folder.split("_")[1])
            numbers.append(number)
        except (IndexError, ValueError):
            continue
    experiment_nb = max(numbers) + 1 if numbers else 1
    config = _prepare_config_for_results(config, experiment_nb)
    experiment_folder = os.path.join(results_dir, f"experiment_{experiment_nb}")
    os.makedirs(experiment_folder, exist_ok=True)
    
    
    # Save the configuration file
    config_path = os.path.join(experiment_folder, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    return experiment_folder


def setup_seeds(seed: int, seed_split: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): The seed to use.
    """
    random.seed(seed_split)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def prepare_dataloaders(
    dataset: torch.utils.data.Dataset, seed_split=None, ratio=2,
    batch_size_train=252, batch_size_test=252, train_transform=None, config=None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Splits the dataset and prepares train, validation, and test DataLoaders.

    Args:
        dataset (torch.utils.data.Dataset): The full dataset.
        config (Dict[str, Any]): The data configuration.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test dataloaders.
    """
    n = len(dataset)

    perm = list(range(n))
    if seed_split is not None:
        # Use a generator for local reproducibility of the shuffle
        random.shuffle(perm)

    n_train_samples = int(n // ratio)
    # n_train_samples = 7000
    train_idx = perm[:n_train_samples]
    test_idx = perm[n_train_samples:]

    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)
    #test_dataset = torch.utils.data.Subset(test_dataset, range(len(test_dataset) // 5, len(test_dataset)))

    if train_transform is not None:
        transform = get_model_essentials(
            config["model"]["name"], 
            config["data"]["name"]
            )[f"{train_transform}_transforms"]
        train_dataset.dataset.transform = transform
        print("Transform for train dataset:", transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=False, pin_memory=True, num_workers=10
    )

    val_loader = DataLoader(
        test_dataset, batch_size=batch_size_test, shuffle=False, pin_memory=True, num_workers=10
    )
    
    print("Length of train dataset:", len(train_dataset))
    print("Length of test dataset:", len(test_dataset))
    return train_loader, val_loader

    
        

def main(list_configs, base_config, seed_splits):

    if base_config["data"]["name"] == "gaussian_mixture":

        dataset = get_synthetic_dataset(
            dim= base_config["data"]["dim"],
            n_samples= base_config["data"]["n_samples"],
            n_classes = base_config["data"]["n_classes"],
            seed = base_config["data"]["seed"],
            )
    else:
        dataset = get_dataset(
            dataset_name=base_config["data"]["name"], 
            model_name=base_config["model"]["name"], 
            root=DATA_DIR,
            transform=base_config["data"]["transform"],
            shuffle=False)
                
    if base_config["data"]["class_subset"] is not None:
        sub_idx = [i for i, label in enumerate(dataset.targets)
             if label in base_config["data"]["class_subset"]]
        dataset = Subset(dataset, sub_idx)
        print("Subsetting dataset to classes", base_config["data"]["class_subset"], "->", len(dataset), "samples")
            

    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
    from threadpoolctl import threadpool_info
    print(threadpool_info())
    # model = get_model(model_name=base_config["model"]["name"], 
    #                 dataset_name=base_config["data"]["name"],
    #                 n_classes=base_config["data"]["n_classes"],
    #                 input_dim=base_config["data"]["dim"],
    #                 model_seed=base_config["model"]["model_seed"],
    #                 checkpoint_dir = os.path.join(CHECKPOINTS_DIR_BASE, "ce"),
    #                 desired_indices = base_config["data"]["class_subset"]
    #                 )
    
    # model = model.to(device)
    # model.eval()
    
    # # for name, m in model.named_modules():
    # # #     print(name, m)
    # # exit()
    # for p in model.parameters():
    #     p.requires_grad_(False)  # freeze permanently
    model = None
    seed_split = base_config["data"]["seed_split"]
    print(f"Running seed split {seed_split}...")

    setup_seeds(base_config["seed"], seed_split)

    train_loader, val_loader = prepare_dataloaders(
        dataset, seed_split=seed_split, 
        ratio=base_config["data"]["r"], 
        batch_size_train=base_config["data"]["batch_size_train"], 
        batch_size_test=base_config["data"]["batch_size_test"],
        train_transform=base_config["data"]["transform"],
        config=base_config
        )

    detectors = [get_detector(deepcopy(config), model, device, experiment_folder, CHECKPOINTS_DIR_BASE) for config in list_configs]

    HyperparameterSearch(
        detectors=detectors, 
        model=model, 
        device=device, 
        base_config=base_config, 
        list_configs=list_configs, 
        train_loader=train_loader, 
        val_loader=val_loader,
        result_folder=results_folder)
            


if __name__ == "__main__":
    # import cProfile, pstats
    
    base_config = {
        "seed" : 1,
        "data" : {
            "name" : "imagenet", # gaussian_mixture or cifar10
            "n_classes" : 1000, # 10 classes for gaussian_mixture
            "dim" : 3072, # 3072
            "n_samples" : 10000,
            "seed" : None,
            "seed_split" : 9,
            "n_splits": 3,
            "n_epochs" : 1,
            "r" : 2,
            # "n_samples_train" : 5000,
            # "n_samples_test" : 5000,
            "batch_size_train" : 512,
            "batch_size_test" : 512,
            # "seed_train" : 3,
            # "seed_test" : -5,
            "class_subset" : None,              
            "transform": "test",},
        "model" : {
            "name" : "timm_vit_tiny16", # mlp_synth_dim-10_classes-7, timm_vit_tiny16, timm_vit_base16, densenet121
            "model_seed" : 1,
                   },
        "method_name" : "clustering",
        "metric_learning" : {
            "lbd" : 0.95, 
            "temperature" : 1,
            "magnitude" : 0., # 0.1
            }, 
        "gini" : {
            "temperature" : 1,
            "normalize_gini" : True,
            "magnitude" : 0.005, # 
            },        
        "max_proba" : {
            "temperature" : 1,
            "magnitude" : 0.002, #
            },             
        "clustering" : {
            "name" : "kmeans_torch", # "kmeans", "soft-kmeans", "bregman-hard", minikmeans
            "distance" : None, # "euclidean", "kl", "js", "alpha-divergence"
            "n_clusters" : 100,
            "reorder_embs" : True, # True or False
            "seed" : 1,
            "alpha" : 0.05,
            "init_scheme" : 'random', # kmeans, k-means++
            "n_init" : 10, #5
            "space" : "probits", # "feature" or "classifier"
            "cov_type" : "diag",
            "temperature" : 1,
            "pred_weight" : 0., # None or float
            "normalize_gini" : False,
            "batch_size": 2048,
            "reduction" : {
                "name" : None, # umap
                "dim" : 10,
                "n_neighbors" : 15,
                "seed" : 0,
                }
            },
            "knn" : {
                "n_neighbors" : 50,
                "weights" : "distance", # uniform or distance
                "p" : 2, # 0.1
                "metric" : "minkowski",
                "magnitude" : 0, # 0.1
                "space" : "probits", # "feature" or "classifier"
                "reorder_embs" : False, # True or False,
                "temperature" : 1
            },
            "logistic" : {
                "penalty" : "l2", # l1, l2, elasticnet
                "C" : 1,
                "reorder_embs" : False, # True or False,
                "space" : "probits", # "feature" or "classifier"
                "temperature" : 1,
                "feature_space": "probits",
            },
            "random_forest" : {
                "temperature" : 1,
                "space" : "logits", 
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",  # 0.3–1.0
                "ccp_alpha": 1e-3,
                "max_samples":  0.5,
                "class_weight": None,  # classifier only
            }
            }
    # for seed_split in [1, 2, 3, 4, 5, 7, 8, ]:
        # base_config["data"]["seed_split"] = seed_split
    root = f"results/{base_config['data']['name']}_{base_config['model']['name']}_r-{base_config['data']['r']}_seed-split-{base_config['data']['seed_split']}"
    if base_config['method_name'] in ["clustering", "metric_learning", "random_forest"]:
        root += f"/transform-{base_config['data']['transform']}_n-epoch{base_config['data']['n_epochs']}_n-folds{base_config['data']['n_splits']}_{base_config['clustering']['space']}"
        if base_config['method_name'] == "clustering":
            if base_config['clustering']['distance'] is not None:
                root += f"_distance-{base_config['clustering']['distance']}"
            if base_config['clustering']['name'] != "soft-kmeans":
                root += f"_{base_config['clustering']['name']}"
            # root += f"_probweights"
            
    else:
        root += f"/transform-{base_config['data']['transform']}"
    results_folder = root + f"_{base_config['method_name']}"

    # results_folder = f"results/{base_config['data']['name']}_{base_config['model']['name']}_r-{base_config['data']['r']}_seed-split-{base_config['data']['seed_split']}_transform-{base_config['data']['transform']}_n-epoch{base_config['data']['n_epochs']}/{base_config['method_name']}" # "synth_results/resnet3072_test/all_results.csv"
    experiment_folder = None

    # Don't put seed_splits in parameter_space !

    # Hyperparameter grid
    n_neighbors = [10, 20, 30, 40, 50, 60, 80, 100]
    # magnitudes = [0 + 0.001 * i for i in range(1, 30)] 
    temperatures = [1. + 0.1 * i for i in range(10)]  # 1.0, 1.1, ..., 1.9
    n_clusters = [250 + 20 * i for i in range(20)]  # 450, 470, ..., 650
    weights = ["uniform", "distance"]

    parameter_space = {
        # 'metric_learning.magnitude': [0. + 0.005 * i for i in range(10)],  # 0.1, 0.2, ..., 1.0
        # 'metric_learning.temperature': [0.5 + 0.2 * i for i in range(7)],  # 0.5, 0.7, ..., 1.5
        # 'metric_learning.lbd': [0.75 + 0.05 * i for i in range(2)],  # 0.7, 0.75, ..., 1.0
        # 'metric_learning.magnitudes': magnitudes,
        # 'data.name' : ["cifar10", "cifar100"],
        # 'gini.magnitude': magnitudes,
        # 'knn.n_neighbors': n_neighbors,
        # 'knn.weights': weights,
        # 'knn.p': [1, 2, 3],
        # 'knn.temperature': temperatures,
        # 'logistic.penalty': ["l1", "l2"],
        # 'logistic.C': [0.001, 0.005, 0.01, 0.05, 0.1, 1, 10, 100],
        # 'logistic.reorder_embs': [True, False],
        # 'logistic.temperature': temperatures,
        # 'gini.magnitude': [0 + 0.0005 * i for i in range(20)],
        # 'gini.temperature': [0.5 + 0.1 * i for i in range(20)], # [0.5 + 0.1 * i for i in range(20)],
        # 'data.transform': ["test", "custom1"]
        # 'gini.temperature': [0.7 + 0.1 * i for i in range(7)],  # 0.7, 0.8, ..., 1.3
        # 'gini.magnitude': [0. + 0.002 * i for i in range(10)],  # 0.001, 0.002, ..., 0.02
        'clustering.temperature': temperatures,
        # 'clustering.pred_weight': [0. + 0.0005 * i for i in range(3)],
        'clustering.n_clusters': n_clusters, # n_clusters
        # "n_estimators": [200, 300, 400],
        # "max_depth": [None, 10, 20, 30],
        # "min_samples_split": [2, 5],
        # "min_samples_leaf": [1,  3],
        # "max_features": ["log2"],  # 0.3–1.0
        # "ccp_alpha": [0.0, 1e-3],
        # "max_samples": [None, 0.7 ],
        # "class_weight": [None, "balanced", "balanced_subsample"],  # classifier only
}

    list_configs = make_config_list(base_config, parameter_space)  # test the function
    # keys, values = zip(*parameter_space.items())
    # grid = [dict(zip(keys, combo)) for combo in product(*values)]
    # list_configs = []
    # for params in grid:
    #     config = deepcopy(base_config)
    #     for path, val in params.items():
    #         set_nested(config, path, val) 
    #     list_configs.append(config)

    import time
    t0 = time.time()
    main(list_configs, base_config, base_config["data"]["seed_split"])
    t1 = time.time()
    print(f"Total time: {t1 - t0:.2f} seconds")
    # saving time
    with open(os.path.join(results_folder, "time.txt"), "w") as f:
        f.write(f"Total time: {t1 - t0:.2f} seconds\n")

