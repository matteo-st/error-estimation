import os
import json
import torch
from torch.utils.data import  DataLoader, Subset
from torch.distributions import MultivariateNormal, Categorical
from code.utils.models import BayesClassifier, MLPClassifier, get_model
from code.utils.detection.factory import get_detector
from code.utils.datasets import GaussianMixtureDataset, get_dataset, get_synthetic_dataset
from code.utils.eval import DetectorEvaluator, MultiDetectorEvaluator
import numpy as np
from tqdm import tqdm
from time import localtime, strftime
import pandas as pd
import joblib
import random
from typing import Dict, Any, List, Tuple
import warnings
from copy import deepcopy
from itertools import product
from code.utils import set_nested
from code.utils.detection.methods import MultiDetectors

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*force_all_finite.*"
)

GPU_ID = 1
N_THREADS = 8
os.environ["OMP_NUM_THREADS"]   = f"{N_THREADS}"
os.environ["MKL_NUM_THREADS"]   = f"{N_THREADS}"

torch.set_num_threads(N_THREADS)
torch.set_num_interop_threads(N_THREADS)

# 4. Verify settings
print("OMP_NUM_THREADS =", os.getenv("OMP_NUM_THREADS"))
print("MKL_NUM_THREADS =", os.getenv("MKL_NUM_THREADS"))
print("torch.get_num_threads() =", torch.get_num_threads())
print("torch.get_num_interop_threads() =", torch.get_num_interop_threads())

CHECKPOINTS_DIR_BASE = os.environ.get("CHECKPOINTS_DIR", "checkpoints/")
DATA_DIR = os.environ.get("DATA_DIR", "./data")


def make_config_list(base_config: dict, parameter_space: dict | None) -> list[dict]:
    """
    Expand a dict of parameter lists into a list of full configs.
    If parameter_space is empty/None, return a single config (base_config).
    """
    if not parameter_space:                     # covers {}, None
        return [deepcopy(base_config)]

    keys, values = zip(*parameter_space.items()) 
    grid = [dict(zip(keys, combo)) for combo in product(*values)]
    list_configs = []
    for params in grid:
        config = deepcopy(base_config)
        for path, val in params.items():
            set_nested(config, path, val) 
        list_configs.append(config)
    return list_configs



def _prepare_config_for_results(config, experiment_nb=None):

    def noneify(d):
        """
        Return a new dict with the same keys (and nested dict‐structure),
        but with every non‐dict value replaced by None.
        """
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out[k] = noneify(v)
            else:
                out[k] = None
        return out

    timestamp = strftime("%Y-%m-%d_%H-%M-%S", localtime())
    config["experiment"] = {}
    config["experiment"]["datetime"] = timestamp
    if experiment_nb is not None:
        config["experiment"]["folder"] = f"experiment_{experiment_nb}"
    else:
        config["experiment"]["folder"] = "bwe"

    list_methods = ["gini", "metric_learning", "clustering", "bayes", "max_proba","knn", "logistic"]
    method_name = config.get("method_name")

    if method_name not in list_methods:
        raise ValueError(f"Unknown method '{method_name}'")

    for m in list_methods:
        if m == method_name:
            continue

        subconf = config.get(m)
        if isinstance(subconf, dict):
            # reset all its keys to None
            config[m] = noneify(subconf)
        else:
            # nothing to reset (either missing or not a dict)
            # Optionally, you could initialize it:
            # config[m] = {}
            pass
    if config["clustering"]["reduction"]["name"] is None:
        config["clustering"]["reduction"]= dict.fromkeys(config["clustering"]["reduction"].keys(), None)

    return config


def append_results_to_file(config, train_results, val_results, result_file):

    config = _prepare_config_for_results(config)
    config = pd.json_normalize(config, sep="_")
    results = pd.concat([config, train_results, val_results], axis=1)
    # print(results)
    print(f"Saving results to {result_file}")
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    # df_pa_table = pa.Table.from_pandas(results)
    if not os.path.isfile(result_file):
        results.to_csv(result_file, header=True, index=False)
    else:  # it exists, so append without writing the header
        results.to_csv(result_file, mode="a", header=False, index=False)


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
    batch_size_train=252, batch_size_test=252
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
    train_idx = perm[:n_train_samples]
    test_idx = perm[n_train_samples:]

    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)


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
            # transform=base_config["data"]["transform"],
            shuffle=False)
                
    if base_config["data"]["class_subset"] is not None:
        sub_idx = [i for i, label in enumerate(dataset.targets)
             if label in base_config["data"]["class_subset"]]
        dataset = Subset(dataset, sub_idx)
        print("Subsetting dataset to classes", base_config["data"]["class_subset"], "->", len(dataset), "samples")
            

    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
    
    model = get_model(model_name=base_config["model"]["name"], 
                    dataset_name=base_config["data"]["name"],
                    n_classes=base_config["data"]["n_classes"],
                    input_dim=base_config["data"]["dim"],
                    model_seed=base_config["model"]["model_seed"],
                    checkpoint_dir = os.path.join(CHECKPOINTS_DIR_BASE, "ce"),
                    desired_indices = base_config["data"]["class_subset"]
                    )
    model = model.to(device)
    model.eval()

    for seed_split in seed_splits[::-1]:  # reverse order
        print(f"Running seed split {seed_split}...")

        setup_seeds(base_config["seed"], seed_split)

        train_loader, val_loader = prepare_dataloaders(
            dataset, seed_split=seed_split, 
            ratio=base_config["data"]["r"], 
            batch_size_train=base_config["data"]["batch_size_train"], batch_size_test=base_config["data"]["batch_size_test"])

        detectors = [get_detector(deepcopy(config), model, device, experiment_folder, CHECKPOINTS_DIR_BASE) for config in list_configs]

        multi_detector = MultiDetectors(
            detectors=detectors, model=model, device=device
        )
        if hasattr(detectors[0], "fit"):
            multi_detector.fit(train_loader)

        method_name = base_config.get("method_name")

        evaluator_train = MultiDetectorEvaluator(
            model, train_loader, device, 
            magnitudes=[config[method_name].get("magnitude", 0) for config in list_configs],
            suffix="train",
            )
        list_train_results = evaluator_train.evaluate(detectors)

        evaluator_val = MultiDetectorEvaluator(
            model, val_loader, device, 
                    magnitudes=[config[method_name].get("magnitude", 0) for config in list_configs],
            suffix="val"
                    )
        list_val_results = evaluator_val.evaluate(detectors)


        if results_file is not None:
            for config, train_results, val_results in zip(list_configs, list_train_results, list_val_results):
                config["data"]["seed_split"] = seed_split
                append_results_to_file(config, train_results, val_results, results_file)


if __name__ == "__main__":
    # import cProfile, pstats
    
    base_config = {
        "seed" : 1,
        "data" : {
            "name" : "cifar10", # gaussian_mixture or cifar10
            "n_classes" : 10, # 10 classes for gaussian_mixture
            "dim" : 3072, # 3072
            "n_samples" : 10000,
            "seed" : None,
            "seed_split" : 1,
            "r" : 2,
            # "n_samples_train" : 5000,
            # "n_samples_test" : 5000,
            "batch_size_train" : 512,
            "batch_size_test" : 512,
            # "seed_train" : 3,
            # "seed_test" : -5,
            "class_subset" : None,
            # "transform": "test"
                  },
        "model" : {
            "name" : "resnet34", # mlp_synth_dim-10_classes-7, timm_vit_tiny16, timm_vit_base16
            "model_seed" : 1,
                   },
        "method_name" : "logistic",
        "metric_learning" : {
            "lbd" : 0.8, 
            "temperature" : 1.1,
            "magnitude" : 0, # 0.1
            }, 
        "gini" : {
            "temperature" : 1,
            "normalize_gini" : False,
            "magnitude" : 0, # 
            },        
        "max_proba" : {
            "temperature" : 1,
            "magnitude" : 0, #
            },             
        "clustering" : {
            "name" : "soft-kmeans", # "kmeans", "soft-kmeans", "bregman-hard"
            "distance" : None, # "euclidean", "kl", "js", "alpha-divergence"
            "n_clusters" : 70,
            "reorder_embs" : True, # True or False
            "seed" : 0,
            "alpha" : 0.05,
            "init_scheme" : "kmeans",
            "n_init" : 5,
            "space" : "probits", # "feature" or "classifier"
            "cov_type" : "diag",
            "temperature" : 7.1,
            "normalize_gini" : False,
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
                "feature_space": "probits"
            }
            }
    
    results_file = "results/cifar10_logistic/all_results.csv" # "synth_results/resnet3072_test/all_results.csv"  
    experiment_folder = None

    # Don't put seed_splits in parameter_space !
    seed_splits = list(range(1, 10))  # [1, 2, 3, 4, 5]

    # Hyperparameter grid
    n_neighbors = [10, 20, 30, 40, 50, 60, 80, 100]
    # magnitudes = [0 + 0.001 * i for i in range(1, 30)] 
    temperatures = [1 + 0.2 * i for i in range(10)]
    weights = ["uniform", "distance"]

    parameter_space = {
        # 'metric_learning.magnitude': [0.00001 + 0.00001 * i for i in range(100)],  # 0.1, 0.2, ..., 1.0
        # 'metric_learning.temperature': temperatures,  # 0.1, 0.2, ..., 1.0
        # 'metric_learning.magnitudes': magnitudes,
        # 'data.name' : ["cifar10", "cifar100"],
        # 'gini.magnitude': magnitudes,
        # 'knn.n_neighbors': n_neighbors,
        # 'knn.weights': weights,
        # 'knn.p': [1, 2, 3],
        # 'knn.temperature': temperatures,
        'logistic.penalty': ["l1", "l2"],
        'logistic.C': [0.001, 0.005, 0.01, 0.05, 0.1, 1, 10, 100],
        'logistic.reorder_embs': [True, False],
        # 'logistic.temperature': temperatures,
      
        # 'gini.temperature': temperatures,
        # 'data.transform': ["test", "custom1"]
        # 'max_proba.temperature': [0.1 + 0.1 * i for i in range(40)],  # 0.1, 0.2, ..., 1.0
        # 'max_proba.magnitude': [0. + 0.001 * i for i in range(20)],  # 0.001, 0.002, ..., 0.02
        # 'clustering.temperature': temperatures,
        # 'clustering.n_clusters': n_clusters,
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
    main(list_configs, base_config, seed_splits)
    t1 = time.time()
    print(f"Total time: {t1 - t0:.2f} seconds")
