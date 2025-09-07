import os
import json
import torch
from torch.utils.data import  DataLoader, Subset
from torch.distributions import MultivariateNormal, Categorical
from synthetic_code.utils.models import BayesClassifier, MLPClassifier, get_model
from synthetic_code.utils.detection.factory import get_detector
from synthetic_code.utils.datasets import GaussianMixtureDataset, get_dataset, get_synthetic_dataset
from synthetic_code.utils.eval import DetectorEvaluator
import numpy as np
from tqdm import tqdm
from time import localtime, strftime
import pandas as pd
import joblib
from joblib import Parallel, delayed
from multiprocessing import Process, Queue, cpu_count
import random
from typing import Dict, Any, List, Tuple
import warnings
from copy import deepcopy
import multiprocessing
from itertools import product
from tqdm_joblib import tqdm_joblib

print("torch.get_num_threads() =", torch.get_num_threads())

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*force_all_finite.*"
)

GPU_ID = 1

CHECKPOINTS_DIR_BASE = os.environ.get("CHECKPOINTS_DIR", "checkpoints/")
DATA_DIR = os.environ.get("DATA_DIR", "./data")

import os, psutil
import torch

def log_mem(stage: str):
    pid = os.getpid()
    proc = psutil.Process(pid)
    rss = proc.memory_info().rss / (1024**2)
    vms = proc.memory_info().vms / (1024**2)
    print(f"[PID {pid}] {stage:20s} RSS={rss:6.1f}MiB  VMS={vms:6.1f}MiB")
    # if torch.cuda.is_available():
    #     a = torch.cuda.memory_allocated()/1024**2
    #     r = torch.cuda.memory_reserved()/1024**2
    #     print(f"     CUDA alloc={a:6.1f}MiB  cached={r:6.1f}MiB")


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

    list_methods = ["gini", "metric_learning", "clustering", "bayes", "max_proba"]
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


def append_results_to_file(config, results, result_file):

    config = _prepare_config_for_results(config)
    config = pd.json_normalize(config, sep="_")
    results = pd.concat([config, results], axis=1)
    # df_pa_table = pa.Table.from_pandas(results)
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    if not os.path.isfile(result_file):
        results.to_csv(result_file, header=True, index=False)
    else:  # it exists, so append without writing the header
        results.to_csv(result_file, mode="a", header=False, index=False)


# def append_results_to_file(config, results, result_file):

#     def noneify(d):
#         """
#         Return a new dict with the same keys (and nested dict‐structure),
#         but with every non‐dict value replaced by None.
#         """
#         out = {}
#         for k, v in d.items():
#             if isinstance(v, dict):
#                 out[k] = noneify(v)
#             else:
#                 out[k] = None
#         return out

#     os.makedirs(os.path.dirname(result_file), exist_ok=True)

#     timestamp = strftime("%Y-%m-%d_%H-%M-%S", localtime())
#     config["experiment"] = {}
#     config["experiment"]["datetime"] = timestamp

#     list_methods = ["gini", "metric_learning", "clustering", "bayes", "max_proba"]
#     method_name = config.get("method_name")

#     if method_name not in list_methods:
#         raise ValueError(f"Unknown method '{method_name}'")

#     for m in list_methods:
#         if m == method_name:
#             continue

#         subconf = config.get(m)
#         if isinstance(subconf, dict):
#             # reset all its keys to None
#             config[m] = noneify(subconf)
#         else:
#             # nothing to reset (either missing or not a dict)
#             # Optionally, you could initialize it:
#             # config[m] = {}
#             pass
#     if config["clustering"]["reduction"]["name"] is None:
#         config["clustering"]["reduction"]= dict.fromkeys(config["clustering"]["reduction"].keys(), None)
    


#     config = pd.json_normalize(config, sep="_")
#     config["experiment_folder"] = "bwe"
#     results = pd.concat([config, results], axis=1)
#     print(results)
#     print(f"Saving results to {result_file}")
#     # df_pa_table = pa.Table.from_pandas(results)
#     if not os.path.isfile(result_file):
#         results.to_csv(result_file, header=True, index=False)
#     else:  # it exists, so append without writing the header
#         results.to_csv(result_file, mode="a", header=False, index=False)





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
    dataset: torch.utils.data.Dataset, config: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader, DataLoader]:
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
    if config["data"]["seed_split"] is not None:
        # Use a generator for local reproducibility of the shuffle
        random.shuffle(perm)

    n_train_samples = int(n // config["data"]["r"])
    train_idx = perm[:n_train_samples]
    test_idx = perm[n_train_samples:]

    n_val = len(test_idx) // 5
    val_idx, test_idx = test_idx[:n_val], test_idx[n_val:]
    
    # log_mem("Before creating subsets")
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    # log_mem("After creating subsets")

    train_loader = DataLoader(
        train_dataset, batch_size=config["data"]["batch_size_train"], shuffle=False, pin_memory=True, num_workers=8
    )
    val__loader = DataLoader(
        val_dataset, batch_size=config["data"]["batch_size_test"], shuffle=False, pin_memory=True, num_workers=8
    )
    val_loader = DataLoader(
        test_dataset, batch_size=config["data"]["batch_size_test"], shuffle=False, pin_memory=True, num_workers=8
    )
    # log_mem("After creating dataloaders")
    
    return train_loader, val__loader, val_loader

    
        

def main(config):

    setup_seeds(config["seed"], config["data"]["seed_split"])

    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
    # log_mem("Downloading  model")
    model = get_model(model_name=config["model"]["name"], 
                    dataset_name=config["data"]["name"],
                    n_classes=config["data"]["n_classes"],
                    input_dim=config["data"]["dim"],
                    model_seed=config["model"]["model_seed"],
                    checkpoint_dir = os.path.join(CHECKPOINTS_DIR_BASE, "ce"),
                    desired_indices = config["data"]["class_subset"]
                    )
    model = model.to(device)
    model.eval()
    # log_mem("Model  Downloaded")

    train_loader, _, val_loader = prepare_dataloaders(dataset, config)

    detector = get_detector(deepcopy(config), model, device, experiment_folder, CHECKPOINTS_DIR_BASE)

    if hasattr(detector, "fit"):
        detector.fit(train_loader)

    method_name = config.get("method_name")
    evaluator_train = DetectorEvaluator(
        model, train_loader, device, 
        magnitude=config[method_name].get("magnitude", 0))
    fpr_train, tpr_train, thr_train, roc_auc_train, model_acc_train, aurc_train = evaluator_train.evaluate(detector)

    evaluator_val = DetectorEvaluator(
        model, val_loader, device, 
        magnitude=config[method_name].get("magnitude", 0)
    )
    fpr_val, tpr_val, thr_val, roc_auc_val, model_acc_val, aurc_val = evaluator_val.evaluate(detector)


    results = pd.DataFrame([{
        "fpr_train": fpr_train, "tpr_train": tpr_train, "thr_train": thr_train,
        "roc_auc_train": roc_auc_train, "model_acc_train": model_acc_train, "aurc_train": aurc_train,
        "fpr_val": fpr_val, "tpr_val": tpr_val, "thr_val": thr_val,
        "roc_auc_val": roc_auc_val, "model_acc_val": model_acc_val, "aurc_val": aurc_val
    }])

    return results


def writer_loop(queue: Queue, results_file: str):
    """
    Dedicated process: reads (config, results_df) tuples from queue and
    safely appends them to a single CSV via append_results_to_file.
    """
    while True:
        item = queue.get()
        if item is None:  # sentinel to exit
            break
        config, results = item
        append_results_to_file(config, results, results_file)

def run_experiment(params: dict):
    """
    Applies a single combination of hyperparameters to base_config,
    runs main(), and pushes (config, results_df) to the writer queue.
    """
    config = deepcopy(base_config)
    for path, val in params.items():
        set_nested(config, path, val)

    # Optionally skip baseline combos here if you wish

    results = main(config)
    queue.put((config, results))

def set_nested(config: dict, key_path: str, value):
    """
    Given 'a.b.c', sets config['a']['b']['c'] = value (creating intermediate dicts).
    """
    keys = key_path.split('.')
    d = config
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value

if __name__ == "__main__":
    # import cProfile, pstats
    global base_config, queue

    base_config = {
        "seed" : 1,
        "data" : {
            "name" : "imagenet", # gaussian_mixture or cifar10
            "n_classes" : 1000, # 10 classes for gaussian_mixture
            "dim" : None, # 3072
            "n_samples" : None,
            "seed" : None,
            "seed_split" : 1,
            "r" : 2,
            # "n_samples_train" : 5000,
            # "n_samples_test" : 5000,
            "batch_size_train" : 256,
            "batch_size_test" : 256,
            # "seed_train" : 3,
            # "seed_test" : -5,
            "class_subset" : None
                  },
        "model" : {
            "name" : "vit_base16", # mlp_synth_dim-10_classes-7
            "model_seed" : 1,
                   },
        "method_name" : "clustering",
        "metric_learning" : {
            "lbd" : 0.80, 
            "temperature" : 1.1,
            "magnitude" : 0.0025, # 0.1
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
            "distance" : None, # "euclidean", "kl", "js", "alpha-divergence", "Itakura_Saito"
            "n_clusters" : 150,
            "reorder_embs" : True, # True or False
            "seed" : 0,
            "alpha" : 0.05,
            "init_scheme" : "kmeans",
            "n_init" : 5,
            "space" : "probits", # "feature" or "classifier"
            "cov_type" : "diag",
            "temperature" : 10.7,
            "normalize_gini" : False,
            "reduction" : {
                "name" : None, # umap
                "dim" : 10,
                "n_neighbors" : 15,
                "seed" : 0,
                }
            },
  
            }
    print("Dataset:", base_config["data"]["name"])
    print("Model:", base_config["model"]["name"])
    print("Method:", base_config["method_name"])

    download_data = True
    experiment_folder = None
    
    results_file = f"synth_results/{base_config['data']['name']}/all_results.csv" # "synth_results/resnet3072_test/all_results.csv"  
    # log_mem("start download_data")
    if download_data:
        # Download the dataset if it does not exist
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
                shuffle=False)
    # log_mem("data downloaded")

    if base_config["data"]["class_subset"] is not None:
        sub_idx = [i for i, label in enumerate(dataset.targets)
             if label in base_config["data"]["class_subset"]]
        dataset = Subset(dataset, sub_idx)
        print("Subsetting dataset to classes", base_config["data"]["class_subset"], "->", len(dataset), "samples")
            


    # magnitudes = [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.0040, 0.0036, 0.0038, 0.004, 0.0042, 0.0044, 0.0046, 0.0048]
    # tempratures = [0.05 + 0.05 * i for i in range(40)]
    # lambdas = [0.80 + 0.02 * i for i in range(7)]
    # n_clusters = list(range(80, 100, 2))
    # temperatures = [3.1 + 0.2 * i for i in range(7)]
    # seed_splits = list(range(1, 10))
    temperatures = [0.5 + 0.2 * i for i in range(7)]
    magnitudes = [0.0002 * i for i in range(7)]
    seed_splits = list(range(1, 10))
    # temperatures = [0.01 + 0.02 * i for i in range(20)]
    # temperatures = [100 + 100 * i for i in range(10)]
    # magnitudes = [0. + 0.0007 * i for i in range(3)] 

    parameter_space = {
        # 'metric_learning.lbd': lambdas,
        # 'data.name' : ["cifar10", "cifar100"],
        # 'gini.magnitude': magnitudes,
        # 'gini.temperature': temperatures,
        'data.seed_split': seed_splits
    }

    keys, values = zip(*parameter_space.items())
    grid = [dict(zip(keys, combo)) for combo in product(*values)]
    
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    queue = Queue(maxsize=10000)
    writer = Process(target=writer_loop, args=(queue, results_file))
    writer.start()

    # --- 5. Parallel dispatch ---
    # n_jobs = max(1, round(cpu_count() * 0.5))
    n_jobs = 1


    # 2) wrap your Parallel call in a tqdm_joblib context:
    with tqdm_joblib(tqdm(desc="Experiments", total=len(grid))) as progress_bar:
        Parallel(
            n_jobs=n_jobs,
            backend='multiprocessing',
            verbose=0 , # silence joblib’s own logging,

        )(
            delayed(run_experiment)(params)
            for params in grid
        )
    # Parallel(n_jobs=n_jobs, backend='multiprocessing', verbose=10)(
    #     delayed(run_experiment)(params)
    #     for params in grid
    # )

    # --- 6. Shutdown writer ---
    queue.put(None)
    writer.join()
    print("All experiments completed and results_file populated.")