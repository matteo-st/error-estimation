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
import random
from typing import Dict, Any, List, Tuple
import warnings
from copy import deepcopy

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*force_all_finite.*"
)

GPU_ID = 1
N_THREADS = 1
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
    print(results)
    print(f"Saving results to {result_file}")
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    # df_pa_table = pa.Table.from_pandas(results)
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



# def prepare_dataloaders(
#     dataset: torch.utils.data.Dataset, config: Dict[str, Any]
# ) -> Tuple[DataLoader, DataLoader, DataLoader]:
#     """
#     Splits the dataset and prepares train, validation, and test DataLoaders.

#     Args:
#         dataset (torch.utils.data.Dataset): The full dataset.
#         config (Dict[str, Any]): The data configuration.

#     Returns:
#         Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test dataloaders.
#     """
#     n = len(dataset)

#     perm = list(range(n))
#     if config["data"]["seed_split"] is not None:
#         # Use a generator for local reproducibility of the shuffle
#         random.shuffle(perm)

#     n_train_samples = int(n // config["data"]["r"])
#     train_idx = perm[:n_train_samples]
#     test_idx = perm[n_train_samples:]

#     n_val = len(test_idx) // 5
#     val_idx, test_idx = test_idx[:n_val], test_idx[n_val:]

#     train_dataset = Subset(dataset, train_idx)
#     val_dataset = Subset(dataset, val_idx)
#     test_dataset = Subset(dataset, test_idx)


#     train_loader = DataLoader(
#         train_dataset, batch_size=config["data"]["batch_size_train"], shuffle=False, pin_memory=True, num_workers=4
#     )
#     val__loader = DataLoader(
#         val_dataset, batch_size=config["data"]["batch_size_test"], shuffle=False, pin_memory=True, num_workers=4
#     )
#     val_loader = DataLoader(
#         test_dataset, batch_size=config["data"]["batch_size_test"], shuffle=False, pin_memory=True, num_workers=4
#     )
#     print("Length of train dataset:", len(train_dataset))
#     print("Length of validation dataset:", len(val_dataset))
#     print("Length of test dataset:", len(test_dataset))
#     return train_loader, val__loader, val_loader


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

    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)


    train_loader = DataLoader(
        train_dataset, batch_size=config["data"]["batch_size_train"], shuffle=False, pin_memory=True, num_workers=4
    )
    val__loader = None
    val_loader = DataLoader(
        test_dataset, batch_size=config["data"]["batch_size_test"], shuffle=False, pin_memory=True, num_workers=4
    )
    print("Length of train dataset:", len(train_dataset))
    print("Length of test dataset:", len(test_dataset))
    return train_loader, val__loader, val_loader

    
        

def main(config):

    setup_seeds(config["seed"], config["data"]["seed_split"])

    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cpu")  # Force CPU for reproducibility

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


    train_loader, _, val_loader = prepare_dataloaders(dataset, config)

    detector = get_detector(deepcopy(config), model, device, experiment_folder, CHECKPOINTS_DIR_BASE)

    if hasattr(detector, "fit"):
        detector.fit(train_loader)

    method_name = config.get("method_name")
    # evaluator_train = DetectorEvaluator(model, train_loader, device, magnitude=config[method_name].get("magnitude", 0),
    #                                     return_embs=return_embs, return_labels=return_labels, 
    #                                     path=os.path.join(experiment_folder, "detector_train_predictions.npz") if results_dir is not None else None
    #                                     )
    # fpr_train, tpr_train, thr_train, roc_auc_train, model_acc_train, aurc_train = evaluator_train.evaluate(detector, return_clusters=return_clusters)

    # evaluator_val = DetectorEvaluator(model, val_loader, device, magnitude=config[method_name].get("magnitude", 0),
    #                                   return_embs=return_embs, return_labels=return_labels,
    #                                   path=os.path.join(experiment_folder, "detector_val_predictions.npz") if results_dir is not None else None
    #             )       
    # fpr_val, tpr_val, thr_val = evaluator_val.evaluate(detector, return_clusters=return_clusters)
    evaluator_train = DetectorEvaluator(
        model, train_loader, device, 
        magnitude=config[method_name].get("magnitude", 0))
    fpr_train, tpr_train, thr_train, roc_auc_train, model_acc_train, aurc_train, aupr_err_train, aupr_success_train = evaluator_train.evaluate(detector)

    evaluator_val = DetectorEvaluator(
        model, val_loader, device, 
        magnitude=config[method_name].get("magnitude", 0)
    )
    fpr_val, tpr_val, thr_val, roc_auc_val, model_acc_val, aurc_val, aupr_err_val, aupr_success_val = evaluator_val.evaluate(detector)


    results = pd.DataFrame([{
        "fpr_train": fpr_train, "tpr_train": tpr_train, "thr_train": thr_train,
        "roc_auc_train": roc_auc_train, "model_acc_train": model_acc_train, "aurc_train": aurc_train, "aupr_err_train": aupr_err_train, "aupr_success_train": aupr_success_train,
        "fpr_val": fpr_val, "tpr_val": tpr_val, "thr_val": thr_val,
        "roc_auc_val": roc_auc_val, "model_acc_val": model_acc_val, "aurc_val": aurc_val, "aupr_err_val": aupr_err_val, "aupr_success_val": aupr_success_val
    }])


    # results = pd.DataFrame([{
    #     "fpr_train": fpr_train, "tpr_train": tpr_train, "thr_train": thr_train,
    #     "fpr_val": fpr_val, "tpr_val": tpr_val, "thr_val": thr_val
    #     }])

    # Save the results to a CSV file
    if results_dir is not None:
        results.to_csv(os.path.join(experiment_folder, "detector_results.csv"), index=False)
    if results_file is not None:
        append_results_to_file(config, results, results_file)



if __name__ == "__main__":
    # import cProfile, pstats
    
    config = {
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
            "class_subset" : None
                  },
        "model" : {
            "name" : "resnet34", # mlp_synth_dim-10_classes-7
            "model_seed" : 1,
                   },
        "method_name" : "clustering",
        "metric_learning" : {
            "lbd" : 0.95, 
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
            "distance" : None, # "euclidean", "kl", "js", "alpha-divergence"
            "n_clusters" : 70,
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
    return_embs = False
    return_clusters = False
    return_labels = False
    download_data = True
    
    results_file = "synth_results/cifar10_test/all_results.csv" # "synth_results/resnet3072_test/all_results.csv"  
    results_dir = None # "synth_results/resnet3072_test" # "synth_results/cifar10_restricted"
    experiment_folder = None

    # pr = cProfile.Profile()
    # pr.enable()

    if download_data:
        # Download the dataset if it does not exist
        if config["data"]["name"] == "gaussian_mixture":

            dataset = get_synthetic_dataset(
                dim= config["data"]["dim"],
                n_samples= config["data"]["n_samples"],
                n_classes = config["data"]["n_classes"],
                seed = config["data"]["seed"],
                )
        else:
            dataset = get_dataset(
                dataset_name=config["data"]["name"], 
                model_name=config["model"]["name"], 
                root=DATA_DIR,  
                shuffle=False)
                
    if config["data"]["class_subset"] is not None:
        sub_idx = [i for i, label in enumerate(dataset.targets)
             if label in config["data"]["class_subset"]]
        dataset = Subset(dataset, sub_idx)
        print("Subsetting dataset to classes", config["data"]["class_subset"], "->", len(dataset), "samples")
            

    # for n_cluster in [40]: 
    #     # for temperature in [6.9 + 0.2 * i for i in range(20)]: #[0.1 + 0.025 * i for i in range(40)]:
    #         for seed_split in range(1, 10):
    #             config["data"]["seed_split"] = seed_split
    #             # config["clustering"]["temperature"] = temperature
    #             config["clustering"]["n_clusters"] = n_cluster
    #             # config["clustering"]["distance"] = divergence
    #             print("Running with seed_split:", config["data"]["seed_split"])
    #             # print("Running with temperature:", temperature)
    #             print("Running with n_cluster:", n_cluster)
               

    # for seed_split in range(1, 10):
    #     config["data"]["seed_split"] = seed_split
    #     print("Running with seed_split:", config["data"]["seed_split"])

    # for magnitude in [
    #     0,
    #     0.0005,
    #     0.001,
    #     0.0015,
    #     0.002,
    #     0.0025,
    #     0.003,
    #     0.0035,
    #     0.0040,
    #     0.0036,
    #     0.0038,
    #     0.004,
    #     0.0042,
    #     0.0044,
    #     0.0046,
    #     0.0048]:
    #     for  temperature in [0.05 + 0.05 * i for i in range(40)]:
    #         for lbd in [0.80 + 0.01 * i for i in range(20)]:
    #             for seed_split in range(1, 10):

    #                 if (temperature == 1.1 and lbd == 0.95 and magnitude == 0.0025):
    #                     continue
    #                 if (temperature == 1 and lbd == 0.5 and magnitude == 0):
    #                     continue
    #                 config["metric_learning"]["temperature"] = temperature
    #                 config["metric_learning"]["lbd"] = lbd
    #                 config["metric_learning"]["magnitude"] = magnitude
    #                 config["data"]["seed_split"] = seed_split
    #                 print("Running with seed_split:", config["data"]["seed_split"]) 
    #                 print("Running with temperature:", temperature)
    #                 print("Running with lbd:", lbd)
    #                 print("Running with magnitude:", magnitude)
    import time
    t0 = time.time()
    # for magnitude in [0.0010, 0.0015 ]:

    #     for seed_split in range(1, 10):
    #         config["data"]["seed_split"] = seed_split
    #         config["metric_learning"]["magnitude"] = magnitude
    #         print("Running with seed_split:", config["data"]["seed_split"]) 
    #         print("Running with magnitude:", magnitude)
    # for seed_split in range(1, 10):
    #     config["data"]["seed_split"] = seed_split
    #     print("Running with seed_split:", config["data"]["seed_split"])

    if results_dir is not None:
        experiment_folder = create_experiment_folder(config, results_dir=results_dir)
    main(config)
    t1 = time.time()
    print(f"Total time: {t1 - t0:.2f} seconds")
