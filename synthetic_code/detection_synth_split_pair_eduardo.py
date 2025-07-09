import os
import json
import torch
from torch.utils.data import  DataLoader, Subset
from torch.distributions import MultivariateNormal, Categorical
from synthetic_code.utils.models import BayesClassifier, MLPClassifier, get_model
from synthetic_code.utils.detector import PartitionDetector, MetricLearningLagrange, BayesDetector, GiniDetector
from synthetic_code.utils.datasets import GaussianMixtureDataset, get_dataset, get_synthetic_dataset
from synthetic_code.utils.eval import DetectorEvaluator
import numpy as np
from tqdm import tqdm
from time import localtime, strftime
import pandas as pd
import joblib
import random

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
    os.makedirs(results_dir, exist_ok=True)
    # Create a unique folder name based on the current date and time
    timestamp = strftime("%Y-%m-%d_%H-%M-%S", localtime())
    config["experiment"] = {}
    config["experiment"]["datetime"] = timestamp

    method_name = config["method_name"]
    if method_name == "clustering":
        config["metric_learning"] = {k: None for k in config["metric_learning"].keys()}
    elif method_name == "metric_learning":
        config["clustering"] = {k: None for k in config["clustering"].keys()}

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
    new_number = max(numbers) + 1 if numbers else 1
    experiment_number = f"experiment_{new_number}"
    config["experiment"]["folder"] = experiment_number
    experiment_folder = os.path.join(results_dir, experiment_number)
    os.makedirs(experiment_folder, exist_ok=True)
    
    
    # Save the configuration file
    config_path = os.path.join(experiment_folder, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    return experiment_folder, experiment_number


    
        

def main(config):

    random.seed(config["data"]["seed_split"])
    # Prints random item 
    # print(random.random())
    np.random.seed(config["data"]["seed_split"])
    torch.manual_seed(config["data"]["seed_split"])
    torch.cuda.manual_seed_all(config["data"]["seed_split"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    # torch.use_deterministic_algorithms(True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(model_name=config["model"]["name"], 
                    dataset_name=config["data"]["name"],
                    model_seed=config["model"]["model_seed"],
                    checkpoint_dir = os.path.join(CHECKPOINTS_DIR_BASE, "ce")
                    )
    model = model.to(device)
    model.eval()
    # print("model keys", model.state_dict().keys())

    if config["data"]["name"] == "gaussian_mixture":

        dataset = get_synthetic_dataset(
            model_name=config["model"]["name"],
            n_samples = config["data"]["n_samples"],
            seed = config["data"]["seed"],
            device = device
            )
    else:
        dataset = get_dataset(
            dataset_name=config["data"]["name"], 
            model_name=config["model"]["name"], 
            root=DATA_DIR,  
            shuffle=False)


                                       
    n = len(dataset)
    print("Dataset size", n)
    # print("model first sample", model(train_dataset[0][0].unsqueeze(0).to(device)))
    # # reproducible permutation
    # gen = torch.Generator().manual_seed(config["data"]["seed_split"])
    # perm = torch.randperm(n, generator=gen).tolist()
    perm = list(range(len(dataset)))
    random.shuffle(perm)

    # dataset = Subset(dataset, perm)
    
    n_train_samples = int(n // config["data"]["r"])
    train_idx = perm[:n_train_samples]
    test_idx = perm[n_train_samples:]

    # carve out a val-subset of the test split
    n_val = len(test_idx) // 5
    val_idx, test_idx = test_idx[:n_val], test_idx[n_val:]

    # 4) make your Subsets (one wrap each)
    train_dataset = Subset(dataset, train_idx)
    # print("train_dataset first sample", train_dataset[0][0][0,0,:])
    val_dataset   = Subset(dataset, val_idx)
    # print("val_dataset first sample", val_dataset[0][0][0,0,:])
    test_dataset  = Subset(dataset, test_idx)
    # print("test_dataset first sample", test_dataset[0][0][0,0,:])


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["data"]["batch_size_train"], shuffle=False, pin_memory=True, num_workers=6, prefetch_factor=2
    )
    val__loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["data"]["batch_size_test"], shuffle=False, pin_memory=True, num_workers=6, prefetch_factor=2
    )
    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["data"]["batch_size_test"], shuffle=False, pin_memory=True, num_workers=6, prefetch_factor=2
    )
  
    
    if config["method_name"] == "clustering":

        detector = PartitionDetector(model, 
                                    #  dataset.weights, dataset.means, dataset.stds, 
                                    n_cluster=config["clustering"]["n_clusters"], 
                                    alpha=config["clustering"]["alpha"], 
                                    method=config["clustering"]["name"], 
                                    device=device, 
                                    kmeans_seed=config["clustering"]["seed"], 
                                     init_scheme=config["clustering"]["init_scheme"],
                                    temperature=config["clustering"]["temperature"],
                                    partionning_space=config["clustering"]["space"],
                                    #  cov_type=config["clustering"]["cov_type"]
                                    )
    elif config["method_name"] == "metric_learning":
        detector = MetricLearningLagrange(model, 
                                            lbd=config["metric_learning"]["lambda"], 
                                            temperature=config["metric_learning"]["temperature"]
                                            )
    detector.fit(train_loader)


    # detector = GiniDetector(model, temperature=config["temperature"], normalize = config["normalize_gini"], device=device)
    # detector = BayesDetector(model, weights, means, stds, config_model["n_classes"], device=device)
    # print("converged ?", detector.clustering_algo.converged_)
   
    if (config["method_name"] == "clustering"):
        if (config["clustering"]["name"] == "soft-kmeans"):
            # with open(os.path.join(experiment_folder, 'clustering_algo.pkl'), "wb") as f:
            #     pickle.dump(detector.clustering_algo, f)
            joblib.dump(detector.clustering_algo, os.path.join(experiment_folder, 'clustering_algo.pkl'))
        # Save the clustering algorithm

    if (config["method_name"] == "clustering"):
        if (config["clustering"]["name"] == "kmeans"):
            cluster_center =  detector.clustering_algo.cluster_centers_.flatten()
            # sort_permut = np.argsort(cluster_center)
            # print("sort_permut", sort_permut)

            # cluster_center_sorted = cluster_center[sort_permut]
            # cluster_intervals_sorted = np.array(detector.cluster_intervals)[sort_permut]
            df_cluster_centers = pd.DataFrame({"centers": cluster_center}).reset_index().rename(columns={"index": "cluster"})
            df_cluster_centers.sort_values(by="centers", ascending=True, inplace=True)
            
            # print("cluster centers", cluster_center_sorted)
            # print("boundary clusters", (cluster_center_sorted[1:] + cluster_center_sorted[:-1]) / 2)
            # print("cluster intervals", cluster_intervals_sorted)
            
            df_cluster_centers.to_csv(os.path.join(experiment_folder, "cluster_centers.csv"), index=False)


    evaluator_train = DetectorEvaluator(model, train_loader, device, return_embs=return_embs)
    fpr_train, tpr_train, thr_train, train_detector_preds, train_detector_labels, train_clusters, train_embs = evaluator_train.evaluate(detector)

    df_train_detection = pd.DataFrame({
        "embs": train_embs,
        'detector_preds' : train_detector_preds,
        "detector_labels": train_detector_labels,
        "clusters": train_clusters,
        })

    df_train_detection.to_csv(os.path.join(experiment_folder, "detector_train_predictions.csv"), index=False)

    print("--- Train Results ----")
    print("FPR at TPR=0.95:", fpr_train)
    print("TPR at TPR=0.95:", tpr_train)
    print("Threshold at TPR=0.95:", thr_train)
    
    evaluator_val = DetectorEvaluator(model, val_loader, device, return_embs=return_embs)
    fpr_val, tpr_val, thr_val, val_detector_preds, val_detector_labels, val_clusters, val_embs = evaluator_val.evaluate(detector)

    df_val_detection = pd.DataFrame({
        "embs": val_embs,
        "detector_preds":  val_detector_preds,
        "detector_labels": val_detector_labels,
        "clusters": val_clusters,
        })

    df_val_detection.to_csv(os.path.join(experiment_folder, "detector_val_predictions.csv"), index=False)
    
    print("--- Validation Results ----")
    print("FPR at TPR=0.95:", fpr_val)
    print("TPR at TPR=0.95:", tpr_val)
    print("Threshold at TPR=0.95:", thr_val)

    results = [{
        "fpr_train": fpr_train, "tpr_train": tpr_train, "thr_train": thr_train,
        "fpr_val": fpr_val, "tpr_val": tpr_val, "thr_val": thr_val,
        # "inertia": detector.inertia,
        }]
    # print(detector.inertia)
    results_df = pd.DataFrame(results)
    # Save the results to a CSV file
    results_df.to_csv(os.path.join(experiment_folder, "detector_results.csv"), index=False)


if __name__ == "__main__":

    
    config = {
        "seed" : 1,
        "data" : {
            "name" : "cifar10", # gaussian_mixture
            # "n_samples" : 10000,
            "seed" : 1,
            "seed_split" : 1,
            "r" : 2,
            # "n_samples_train" : 5000,
            # "n_samples_test" : 5000,
            "batch_size_train" : 100000,
            "batch_size_test" : 100000,
            # "seed_train" : 3,
            # "seed_test" : -5
                  },
        "model" : {"name" : "resnet34", # mlp_synth_dim-10_classes-7
                   "model_seed" : None,
                   },
        "method_name" : "metric_learning",
        "metric_learning" : {"lambda" : 0.5, "temperature" : 1},                  
        "clustering" : {
            "name" : "kmeans",
            "n_clusters" : 50,
            "seed" : 0,
            "alpha" : 0.05,
            "init_scheme" : "k-means++",
            "space" : "linear", # "feature" or "classifier"
            # "cov_type" : "spherical"
            "temperature" : 1,
            # "normalize_gini" : False
            }
        }
    return_embs = False

    # for seed_split in range(1, 11):

    #     model_seed = seed_split
    #     config["data"]["seed_split"] = seed_split
    #     config["model"]["model_seed"] = model_seed
    #     print("Running with seed_split:", config["data"]["seed_split"],
    #         "and model_seed:", config["model"]["model_seed"])

    #     # config["data"]["seed_split"] = seed_split

    experiment_folder, experiment_number = create_experiment_folder(config, results_dir="synth_results/resnet3072")
    main(config)
