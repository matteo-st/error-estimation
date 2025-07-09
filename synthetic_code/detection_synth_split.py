import os
import json
import torch
from torch.utils.data import  DataLoader, Subset
from torch.distributions import MultivariateNormal, Categorical
from synthetic_code.utils.models import BayesClassifier, MLPClassifier, get_model
from synthetic_code.utils.detector import PartitionDetector, MetricLearningLagrange, BayesDetector, GiniDetector, MaxProbaDetector
from synthetic_code.utils.datasets import GaussianMixtureDataset, get_dataset, get_synthetic_dataset
from synthetic_code.utils.eval import DetectorEvaluator
import numpy as np
from tqdm import tqdm
from time import localtime, strftime
import pandas as pd
import joblib
import random
import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*force_all_finite.*"
)


CHECKPOINTS_DIR_BASE = os.environ.get("CHECKPOINTS_DIR", "checkpoints/")
DATA_DIR = os.environ.get("DATA_DIR", "./data")


def append_results_to_file(config, results, result_file):
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    timestamp = strftime("%Y-%m-%d_%H-%M-%S", localtime())
    config["experiment"] = {}
    config["experiment"]["datetime"] = timestamp

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
            config[m] = dict.fromkeys(subconf.keys(), None)
        else:
            # nothing to reset (either missing or not a dict)
            # Optionally, you could initialize it:
            # config[m] = {}
            pass
    if config["reduction"]["name"] is None:
        config["reduction"] = dict.fromkeys(config["reduction"].keys(), None)
    


    config = pd.json_normalize(config, sep="_")
    config["experiment_folder"] = "bwe"
    results = pd.concat([config, results], axis=1)
    print(results)
    print(f"Saving results to {result_file}")
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
    # Create a unique folder name based on the current date and time
    timestamp = strftime("%Y-%m-%d_%H-%M-%S", localtime())
    config["experiment"] = {}
    config["experiment"]["datetime"] = timestamp

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
            config[m] = dict.fromkeys(subconf.keys(), None)
        else:
            # nothing to reset (either missing or not a dict)
            # Optionally, you could initialize it:
            # config[m] = {}
            pass
    if config["reduction"]["name"] is None:
        config["reduction"] = dict.fromkeys(config["reduction"].keys(), None)


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
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    # torch.use_deterministic_algorithms(True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    # for name, module in model.named_modules():
    #     print(name, "→", module)
    # print("model keys", model.state_dict().keys())
    # exit()
    # if config["data"]["name"] == "gaussian_mixture":

    #     dataset = get_synthetic_dataset(
    #         model_name=config["model"]["name"],
    #         n_samples = config["data"]["n_samples"],
    #         input_dim= config["data"]["dim"],
    #         n_classes = config["data"]["n_classes"],
    #         seed = config["data"]["seed"],
    #         device = device
    #         )
    # else:
    #     dataset = get_dataset(
    #         dataset_name=config["data"]["name"], 
    #         model_name=config["model"]["name"], 
    #         root=DATA_DIR,  
    #         shuffle=False)


                                       
    n = len(dataset)
    print("Dataset size", n)
    # print("model first sample", model(train_dataset[0][0].unsqueeze(0).to(device)))
    # # reproducible permutation
    # gen = torch.Generator().manual_seed(config["data"]["seed_split"])
    # perm = torch.randperm(n, generator=gen).tolist()
    perm = list(range(len(dataset)))
    if config["data"]["seed_split"] is not None:
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
    print("Train dataset size:", len(train_dataset)
          , "Val dataset size:", len(val_dataset), "Test dataset size:", len(test_dataset))


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
        print('config["clustering"]["distance"]', config["clustering"]["distance"])
        detector = PartitionDetector(model, 
                                    #  dataset.weights, dataset.means, dataset.stds, 
                                    n_cluster=config["clustering"]["n_clusters"], 
                                    alpha=config["clustering"]["alpha"], 
                                    method=config["clustering"]["name"], 
                                    device=device, 
                                    n_classes= config["data"]["n_classes"],
                                    kmeans_seed=config["clustering"]["seed"], 
                                     init_scheme=config["clustering"]["init_scheme"],
                                     n_init=config["clustering"]["n_init"],
                                    temperature=config["clustering"]["temperature"],
                                    partionning_space=config["clustering"]["space"],
                                     cov_type=config["clustering"]["cov_type"],
                                     reducer=config["reduction"]["name"],
                                     reduction_dim=config["reduction"]["reduction_dim"],
                                     n_neighbors=config["reduction"]["n_neighbors"],
                                     reducer_seed=config["reduction"]["seed"],
                                     normalize_gini=config["clustering"]["normalize_gini"], 
                                     divergence=config["clustering"]["distance"]
                                    )

        detector.fit(train_loader)

        # np.savez_compressed(
        #     os.path.join(experiment_folder, "cluster_results.npz"),
        #         cluster_counts=detector.cluster_counts,
        #         cluster_error_means=detector.cluster_error_means,
        #         cluster_error_vars=detector.cluster_error_vars,
        #         cluster_intervals=detector.cluster_intervals
        #     )
        
    elif config["method_name"] == "metric_learning":
        detector = MetricLearningLagrange(model, 
                                            lbd=config["metric_learning"]["lambda"], 
                                            temperature=config["metric_learning"]["temperature"]
                                            )
        detector.fit(train_loader)
        
    elif config["method_name"] == "gini":
        detector = GiniDetector(model, 
                                temperature=config["gini"]["temperature"], 
                                normalize=config["gini"]["normalize_gini"], 
                                device=device)
    elif config["method_name"] == "max_proba":
        detector = MaxProbaDetector(model, 
                                temperature=config["max_proba"]["temperature"], 
                                device=device)
    elif config["method_name"] == "bayes":
        param_path = os.path.join(CHECKPOINTS_DIR_BASE, "ce", 
                                  f"resnet34_synth_dim-{config['data']['dim']}_classes-{config['data']['n_classes']}",
                                  "data_parameters.npz")
        params = np.load(param_path)
        means   = params["means"]    # [n_classes, dim]
        covs    = params["covs"]     # [n_classes, dim, dim]
        weights = params["weights"]  # [n_classes]

        # --- Move to GPU & factorize covariances ---

        means   = torch.from_numpy(means).float().to(device)      # [n_classes, dim]
        covs    = torch.from_numpy(covs).float().to(device)       # [n_classes, dim, dim]
        weights = torch.from_numpy(weights).float().to(device)    # [n_classes]
        detector = BayesDetector(model, 
                                weights, 
                                means, 
                                covs, 
                                config["data"]["n_classes"], 
                                device=device)
        
    

    # print("converged ?", detector.clustering_algo.converged_)
   
    # if (config["method_name"] == "clustering"):
    #     if (config["clustering"]["name"] == "soft-kmeans"):
    #         # with open(os.path.join(experiment_folder, 'clustering_algo.pkl'), "wb") as f:
    #         #     pickle.dump(detector.clustering_algo, f)
    #         joblib.dump(detector.clustering_algo, os.path.join(experiment_folder, 'clustering_algo.pkl'))
        # Save the clustering algorithm

    # if (config["method_name"] == "clustering"):
    #     if config["clustering"]["name"] == "kmeans":

    #         cluster_center =  detector.clustering_algo.cluster_centers_
    #         np.savez_compressed(
    #             os.path.join(experiment_folder, "cluster_centers"), 
    #             centers=cluster_center)
            
    #         joblib.dump(detector.clustering_algo, os.path.join(experiment_folder, 'clustering_algo.pkl'))

            # sort_permut = np.argsort(cluster_center)
            # print("sort_permut", sort_permut)

            # cluster_center_sorted = cluster_center[sort_permut]
            # cluster_intervals_sorted = np.array(detector.cluster_intervals)[sort_permut]
            # df_cluster_centers = pd.DataFrame({"centers": cluster_center}).reset_index().rename(columns={"index": "cluster"})
            # df_cluster_centers.sort_values(by="centers", ascending=True, inplace=True)
            
            # print("cluster centers", cluster_center_sorted)
            # print("boundary clusters", (cluster_center_sorted[1:] + cluster_center_sorted[:-1]) / 2)
            # print("cluster intervals", cluster_intervals_sorted)


            # df_cluster_centers.to_csv(os.path.join(experiment_folder, "cluster_centers.csv"), index=False)


    evaluator_train = DetectorEvaluator(model, train_loader, device, return_embs=return_embs, return_labels=return_labels)
    fpr_train, tpr_train, thr_train, train_detector_preds, train_detector_labels, train_clusters, train_embs, train_labels = evaluator_train.evaluate(detector, return_clusters=return_clusters)


    # np.savez_compressed(
    #     os.path.join(experiment_folder, "detector_train_predictions.npz"),
    #     embs=train_embs,
    #     detector_preds=train_detector_preds,
    #     detector_labels=train_detector_labels,
    #     clusters=train_clusters,
    #     labels=train_labels,
    # )

    # df_train_detection.to_csv(os.path.join(experiment_folder, "detector_train_predictions.csv"), index=False)

    # print("--- Train Results ----")
    # print("FPR at TPR=0.95:", fpr_train)
    # print("TPR at TPR=0.95:", tpr_train)
    # print("Threshold at TPR=0.95:", thr_train)
    
    evaluator_val = DetectorEvaluator(model, val_loader, device, return_embs=return_embs, return_labels=return_labels)
    fpr_val, tpr_val, thr_val, val_detector_preds, val_detector_labels, val_clusters, val_embs, val_labels = evaluator_val.evaluate(detector, return_clusters=return_clusters)

    # df_val_detection = pd.DataFrame({
    #     "embs": val_embs,
    #     "detector_preds":  val_detector_preds,
    #     "detector_labels": val_detector_labels,
    #     "clusters": val_clusters,
    #     })
    # np.savez_compressed(
    #     os.path.join(experiment_folder, "detector_val_predictions.npz"),
    #     embs=val_embs,
    #     detector_preds=val_detector_preds,
    #     detector_labels=val_detector_labels,
    #     clusters=val_clusters,
    #     labels=val_labels,
    # )

    # df_val_detection.to_csv(os.path.join(experiment_folder, "detector_val_predictions.csv"), index=False)
    
    # print("--- Validation Results ----")
    # print("FPR at TPR=0.95:", fpr_val)
    # print("TPR at TPR=0.95:", tpr_val)
    # print("Threshold at TPR=0.95:", thr_val)

    results = [{
        "fpr_train": fpr_train, "tpr_train": tpr_train, "thr_train": thr_train,
        "fpr_val": fpr_val, "tpr_val": tpr_val, "thr_val": thr_val,
        # "inertia": detector.inertia,
        # "n_iter": detector.clustering_algo.n_iter_,
        }]
    # print(detector.inertia)
    results = pd.DataFrame(results)
    # Save the results to a CSV file

    # results.to_csv(os.path.join(experiment_folder, "detector_results.csv"), index=False)
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
            "seed" : 2,
            "seed_split" : 2,
            "r" : 2,
            # "n_samples_train" : 5000,
            # "n_samples_test" : 5000,
            "batch_size_train" : 256,
            "batch_size_test" : 256,
            # "seed_train" : 3,
            # "seed_test" : -5,
            "class_subset" : None
                  },
        "model" : {"name" : "resnet34", # mlp_synth_dim-10_classes-7
                   "model_seed" : 1,
                   },
        "method_name" : "clustering",
        "metric_learning" : {"lambda" : 0.5, "temperature" : 1}, 
        "gini" : {
            "temperature" : 2,
            "normalize_gini" : False
            },        
        "max_proba" : {
            "temperature" : 1,
            },             
        "clustering" : {
            "name" : "bregman-hard", # "kmeans", "soft-kmeans", "bregman-hard"
            "distance" : "Itakura_Saito", # "euclidean", "kl", "js", "alpha-divergence"
            "n_clusters" : 20,
            "seed" : 0,
            "alpha" : 0.05,
            "init_scheme" : "random",
            "n_init" : 5,
            "space" : "probits", # "feature" or "classifier"
            "cov_type" : "diag",
            "temperature" : 0.25,
            "normalize_gini" : False
            },
        "reduction" : {
            "name" : None, # umap
            "reduction_dim" : 10,
            "n_neighbors" : 15,
            "seed" : 0,
            }
            }
    return_embs = False
    return_clusters = False
    return_labels = False
    download_data = True
    
    results_file = "synth_results/cifar10/all_results.csv"
    # results_dir = "synth_results/cifar10"

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
            
        # dataset = get_synthetic_dataset(
        #     dim= config["data"]["dim"],
        #     n_samples= config["data"]["n_samples"],
        #     n_classes = config["data"]["n_classes"],
        #     seed = config["data"]["seed"],
        #     )
    # [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]:
    # for divergence in [ "KL",]:
    for n_cluster in range(26, 100):
        for temperature in [0.1 + 0.025 * i for i in range(40)]:
            for seed_split in range(1, 10):
                config["data"]["seed_split"] = seed_split
                config["clustering"]["temperature"] = temperature
                config["clustering"]["n_clusters"] = n_cluster
                # config["clustering"]["distance"] = divergence
                print("Running with seed_split:", config["data"]["seed_split"])
                print("Running with temperature:", temperature)
                print("Running with n_cluster:", n_cluster)
               

                # Create a folder for the experiment results
                # experiment_folder, experiment_number = create_experiment_folder(config, results_dir=results_dir)
                main(config)

    # for temperature in [100, 200, 500, 700]:
    #     for seed_split in range(1, 10):
    #         config["data"]["seed_split"] = seed_split
    #         config["gini"]["temperature"] = temperature
    #         # config["clustering"]["n_clusters"] = n_cluster
    #         print("Running with temperature:", temperature, "| seed_split:", config["data"]["seed_split"])
 
    #         # Create a folder for the experiment results
    #         experiment_folder, experiment_number = create_experiment_folder(config, results_dir=results_dir)
    #         main(config)

    # for clustering_name in ["kmeans", "soft-kmeans"]:
    #     for n_cluster in [200, 500]:
    #         # for reduction_dim in [10, 50, 100]:
    #         #     for n_neighbors in [5, 10, 15, 20]:
    #                 for seed_split in range(1, 5):
    #                     config["clustering_name"] = clustering_name
    #                     config["clustering"]["n_clusters"] = n_cluster
    #                     # config["reduction"]["reduction_dim"] = reduction_dim
    #                     # config["reduction"]["n_neighbors"] = n_neighbors
    #                     config["data"]["seed_split"] = seed_split
    #                     print("Running with n_cluster:", n_cluster, 
    #                         "| clustering_name:", clustering_name,
    #                         # "| reduction_dim:", reduction_dim, 
    #                         # "| n_neighbors:", n_neighbors,
    #                         "| seed_split:", config["data"]["seed_split"])
                        # experiment_folder, experiment_number = create_experiment_folder(config, results_dir="synth_results/resnet3072_kmeans_reduced_pooling")
                        # main(config)


    
    # pr.disable()
    # ps = pstats.Stats(pr).sort_stats('cumtime')
    # ps.print_stats(20)   # top 20 entries
    # for n_cluster in [200, 500]:
    #     for seed_split in range(1, 11):
          

    #     #     model_seed = seed_split
    #         print("Running with n_cluster:", n_cluster, "| seed_split:", seed_split,)
    #         config["clustering"]["n_clusters"] = n_cluster
        
    #         config["data"]["seed_split"] = seed_split
    #     #     config["model"]["model_seed"] = model_seed
    #     #     print("Running with seed_split:", config["data"]["seed_split"],
    #     #         "and model_seed:", config["model"]["model_seed"])

    #     #     # config["data"]["seed_split"] = seed_split

    #         experiment_folder, experiment_number = create_experiment_folder(config, results_dir="synth_results/test_speed")
    #         main(config)
