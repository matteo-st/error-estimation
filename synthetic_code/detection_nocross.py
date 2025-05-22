import os
import json
import torch
import random
import pickle
from torch.utils.data import  DataLoader, Subset
from torch.distributions import MultivariateNormal, Categorical
from torchvision.models.feature_extraction import create_feature_extractor
from synthetic_code.utils.models import BayesClassifier, MLPClassifier, get_model
from synthetic_code.utils.detector import MetricLearningLagrange, BayesDetector, GiniDetector
from synthetic_code.utils.datasets import get_dataset, get_synthetic_dataset
# from ..models import get_model
import numpy as np
from tqdm import tqdm
from sklearn.metrics import auc, roc_curve
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from time import localtime, strftime
import pandas as pd
import joblib

DATA_DIR = os.environ.get("DATA_DIR", "./data")
CHECKPOINTS_DIR_BASE = os.environ.get("CHECKPOINTS_DIR", "checkpoints/")

def create_experiment_folder(config, results_dir = "results"):
    """
    Create a folder for the experiment results.
    
    Args:
        config (dict): Configuration dictionary containing parameters for the experiment.
        
    Returns:
        str: Path to the created experiment folder.
    """
    results_dir = os.path.join(results_dir, config["data"]["name"])
    os.makedirs(results_dir, exist_ok=True)
    # Create a unique folder name based on the current date and time
    timestamp = strftime("%Y-%m-%d_%H-%M-%S", localtime())
    config["experiment"] = {}
    config["experiment"]["datetime"] = timestamp

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


def gini(logits, temperature=1.0, normalize=False):
    g =torch.sum(torch.softmax(logits / temperature, dim=1) ** 2, dim=1, keepdim=True)
    if normalize:
        return  (1 - g) / g 
    else:
        return 1 - g


class PartitionDetector:
    def __init__(
            self, model, weights=None, means=None, stds=None,
            n_cluster=100, alpha=0.05, method="uniform", device=torch.device('cpu'),
            n_classes=7, kmeans_seed=0, init_scheme="k-means++", # "random" or "k-means++",
            partionning_space="true_proba_error", temperature=1.0, cov_type = None
            ):
        """
        Args:
            classifier (nn.Module): A PyTorch model that takes an input tensor of shape [1, dim] and returns (logits, probs).
            weights (torch.Tensor): Tensor of shape [n_classes] (e.g., [7]).
            means (torch.Tensor): Tensor of shape [n_classes, dim] (e.g., [7, 10]).
            stds (torch.Tensor): Tensor of shape [n_classes, dim] (e.g., [7, 10]).
            n_cluster (int): Number of clusters to partition the error probability into.
            alpha (float): Confidence level parameter for interval widths.
            method (str): The method to compute the cluster. (Currently only "uniform" is supported.)
            seed (int): Random seed for data generation.
            device (torch.device): Device on which to run the classifier.
        """
        self.n_cluster = n_cluster
        self.model = model.to(device)
        self.alpha = alpha
        self.weights = weights      # torch.Tensor, shape: [n_classes]
        self.means = means          # torch.Tensor, shape: [n_classes, dim]
        self.stds = stds            # torch.Tensor, shape: [n_classes, dim]
        self.covs = torch.diag_embed(stds ** 2)
        self.method = method
        self.device = device
        self.n_classes = n_classes
        self.kmeans_seed = kmeans_seed
        self.init_scheme = init_scheme
        self.partionning_space = partionning_space
        self.cov_type = cov_type
        self.temperature = temperature

        # Initilize the density function
        self.pdfs = [MultivariateNormal(loc=self.means[i], covariance_matrix=self.covs[i]) for i in range(self.n_classes)]

        # Statistics to be computed in fit():
        self.cluster_counts = None
        self.cluster_error_means = None
        self.cluster_error_vars = None
        self.cluster_intervals = None

        # Initialize the clustering algorithm
        if self.method == "uniform":
            self.clustering_algo = None
        elif self.method == "kmeans":
            self.clustering_algo = KMeans(n_clusters=self.n_cluster, 
                                          random_state=self.kmeans_seed, 
                                          init=self.init_scheme, verbose=0)
        elif self.method == "soft-kmeans":
            self.clustering_algo = GaussianMixture(n_components=self.n_cluster, 
                                                   random_state=self.kmeans_seed, 
                                                   covariance_type=self.cov_type, 
                                                   init_params=self.init_scheme)
            
        
        # Initialize Feature extractor
        if self.partionning_space == "gini":
            self.feature_extractor = lambda x: gini(self.model(x)[0], temperature=self.temperature)
        elif self.partionning_space == "true_proba_error":
            self.feature_extractor = lambda x: self.func_proba_error(x)
        else: 
            self.extractor = create_feature_extractor(self.model, {partionning_space : partionning_space})
            self.feature_extractor = lambda x: self.extractor(x)[partionning_space]
    
    def func_proba_error(self, x):
        """
        Compute an error probability for a given input x.
        A simple proxy is: error probability = 1 - max(predicted probability)
        
        Args:
            x (np.ndarray): A 1D NumPy array of shape [dim].
            classifier (nn.Module): The classifier.
            device (torch.device): Device to use.
            
        Returns:
            proba_error (float): The error probability.
        """
        with torch.no_grad():
            # classifier returns (logits, probs)
            _, model_probs = self.model(x)
            model_pred = torch.argmax(model_probs, dim=1, keepdim=True)
        data_probs = [self.weights[i] *  torch.exp(self.pdfs[i].log_prob(x)) for i in range(self.n_classes)]
 
        data_probs = torch.stack(data_probs, dim=1) # [batch_size, n_classes]
        pred_prob = data_probs.gather(1, model_pred) # [batch_size, 1]
        # Normalize the probabilities
        return 1 - pred_prob / data_probs.sum(dim=1, keepdim=True)  # [batch_size, n_classes]


    def predict_clusters(self, x):
       
        if self.method == "uniform":
            embs = self.feature_extractor(x)
            cluster = torch.floor(embs * self.n_cluster).long()
            # Handle edge case when proba_error == 1
            cluster[cluster == self.n_cluster] = self.n_cluster - 1
            return cluster
        elif self.method in ["kmeans", "soft-kmeans"]:
            embs = self.feature_extractor(x)
            cluster = torch.tensor(self.clustering_algo.predict(embs.cpu().numpy()), 
                                   device=self.device)
            return cluster
        else:
            raise ValueError("Unsupported method")

    def fit(self, train_dataloader):
         
        self.model.eval()

        all_model_preds = []
        all_detector_labels = []
        all_clusters = []
        all_embs = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(train_dataloader, total=len(train_dataloader), desc="Training Detector"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                logits, _ = self.model(inputs)  # logits: [batch_size, num_classes]
                model_preds = torch.argmax(logits, dim=1)  # [batch_size]

                detector_labels = (model_preds != targets).float()

                if self.method == "uniform":
                    clusters = self.predict_clusters(inputs)
                    all_clusters.append(clusters)
                elif self.method in ["kmeans", "soft-kmeans"]:
                    embs = self.feature_extractor(inputs)
                    all_embs.append(embs)

                all_model_preds.append(model_preds)
                all_detector_labels.append(detector_labels)
        
        
        if self.method == "uniform":
            clusters = torch.cat(all_clusters, dim=0)
        elif self.method == "kmeans":
            all_embs = torch.cat(all_embs, dim=0)
            # self.all_embs = all_embs.cpu().numpy().squeeze(-1)
            clusters = self.clustering_algo.fit_predict(all_embs.cpu().numpy())
            clusters = torch.tensor(clusters, device=self.device)
            self.inertia = self.clustering_algo.inertia_
        elif self.method == "soft-kmeans":
            all_embs = torch.cat(all_embs, dim=0)
            clusters = self.clustering_algo.fit_predict(all_embs.cpu().numpy())
            clusters = torch.tensor(clusters, device=self.device)
            self.inertia = self.clustering_algo.lower_bound_
            
        
        detector_labels = torch.cat(all_detector_labels, dim=0)
        
        
        # Initialize lists to store per-cluster statistics.
        self.cluster_counts = []
        self.cluster_error_means = []
        self.cluster_error_vars = []
        self.cluster_intervals = []
        
        # For each cluster, compute the sample mean and variance of the error indicator.
        for i in range(self.n_cluster):
            idx = (clusters == i).nonzero(as_tuple=True)[0]
            count = idx.numel()
            self.cluster_counts.append(count)

            if count > 0:
                cluster_detector_labels = detector_labels[idx]

                error_mean = cluster_detector_labels.mean().item()
                error_vars = cluster_detector_labels.var(unbiased=False).item()

                self.cluster_error_means.append(error_mean)
                self.cluster_error_vars.append(error_vars)

                # Confidence interval half-width using a Hoeffding-type bound.
                half_width = torch.sqrt(torch.log(torch.tensor(2 / self.alpha, device=self.device)) / (2 * count)).item()
                lower_bound = max(0.0, error_mean - half_width)
                upper_bound = min(1.0, error_mean + half_width)
                self.cluster_intervals.append((lower_bound, upper_bound))
    
            else:
                self.cluster_error_means.append(0.0)
                self.cluster_error_vars.append(0.0)
                self.cluster_intervals.append((0, 1))

    def __call__(self, x, save_embs=True):
        """
        Given an input x (as a NumPy array of shape [dim]), 
        returns the upper bound of the estimated error interval for the cluster into which x falls.
        
        Args:
            x (np.ndarray): Input sample, shape [dim].
            
        Returns:
            upper_bound (float): The upper bound of the error confidence interval.
        """
        cluster = self.predict_clusters(x)
        all_upper_bounds = torch.tensor([ub for (_, ub) in self.cluster_intervals],
                                         dtype=torch.float32,
                                         device=self.device)
        detector_preds = all_upper_bounds[cluster]
        # if save_embs:
        #     # Save the embeddings for further analysis
        #     self.embs = self.feature_extractor(x)
        #     self.clusters = cluster
        return detector_preds
    

class DetectorEvaluator:
    def __init__(self, model, dataloader, device, return_embs=False):
        """
        Evaluator for measuring model accuracy.
        
        Args:
            model (nn.Module): The trained classifier.
            dataloader (DataLoader): DataLoader for evaluation data.
            device (torch.device): Device to perform evaluation on.
        """
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.return_embs = return_embs

    def fpr_at_fixed_tpr(self, fprs, tprs, thresholds, tpr_level: float = 0.95):
        
        idxs = [i for i, x in enumerate(tprs) if x >= tpr_level]
        if len(idxs) > 0:
            idx = min(idxs)
        else:
            idx = 0
        return fprs[idx], tprs[idx], thresholds[idx]

    def evaluate(self, detector, return_clusters=False):

        self.model.eval()

        all_clusters = []
        all_embs = []
        all_detector_preds = []
        all_detector_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(self.dataloader, desc="Evaluating Detector"):
                # x: [batch_size, dim], labels: [batch_size]
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(inputs)  # logits: [batch_size, num_classes]
                # logits, _ = self.model(inputs)  # logits: [batch_size, num_classes]
                model_preds = torch.argmax(logits, dim=1)  # [batch_size]

                detector_labels = model_preds != labels
                if return_clusters:
                    clusters = detector.predict_clusters(inputs)
                    embs = detector.feature_extractor(inputs).squeeze(-1)
                else:
                    clusters = [np.nan] * inputs.shape[0]
                    embs = [np.nan] * inputs.shape[0]
                    
                detector_preds = detector(inputs)

                # all_model_preds.append(model_preds.cpu().numpy())
                # all_clusters.append(clusters.cpu().numpy())
                # all_embs.append(embs.cpu().numpy())
                all_detector_labels.append(detector_labels.cpu().numpy())
                all_detector_preds.append(detector_preds.cpu().numpy())

        all_detector_preds = np.concatenate(all_detector_preds, axis=0)
        all_detector_labels = np.concatenate(all_detector_labels, axis=0)
        # all_clusters = np.concatenate(all_clusters, axis=0)
        # all_embs = np.concatenate(all_embs, axis=0)

        fprs, tprs, thrs = roc_curve(all_detector_labels, all_detector_preds)
        # Compute the area under the ROC curve
        fpr, tpr, thr = self.fpr_at_fixed_tpr(fprs, tprs, thrs, 0.95)
        if self.return_embs:
            return fpr, tpr, thr, all_detector_preds, all_detector_labels, all_clusters, all_embs
        else:
            return fpr, tpr, thr, all_detector_preds, all_detector_labels, [None] * len(all_detector_labels), [None] * len(all_detector_labels)
        

def main(config):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    model = get_model(model_name=config["model"]["name"], 
                    dataset_name=config["data"]["name"],
                    model_seed=config["model"]["model_seed"],
                    checkpoint_dir = os.path.join(CHECKPOINTS_DIR_BASE, "ce")
                    )
    model = model.to(device)
    model.eval()

    if config["data"]["name"] == "gaussian_mixture":

        train_dataset, test_dataset = get_synthetic_dataset(
            model_name=config["model"]["name"],
            n_samples_train = config["data"]["n_samples_train"],
            n_samples_test = config["data"]["n_samples_test"],
            seed_train = config["data"]["seed_train"],
            seed_test = config["data"]["seed_test"],
            device = device
            )
    else:
        dataset = get_dataset(
            dataset_name=config["data"]["name"], 
            model_name=config["model"]["name"], 
            root=DATA_DIR,  
            shuffle=True, 
            random_state=seed)

    print("Dataset size", len(dataset))



    # Create DataLoaders.
    train_loader = DataLoader(train_dataset, batch_size=config["data"]["batch_size_train"], shuffle=False, num_workers=4)  # Each batch: [32, 10]
    val_loader = DataLoader(test_dataset, batch_size=config["data"]["batch_size_test"], shuffle=False, num_workers=4)       # Each batch: [32, 10]


    if config["method_name"] == "clustering":
        detector = PartitionDetector(model, weights, means, stds, 
                                    n_cluster=config["clustering"]["n_clusters"], 
                                    alpha=config["clustering"]["alpha"], 
                                    method=config["clustering"]["name"], 
                                    device=device, 
                                    kmeans_seed=config["clustering"]["seed"], 
                                    #  init_scheme=config["k-means++"],
                                    #  temperature=config["temperature"],
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

    if (config["method_name"] == "clustering") & (config["clustering"]["name"] == "soft-kmeans"):
        # with open(os.path.join(experiment_folder, 'clustering_algo.pkl'), "wb") as f:
        #     pickle.dump(detector.clustering_algo, f)
        joblib.dump(detector.clustering_algo, os.path.join(experiment_folder, 'clustering_algo.pkl'))
    # Save the clustering algorithm

    if (config["method_name"] == "clustering") & (config["clustering"]["name"] == "kmeans"):
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
    seed = 1
    config = {
        "data" : {"name": "gaussian_mixture", #gaussian_mixture,
                  "num_crossval" : 1,
                  "n_samples_train" : None,
                  "n_samples_test" : None,
                  "batch_size_train" : 100000,
                  "batch_size_test" : 100000,
                  "seed_train" : None,
                  "seed_test" : None
                  },
        "model" : {"name" : "mlp_synth_dim-10_classes-7",
                   "model_seed" : None,
                   },
        "method_name" : "metric_learning",
        "metric_learning" : {"lambda" : 0.5, "temperature" : 1},                  
        "clustering" : {
            "name" : "kmeans",
            "n_clusters" : 10,
            "seed" : 0,
            "alpha" : 0.05,
            "init_scheme" : "k-means++",
            "space" : "softmax",
            # "cov_type" : "spherical"
            # "temperature" : 1,
            # "normalize_gini" : False
            }
        }
    
    return_embs = False
    experiment_folder, experiment_number = create_experiment_folder(config)
    main(config)
