import torch
import numpy as np
from torch.distributions import MultivariateNormal
from tqdm import tqdm
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from torchvision.models.feature_extraction import create_feature_extractor
import umap
import os
import joblib
from sklearn.decomposition import PCA
from code.utils.clustering.models import BregmanHard
from code.utils.clustering.divergences import (
    euclidean,
    kullback_leibler,
    itakura_saito,
    alpha_divergence_factory,
)

from code.utils.detection.registry import register_detector

def gini(logits, temperature=1.0, normalize=False):
    g =torch.sum(torch.softmax(logits / temperature, dim=1) ** 2, dim=1, keepdim=True)
    if normalize:
        return  (1 - g) / g 
    else:
        return 1 - g



class MultiDetectors:
    def __init__(self, detectors, model, device):

        """
        Args:
            detectors (list): List of detector instances.
        """
        self.detectors = detectors
        self.model = model
        self.device = device
        

    def fit(self, train_dataloader):
        """
        Fit all detectors on the training data.
        
        Args:
            train_dataloader (DataLoader): DataLoader for the training data.
        """
        self.model.eval()

        # all_model_preds = []
        all_detector_labels = []
        all_logits = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(train_dataloader, total=len(train_dataloader), desc="Getting Training Logits", disable=False):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
            
                logits = self.model(inputs).cpu()  # logits: [batch_size, num_classes]
                model_preds = torch.argmax(logits, dim=1)

                detector_labels = (model_preds != targets.cpu()).float()
                # all_model_preds.append(model_preds)
                all_detector_labels.append(detector_labels)
                all_logits.append(logits)
        
        
        # all_model_preds = torch.cat(all_model_preds, dim=0)
        all_detector_labels = torch.cat(all_detector_labels, dim=0)
        all_logits = torch.cat(all_logits, dim=0)

        for dec in tqdm(self.detectors,total=len(self.detectors), desc="Fitting Detectors", disable=False):
            dec.fit(logits=all_logits.to(dec.device), detector_labels=all_detector_labels.to(dec.device))



@register_detector("clustering")
class PartitionDetector:
    def __init__(
            self, model, 
            n_clusters=100, alpha=0.05, name="uniform",
            n_classes=7, seed=0, init_scheme="k-means++", # "random" or "k-means++", 
            n_init=1, # Number of initializations for k-means
            space="true_proba_error", temperature=1.0, cov_type = None,
            reduction_name=None, # For dimensionality reduction
            reduction_dim=2, reduction_n_neighbors=15, reduction_seed=0, # For UMAP
            normalize_gini=False, # Whether to normalize the Gini coefficient
            distance=None, # For BregmanHard clustering
            reorder_embs=False, # Whether to reorder the embeddings based on the clustering
            experiment_folder=None,
            class_subset=None,
            params_path=None,
            device=torch.device('cpu')
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
        self.n_cluster = n_clusters
        # self.model = model.to(device)
        self.model = model.to(device)
        self.alpha = alpha
        # self.weights = weights      # torch.Tensor, shape: [n_classes]
        # self.means = means          # torch.Tensor, shape: [n_classes, dim]
        # self.stds = stds            # torch.Tensor, shape: [n_classes, dim]
        self.params_path = params_path

        self.method = name
        self.device = device
        self.n_classes = n_classes
        self.reorder_embs = reorder_embs
        self.kmeans_seed = seed
        self.init_scheme = init_scheme
        self.n_init = n_init
        self.partionning_space = space
        self.cov_type = cov_type
        self.temperature = temperature
        self.divergence = distance

        self.normalize_gini = normalize_gini

        self.experiment_folder = experiment_folder
        if class_subset is not None:
            self.dict_class_subset = {i: orig for i, orig in enumerate(class_subset)}
        else:
            self.dict_class_subset = None

        self.reduction_name = reduction_name
        self.reducing_dim = reduction_dim
        self.n_neighbors = reduction_n_neighbors
        self.reducer_seed = reduction_seed

        if self.reduction_name == "umap":
            self.reducer = umap.UMAP(n_components=self.reducing_dim,
                                        n_neighbors= self.n_neighbors, 
                                    #  random_state=self.reducer_seed
                                        )
        elif self.reduction_name == "pca":
            self.reducer = PCA(n_components=self.reducing_dim, random_state=self.reducer_seed)
        else:
            self.reducer = None

        # Initilize the density function
        if self.partionning_space == "true_proba_error":

            if self.param_path is None:
                raise ValueError("param_path must be provided when partionning_space is 'true_proba_error'")
   
            params = np.load(self.param_path)
            means   = params["means"]    # [n_classes, dim]
            covs    = params["covs"]     # [n_classes, dim, dim]
            weights = params["weights"]  # [n_classes]

            # --- Move to GPU & factorize covariances ---

            means   = torch.from_numpy(means).float().to(device)      # [n_classes, dim]
            covs    = torch.from_numpy(covs).float().to(device)       # [n_classes, dim, dim]
            weights = torch.from_numpy(weights).float().to(device)    # [n_classes]

            self.log_weights = torch.log(weights.to(device))      # torch.Tensor, shape: [n_classes]
            self.means = means.to(device)          # torch.Tensor, shape: [n_classes, dim]
            self.covs = covs.to(device)            # torch.Tensor, shape: [n_classes, dim]

            # Initilize the density function
            self.pdfs = [MultivariateNormal(loc=self.means[i], covariance_matrix=self.covs[i]) for i in range(self.n_classes)]
        # if stds is not None:
        #     self.covs = torch.diag_embed(stds ** 2)
        #     self.pdfs = [MultivariateNormal(loc=self.means[i], covariance_matrix=self.covs[i]) for i in range(self.n_classes)]

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
                                            n_init=self.n_init,
                                          init=self.init_scheme, verbose=0)
        elif self.method == "soft-kmeans":
            self.clustering_algo = GaussianMixture(n_components=self.n_cluster, 
                                                   random_state=self.kmeans_seed, 
                                                   covariance_type=self.cov_type, 
                                                   init_params=self.init_scheme)
        elif self.method == "bregman-hard":
            divergences = {
                "Euclidean": euclidean,
                "KL": kullback_leibler,
                "Itakura_Saito": itakura_saito,
                "Alpha0.5": alpha_divergence_factory(0.5),
            }

            self.clustering_algo = BregmanHard(
                n_clusters=self.n_cluster,
                divergence=divergences[self.divergence],  # e.g., "kl", "euclidean", etc.
                n_init=self.n_init,
                initializer=self.init_scheme,
                random_state= self.kmeans_seed,
                )
        
    def _extract_embeddings(self, x=None, logits=None):
        """
        Extract embeddings from the model.
        This function is used to create a feature extractor.
        """

        if logits is not None:
            if self.partionning_space == "gini":
                embs = gini(logits, temperature=self.temperature, normalize=self.normalize_gini)
            elif self.partionning_space == "probits":
                embs = torch.softmax(logits / self.temperature, dim=1)
        else:
            logits = self.model(x)
            if self.partionning_space == "gini":
                embs = gini(logits, temperature=self.temperature, normalize=self.normalize_gini)
            elif self.partionning_space == "probits":
                embs = torch.softmax(logits / self.temperature, dim=1)
            else:
                raise ValueError("Unsupported partionning space")


        # Reorder embeddings if needed
        if self.reorder_embs:
            embs = embs.sort(dim=1, descending=True)[0]
        return embs
    
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
        x = x.to(self.device)
        batch_size = x.shape[0]  # batch_size 
        with torch.no_grad():
            logits = self.model(x)
            model_pred = torch.argmax(logits, dim=1, keepdim=True)

                # 2) compute unnormalized log-posteriors: log w_i + log p_i(x)
        x = x.view(batch_size, -1) 
        log_data_probs = torch.stack([
            self.pdfs[i].log_prob(x) + self.log_weights[i]
            for i in range(len(self.pdfs))
        ], dim=1) 

        #  3) log-denominator = logsumexp over classes
        log_den = torch.logsumexp(log_data_probs, dim=1, keepdim=True)  # [1,1]

        #  4) log-posterior per class
        log_post = log_data_probs - log_den 
        # 5) Bayes error probability = 1 − posterior_of_predicted
        log_post_pred = log_post.gather(1, model_pred)            # [1,1]
        error_proba    = 1.0 - torch.exp(log_post_pred)      # [1,1]

        # Normalize the probabilities
        return error_proba


    def predict_clusters(self, x=None, logits=None):

        embs = self._extract_embeddings(x, logits)

        if self.method == "uniform":
            cluster = torch.floor(embs * self.n_cluster).long()
            cluster[cluster == self.n_cluster] = self.n_cluster - 1 # Handle edge case when proba_error == 1
            return cluster
        
        elif self.method in ["kmeans", "soft-kmeans", "bregman-hard"]:
            if self.reducer is not None:
                embs = self.reducer.transform(embs.cpu().numpy())
                cluster = torch.tensor(self.clustering_algo.predict(embs), 
                                    device=self.device)
            else:
                cluster = torch.tensor(self.clustering_algo.predict(embs.cpu().numpy()), 
                                   device=self.device)
            return cluster
        else:
            raise ValueError("Unsupported method")

    def save_results(self, experiment_folder):

            np.savez_compressed(
                os.path.join(experiment_folder, "cluster_results.npz"),
                    cluster_counts=self.cluster_counts,
                    cluster_error_means=self.cluster_error_means,
                    cluster_error_vars=self.cluster_error_vars,
                    cluster_intervals=self.cluster_intervals
                )
            
            joblib.dump(self.clustering_algo, os.path.join(experiment_folder, 'clustering_algo.pkl'))


    def fit(self, logits, detector_labels):

   
        all_embs = self._extract_embeddings(logits=logits)
    
        if self.reducer is not None:
            # If a reducer is used, fit it on the embeddings
            all_embs = torch.tensor(self.reducer.fit_transform(all_embs.cpu().numpy()), device=self.device)
        # self.all_embs = all_embs.cpu().numpy().squeeze(-1)
        clusters = self.clustering_algo.fit_predict(all_embs.cpu().numpy())
        clusters = torch.tensor(clusters, device=self.device)

        if self.method == "kmeans":
            self.inertia = self.clustering_algo.inertia_
        elif self.method == "soft-kmeans":
            self.inertia = self.clustering_algo.lower_bound_

        # print("inertia", self.inertia)
        self.clustering(detector_labels, clusters)
        

        if self.experiment_folder is not None:
            self.save_results(self.experiment_folder)


    def clustering(self, detector_labels, clusters):
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



    def __call__(self, x=None, logits=None, save_embs=True):
        """
        Given an input x (as a NumPy array of shape [dim]), 
        returns the upper bound of the estimated error interval for the cluster into which x falls.
        
        Args:
            x (np.ndarray): Input sample, shape [dim].
            
        Returns:
            upper_bound (float): The upper bound of the error confidence interval.
        """
        cluster = self.predict_clusters(x, logits)
        all_upper_bounds = torch.tensor([ub for (_, ub) in self.cluster_intervals],
                                         dtype=torch.float32,
                                         device=self.device)
        detector_preds = all_upper_bounds[cluster]
        # if save_embs:
        #     # Save the embeddings for further analysis
        #     self.embs = self.feature_extractor(x)
        #     self.clusters = cluster
        return detector_preds

# @register_detector("clustering")
# class PartitionDetector:
#     def __init__(
#             self, model, 
#             n_clusters=100, alpha=0.05, name="uniform",
#             n_classes=7, seed=0, init_scheme="k-means++", # "random" or "k-means++", 
#             n_init=1, # Number of initializations for k-means
#             space="true_proba_error", temperature=1.0, cov_type = None,
#             reduction_name=None, # For dimensionality reduction
#             reduction_dim=2, reduction_n_neighbors=15, reduction_seed=0, # For UMAP
#             normalize_gini=False, # Whether to normalize the Gini coefficient
#             distance=None, # For BregmanHard clustering
#             reorder_embs=False, # Whether to reorder the embeddings based on the clustering
#             experiment_folder=None,
#             class_subset=None,
#             params_path=None,
#             device=torch.device('cpu')
#             ):
#         """
#         Args:
#             classifier (nn.Module): A PyTorch model that takes an input tensor of shape [1, dim] and returns (logits, probs).
#             weights (torch.Tensor): Tensor of shape [n_classes] (e.g., [7]).
#             means (torch.Tensor): Tensor of shape [n_classes, dim] (e.g., [7, 10]).
#             stds (torch.Tensor): Tensor of shape [n_classes, dim] (e.g., [7, 10]).
#             n_cluster (int): Number of clusters to partition the error probability into.
#             alpha (float): Confidence level parameter for interval widths.
#             method (str): The method to compute the cluster. (Currently only "uniform" is supported.)
#             seed (int): Random seed for data generation.
#             device (torch.device): Device on which to run the classifier.
#         """
#         self.n_cluster = n_clusters
#         # self.model = model.to(device)
#         self.model = model.to(device)
#         self.alpha = alpha
#         # self.weights = weights      # torch.Tensor, shape: [n_classes]
#         # self.means = means          # torch.Tensor, shape: [n_classes, dim]
#         # self.stds = stds            # torch.Tensor, shape: [n_classes, dim]
#         self.params_path = params_path

#         self.method = name
#         self.device = device
#         self.n_classes = n_classes
#         self.reorder_embs = reorder_embs
#         self.kmeans_seed = seed
#         self.init_scheme = init_scheme
#         self.n_init = n_init
#         self.partionning_space = space
#         self.cov_type = cov_type
#         self.temperature = temperature
#         self.divergence = distance

#         self.normalize_gini = normalize_gini

#         self.experiment_folder = experiment_folder
#         if class_subset is not None:
#             self.dict_class_subset = {i: orig for i, orig in enumerate(class_subset)}
#         else:
#             self.dict_class_subset = None

#         self.reduction_name = reduction_name
#         self.reducing_dim = reduction_dim
#         self.n_neighbors = reduction_n_neighbors
#         self.reducer_seed = reduction_seed

#         if self.reduction_name == "umap":
#             self.reducer = umap.UMAP(n_components=self.reducing_dim,
#                                         n_neighbors= self.n_neighbors, 
#                                     #  random_state=self.reducer_seed
#                                         )
#         elif self.reduction_name == "pca":
#             self.reducer = PCA(n_components=self.reducing_dim, random_state=self.reducer_seed)
#         else:
#             self.reducer = None

#         # Initilize the density function
#         if self.partionning_space == "true_proba_error":

#             if self.param_path is None:
#                 raise ValueError("param_path must be provided when partionning_space is 'true_proba_error'")
   
#             params = np.load(self.param_path)
#             means   = params["means"]    # [n_classes, dim]
#             covs    = params["covs"]     # [n_classes, dim, dim]
#             weights = params["weights"]  # [n_classes]

#             # --- Move to GPU & factorize covariances ---

#             means   = torch.from_numpy(means).float().to(device)      # [n_classes, dim]
#             covs    = torch.from_numpy(covs).float().to(device)       # [n_classes, dim, dim]
#             weights = torch.from_numpy(weights).float().to(device)    # [n_classes]

#             self.log_weights = torch.log(weights.to(device))      # torch.Tensor, shape: [n_classes]
#             self.means = means.to(device)          # torch.Tensor, shape: [n_classes, dim]
#             self.covs = covs.to(device)            # torch.Tensor, shape: [n_classes, dim]

#             # Initilize the density function
#             self.pdfs = [MultivariateNormal(loc=self.means[i], covariance_matrix=self.covs[i]) for i in range(self.n_classes)]
#         # if stds is not None:
#         #     self.covs = torch.diag_embed(stds ** 2)
#         #     self.pdfs = [MultivariateNormal(loc=self.means[i], covariance_matrix=self.covs[i]) for i in range(self.n_classes)]

#         # Statistics to be computed in fit():
#         self.cluster_counts = None
#         self.cluster_error_means = None
#         self.cluster_error_vars = None
#         self.cluster_intervals = None


#         # Initialize the clustering algorithm
#         if self.method == "uniform":
#             self.clustering_algo = None
#         elif self.method == "kmeans":
#             self.clustering_algo = KMeans(n_clusters=self.n_cluster, 
#                                           random_state=self.kmeans_seed, 
#                                             n_init=self.n_init,
#                                           init=self.init_scheme, verbose=0)
#         elif self.method == "soft-kmeans":
#             self.clustering_algo = GaussianMixture(n_components=self.n_cluster, 
#                                                    random_state=self.kmeans_seed, 
#                                                    covariance_type=self.cov_type, 
#                                                    init_params=self.init_scheme)
#         elif self.method == "bregman-hard":
#             divergences = {
#                 "Euclidean": euclidean,
#                 "KL": kullback_leibler,
#                 "Itakura_Saito": itakura_saito,
#                 "Alpha0.5": alpha_divergence_factory(0.5),
#             }

#             self.clustering_algo = BregmanHard(
#                 n_clusters=self.n_cluster,
#                 divergence=divergences[self.divergence],  # e.g., "kl", "euclidean", etc.
#                 n_init=self.n_init,
#                 initializer=self.init_scheme,
#                 random_state= self.kmeans_seed,
#                 )

        
#         # Initialize Feature extractor
#         if self.partionning_space == "gini":
#             # self.feature_extractor = lambda x: gini(self.model(x)[0], temperature=self.temperature)
#             self.feature_extractor = lambda x: gini(self.model(x), temperature=self.temperature, normalize=self.normalize_gini)
#         elif self.partionning_space == "true_proba_error":
#             self.feature_extractor = lambda x: self.func_proba_error(x)
#         elif self.partionning_space == "probits":
#             self.feature_extractor = lambda x: torch.softmax(self.model(x) / temperature, dim=1) 
#         else: 
#             self.extractor = create_feature_extractor(self.model, {self.partionning_space : self.partionning_space})
#             self.feature_extractor = lambda x: self.extractor(x)[self.partionning_space].flatten(1)


#         # Reorder embeddings if needed
#         if self.reorder_embs:
#             orig_extractor = self.feature_extractor
#             self.feature_extractor = lambda x: orig_extractor(x) \
#                 .sort(dim=1, descending=True)[0]


    
#     # def func_proba_error(self, x):
#     #     """
#     #     Compute an error probability for a given input x.
#     #     A simple proxy is: error probability = 1 - max(predicted probability)
        
#     #     Args:
#     #         x (np.ndarray): A 1D NumPy array of shape [dim].
#     #         classifier (nn.Module): The classifier.
#     #         device (torch.device): Device to use.
            
#     #     Returns:
#     #         proba_error (float): The error probability.
#     #     """
#     #     with torch.no_grad():
#     #         # classifier returns (logits, probs)
#     #         _, model_probs = self.model(x)
#     #         model_pred = torch.argmax(model_probs, dim=1, keepdim=True)
#     #     data_probs = [self.weights[i] *  torch.exp(self.pdfs[i].log_prob(x)) for i in range(self.n_classes)]
 
#     #     data_probs = torch.stack(data_probs, dim=1) # [batch_size, n_classes]
#     #     pred_prob = data_probs.gather(1, model_pred) # [batch_size, 1]
#     #     # Normalize the probabilities
#     #     return 1 - pred_prob / data_probs.sum(dim=1, keepdim=True)  # [batch_size, n_classes]
    
#     def func_proba_error(self, x):
#         """
#         Compute an error probability for a given input x.
#         A simple proxy is: error probability = 1 - max(predicted probability)
        
#         Args:
#             x (np.ndarray): A 1D NumPy array of shape [dim].
#             classifier (nn.Module): The classifier.
#             device (torch.device): Device to use.
            
#         Returns:
#             proba_error (float): The error probability.
#         """
#         x = x.to(self.device)
#         batch_size = x.shape[0]  # batch_size 
#         with torch.no_grad():
#             logits = self.model(x)
#             model_pred = torch.argmax(logits, dim=1, keepdim=True)

#                 # 2) compute unnormalized log-posteriors: log w_i + log p_i(x)
#         x = x.view(batch_size, -1) 
#         log_data_probs = torch.stack([
#             self.pdfs[i].log_prob(x) + self.log_weights[i]
#             for i in range(len(self.pdfs))
#         ], dim=1) 

#         #  3) log-denominator = logsumexp over classes
#         log_den = torch.logsumexp(log_data_probs, dim=1, keepdim=True)  # [1,1]

#         #  4) log-posterior per class
#         log_post = log_data_probs - log_den 
#         # 5) Bayes error probability = 1 − posterior_of_predicted
#         log_post_pred = log_post.gather(1, model_pred)            # [1,1]
#         error_proba    = 1.0 - torch.exp(log_post_pred)      # [1,1]

#         # Normalize the probabilities
#         return error_proba


#     def predict_clusters(self, x):

       
#         if self.method == "uniform":
#             embs = self.feature_extractor(x)
#             cluster = torch.floor(embs * self.n_cluster).long()
#             # Handle edge case when proba_error == 1
#             cluster[cluster == self.n_cluster] = self.n_cluster - 1
#             return cluster
#         elif self.method in ["kmeans", "soft-kmeans", "bregman-hard"]:
#             embs = self.feature_extractor(x)
#             if self.reducer is not None:
#                 embs = self.reducer.transform(embs.cpu().numpy())
#                 cluster = torch.tensor(self.clustering_algo.predict(embs), 
#                                     device=self.device)
#             else:
#                 cluster = torch.tensor(self.clustering_algo.predict(embs.cpu().numpy()), 
#                                    device=self.device)
#             return cluster
#         else:
#             raise ValueError("Unsupported method")

#     def save_results(self, experiment_folder):

#             np.savez_compressed(
#                 os.path.join(experiment_folder, "cluster_results.npz"),
#                     cluster_counts=self.cluster_counts,
#                     cluster_error_means=self.cluster_error_means,
#                     cluster_error_vars=self.cluster_error_vars,
#                     cluster_intervals=self.cluster_intervals
#                 )
            
#             joblib.dump(self.clustering_algo, os.path.join(experiment_folder, 'clustering_algo.pkl'))


#     def fit(self, train_dataloader):
         
#         self.model.eval()

#         all_model_preds = []
#         all_detector_labels = []
#         all_clusters = []
#         all_embs = []
        
#         with torch.no_grad():
#             for inputs, targets in tqdm(train_dataloader, total=len(train_dataloader), desc="Training Detector", disable=False):
#                 inputs = inputs.to(self.device)
#                 targets = targets.to(self.device)
            
#                 logits = self.model(inputs)  # logits: [batch_size, num_classes]

#                 if self.dict_class_subset is not None:
#                     model_preds = torch.tensor([self.dict_class_subset[int(idx)] for idx in torch.argmax(logits, dim=1)],  # [batch_size]
#                                                 dtype=torch.long,
#                                                 device=logits.device
#                                             )
#                 else:
#                     model_preds = torch.argmax(logits, dim=1)

#                 detector_labels = (model_preds != targets).float()

#                 if self.method == "uniform":
#                     clusters = self.predict_clusters(inputs)
#                     all_clusters.append(clusters)
#                 elif self.method in ["kmeans", "soft-kmeans", "bregman-hard"]:
#                     embs = self.feature_extractor(inputs)
#                     all_embs.append(embs)

#                 all_model_preds.append(model_preds)
#                 all_detector_labels.append(detector_labels)
        
        
#         if self.method == "uniform":
#             clusters = torch.cat(all_clusters, dim=0)
#         else:
#             all_embs = torch.cat(all_embs, dim=0)
#             print("embs 0", all_embs[0,0])
#             if self.reducer is not None:
#                 # If a reducer is used, fit it on the embeddings
#                 all_embs = torch.tensor(self.reducer.fit_transform(all_embs.cpu().numpy()), device=self.device)
#             # self.all_embs = all_embs.cpu().numpy().squeeze(-1)
#             clusters = self.clustering_algo.fit_predict(all_embs.cpu().numpy())

#             clusters = torch.tensor(clusters, device=self.device)
#             if self.method == "kmeans":
#                 self.inertia = self.clustering_algo.inertia_
#             elif self.method == "soft-kmeans":
#                 self.inertia = self.clustering_algo.lower_bound_


#         detector_labels = torch.cat(all_detector_labels, dim=0)
#         print("inertia", self.inertia)
#         self.clustering(detector_labels, clusters)
        

#         if self.experiment_folder is not None:
#             self.save_results(self.experiment_folder)


#     def clustering(self, detector_labels, clusters):
#         # Initialize lists to store per-cluster statistics.
#         self.cluster_counts = []
#         self.cluster_error_means = []
#         self.cluster_error_vars = []
#         self.cluster_intervals = []
        
#         # For each cluster, compute the sample mean and variance of the error indicator.
#         for i in range(self.n_cluster):
#             idx = (clusters == i).nonzero(as_tuple=True)[0]
#             count = idx.numel()
#             self.cluster_counts.append(count)

#             if count > 0:
#                 cluster_detector_labels = detector_labels[idx]

#                 error_mean = cluster_detector_labels.mean().item()
#                 error_vars = cluster_detector_labels.var(unbiased=False).item()

#                 self.cluster_error_means.append(error_mean)
#                 self.cluster_error_vars.append(error_vars)

#                 # Confidence interval half-width using a Hoeffding-type bound.
#                 half_width = torch.sqrt(torch.log(torch.tensor(2 / self.alpha, device=self.device)) / (2 * count)).item()
#                 lower_bound = max(0.0, error_mean - half_width)
#                 upper_bound = min(1.0, error_mean + half_width)
#                 self.cluster_intervals.append((lower_bound, upper_bound))
    
#             else:
#                 self.cluster_error_means.append(0.0)
#                 self.cluster_error_vars.append(0.0)
#                 self.cluster_intervals.append((0, 1))



#     def __call__(self, x, save_embs=True):
#         """
#         Given an input x (as a NumPy array of shape [dim]), 
#         returns the upper bound of the estimated error interval for the cluster into which x falls.
        
#         Args:
#             x (np.ndarray): Input sample, shape [dim].
            
#         Returns:
#             upper_bound (float): The upper bound of the error confidence interval.
#         """
#         cluster = self.predict_clusters(x)
#         all_upper_bounds = torch.tensor([ub for (_, ub) in self.cluster_intervals],
#                                          dtype=torch.float32,
#                                          device=self.device)
#         detector_preds = all_upper_bounds[cluster]
#         # if save_embs:
#         #     # Save the embeddings for further analysis
#         #     self.embs = self.feature_extractor(x)
#         #     self.clusters = cluster
#         return detector_preds


# @register_detector("metric_learning")
# class MetricLearningLagrange:
#     def __init__(self, model, lbd=0.5, temperature=1, **kwargs):
#         self.model = model
#         self.device = next(model.parameters()).device
#         self.lbd = lbd
#         self.temperature = temperature
#         self.params = None

#     logits, detector_labels
#     def fit(self, train_dataloader, *args, **kwargs):
#         # get train logits
#         train_logits = []
#         train_labels = []
#         for data, labels in tqdm(train_dataloader, desc="Fitting metric", disable=True):
  
#             data = data.to(self.device)
#             labels = labels.to(self.device)
#             with torch.no_grad():
#                 # logits = self.model(data).cpu()
#                 logits = self.model(data)
#             # if logits.shape[1] % 2 == 1:  # openmix
#             #     print("bwe")
#             #     logits = logits[:, :-1]
#             train_logits.append(logits)
#             train_labels.append(labels)
#         train_logits = torch.cat(train_logits, dim=0)
#         train_pred = train_logits.argmax(dim=1)
#         train_labels = torch.cat(train_labels, dim=0)
#         train_labels = (train_labels != train_pred).int()

#         train_probs = torch.softmax(train_logits / self.temperature, dim=1)

#         train_probs_pos = train_probs[train_labels == 0]
#         train_probs_neg = train_probs[train_labels == 1]
#         # print("train_probs_pos", train_probs_pos[0])

#         self.params = -(1 - self.lbd) * torch.einsum("ij,ik->ijk", train_probs_pos, train_probs_pos).mean(dim=0).to(
#             self.device
#         ) + self.lbd * torch.einsum("ij,ik->ijk", train_probs_neg, train_probs_neg).mean(dim=0).to(self.device)
#         self.params = torch.tril(self.params, diagonal=-1)
#         self.params = self.params + self.params.T
#         self.params = torch.relu(self.params)
#         if torch.all(self.params <= 0):
#             # default to gini
#             self.params = torch.ones(self.params.size()).to(self.device)
#             self.params = torch.tril(self.params, diagonal=-1)
#             self.params = self.params + self.params.T
#         self.params = self.params / self.params.norm()

#     def __call__(self, inputs=None, logits=None, *args, **kwds):
#         if logits is None:
#             if inputs is None:
#                 raise ValueError("Either logits or inputs must be provided")
#             logits = self.model(inputs)
        
#         probs = torch.softmax(logits / self.temperature, dim=1)
#         params = torch.tril(self.params, diagonal=-1)
#         params = params + params.T
#         params = params / params.norm()
#         return torch.diag(probs @ params @ probs.T)

#     def export_matrix(self):
#         return self.params.cpu()


@register_detector("metric_learning")
class MetricLearningLagrange:
    def __init__(self, model, lbd=0.5, temperature=1, **kwargs):
        self.model = model
        self.device = next(model.parameters()).device
        self.lbd = lbd
        self.temperature = temperature
        self.params = None

    
    def fit(self, train_dataloader=None, logits=None, detector_labels=None, *args, **kwargs):
        # get train logits
        with torch.no_grad():
            if (logits is not None) & (detector_labels is not None):
                train_logits = logits
                train_labels = detector_labels

            elif train_dataloader is not None:
                train_logits = []
                train_labels = []
                for data, labels in tqdm(train_dataloader, desc="Fitting metric", disable=True):
        
                    data = data.to(self.device)
                    labels = labels.to(self.device)
                    with torch.no_grad():
                        # logits = self.model(data).cpu()
                        logits = self.model(data)
                    # if logits.shape[1] % 2 == 1:  # openmix
                    #     print("bwe")
                    #     logits = logits[:, :-1]
                    train_logits.append(logits)
                    train_labels.append(labels)
                train_logits = torch.cat(train_logits, dim=0)
                train_pred = train_logits.argmax(dim=1)
                train_labels = torch.cat(train_labels, dim=0)
                train_labels = (train_labels != train_pred).int()

            else:
                raise ValueError("Either train_dataloader or logits and detector_labels must be provided")

            train_probs = torch.softmax(train_logits / self.temperature, dim=1)

            train_probs_pos = train_probs[train_labels == 0]
            train_probs_neg = train_probs[train_labels == 1]
           
            mean_pos = (train_probs_pos.T @ train_probs_pos) / train_probs_pos.size(0)
            mean_neg = (train_probs_neg.T @ train_probs_neg) / train_probs_neg.size(0)
            self.params = -(1 - self.lbd) * mean_pos.to(self.device) + self.lbd * mean_neg.to(self.device)
            self.params = torch.tril(self.params, diagonal=-1)
            self.params = self.params + self.params.T
            self.params = torch.relu(self.params)
       
            if torch.all(self.params <= 0):
                # default to gini
                self.params = torch.ones(self.params.size()).to(self.device)
                self.params = torch.tril(self.params, diagonal=-1)
                self.params = self.params + self.params.T
            self.params = self.params / self.params.norm()

    def __call__(self, inputs=None, logits=None, *args, **kwds):
        if logits is None:
            if inputs is None:
                raise ValueError("Either logits or inputs must be provided")
            logits = self.model(inputs)
        
        probs = torch.softmax(logits / self.temperature, dim=1)
        params = torch.tril(self.params, diagonal=-1)
        params = params + params.T
        params = params / params.norm()
        return torch.diag(probs @ params @ probs.T)

    def export_matrix(self):
        return self.params.cpu()

# class BayesDetector:
#     def __init__(self, classifier, weights, means, stds, n_classes, device=torch.device('cpu')):
#         """
#         Args:
#             classifier (nn.Module): A PyTorch model that takes an input tensor of shape [1, dim] and returns (logits, probs).
#             weights (torch.Tensor): Tensor of shape [n_classes] (e.g., [7]).
#             means (torch.Tensor): Tensor of shape [n_classes, dim] (e.g., [7, 10]).
#             stds (torch.Tensor): Tensor of shape [n_classes, dim] (e.g., [7, 10]).
#             n_cluster (int): Number of clusters to partition the error probability into.
#             alpha (float): Confidence level parameter for interval widths.
#             method (str): The method to compute the cluster. (Currently only "uniform" is supported.)
#             seed (int): Random seed for data generation.
#             device (torch.device): Device on which to run the classifier.
#         """
#         self.classifier = classifier
#         self.weights = weights      # torch.Tensor, shape: [n_classes]
#         self.means = means          # torch.Tensor, shape: [n_classes, dim]
#         self.stds = stds            # torch.Tensor, shape: [n_classes, dim]
#         self.covs = torch.diag_embed(stds ** 2)
#         self.n_classes = n_classes
#         self.device = device

#         # Initilize the density function
#         self.pdfs = [MultivariateNormal(loc=self.means[i], covariance_matrix=self.covs[i]) for i in range(self.n_classes)]


#     def func_proba_error(self, x):
#         """
#         Compute an error probability for a given input x.
#         A simple proxy is: error probability = 1 - max(predicted probability)
        
#         Args:
#             x (np.ndarray): A 1D NumPy array of shape [dim].
#             classifier (nn.Module): The classifier.
#             device (torch.device): Device to use.
            
#         Returns:
#             proba_error (float): The error probability.
#         """
#         with torch.no_grad():
#             # classifier returns (logits, probs)
#             # _, model_probs = self.classifier(x)
#             logits = self.classifier(x)
#             model_pred = torch.argmax(logits, dim=1, keepdim=True)
#         data_probs = [self.weights[i] *  torch.exp(self.pdfs[i].log_prob(x)) for i in range(self.n_classes)]
 
#         data_probs = torch.stack(data_probs, dim=1) # [batch_size, n_classes]
#         pred_prob = data_probs.gather(1, model_pred) # [batch_size, 1]
#         # Normalize the probabilities
#         return 1 - pred_prob / data_probs.sum(dim=1, keepdim=True)  # [batch_size, n_classes]


#     def __call__(self, x):
    
#         return self.func_proba_error(x)


# This was before i load the params inside the function.

# class BayesDetector:
#     def __init__(self, classifier, weights, means, covs, n_classes, device=torch.device('cpu')):
#         """
#         Args:
#             classifier (nn.Module): A PyTorch model that takes an input tensor of shape [1, dim] and returns (logits, probs).
#             weights (torch.Tensor): Tensor of shape [n_classes] (e.g., [7]).
#             means (torch.Tensor): Tensor of shape [n_classes, dim] (e.g., [7, 10]).
#             stds (torch.Tensor): Tensor of shape [n_classes, dim] (e.g., [7, 10]).
#             n_cluster (int): Number of clusters to partition the error probability into.
#             alpha (float): Confidence level parameter for interval widths.
#             method (str): The method to compute the cluster. (Currently only "uniform" is supported.)
#             seed (int): Random seed for data generation.
#             device (torch.device): Device on which to run the classifier.
#         """
#         self.classifier = classifier
#         self.log_weights = torch.log(weights.to(device))      # torch.Tensor, shape: [n_classes]
#         self.means = means.to(device)          # torch.Tensor, shape: [n_classes, dim]
#         self.covs = covs.to(device)            # torch.Tensor, shape: [n_classes, dim]
#         self.n_classes = n_classes
#         self.device = device

#         # Initilize the density function
#         self.pdfs = [MultivariateNormal(loc=self.means[i], covariance_matrix=self.covs[i]) for i in range(self.n_classes)]


#     def func_proba_error(self, x):
#         """
#         Compute an error probability for a given input x.
#         A simple proxy is: error probability = 1 - max(predicted probability)
        
#         Args:
#             x (np.ndarray): A 1D NumPy array of shape [dim].
#             classifier (nn.Module): The classifier.
#             device (torch.device): Device to use.
            
#         Returns:
#             proba_error (float): The error probability.
#         """
#         x = x.to(self.device)
#         batch_size = x.shape[0]  # batch_size 
#         with torch.no_grad():
#             logits = self.classifier(x)
#             model_pred = torch.argmax(logits, dim=1, keepdim=True)

#                 # 2) compute unnormalized log-posteriors: log w_i + log p_i(x)
#         x = x.view(batch_size, -1) 
#         log_data_probs = torch.stack([
#             self.pdfs[i].log_prob(x) + self.log_weights[i]
#             for i in range(len(self.pdfs))
#         ], dim=1) 

#         #  3) log-denominator = logsumexp over classes
#         log_den = torch.logsumexp(log_data_probs, dim=1, keepdim=True)  # [1,1]

#         #  4) log-posterior per class
#         log_post = log_data_probs - log_den 

#         # 5) Bayes error probability = 1 − posterior_of_predicted
#         log_post_pred = log_post.gather(1, model_pred)            # [1,1]
#         error_proba    = 1.0 - torch.exp(log_post_pred)      # [1,1]

#         # Normalize the probabilities
#         return error_proba


#     def __call__(self, x):
    
#         return self.func_proba_error(x)
    
# def gini(logits, temperature=1.0, normalize=False):
#     g = torch.sum(torch.softmax(logits / temperature, dim=1) ** 2, dim=1, keepdim=True)
#     if normalize:
#         return  (1 - g) / g 
#     else:
#         return 1 - g


@register_detector("bayes")
class BayesDetector:
    def __init__(self, classifier, n_classes, weights=None, means=None, covs=None, params_path = None, device=torch.device('cpu')):
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
        self.classifier = classifier

        if params_path is not None:
            params = np.load(params_path)
            self.means   = torch.from_numpy(params["means"]).float().to(device)     # [n_classes, dim]
            self.covs    = torch.from_numpy(params["covs"]).float().to(device)     # [n_classes, dim, dim]
            self.log_weights = torch.log(torch.from_numpy(params["weights"]).float().to(device))   # [n_classes]

        elif weights is not None and means is not None and covs is not None:
            self.log_weights = torch.log(weights.to(device))      # torch.Tensor, shape: [n_classes]
            self.means = means.to(device)          # torch.Tensor, shape: [n_classes, dim]
            self.covs = covs.to(device)            # torch.Tensor, shape: [n_classes, dim]
            
        else:
            raise ValueError("Either params_path or weights, means, and covs must be provided.")
        

               
        self.n_classes = n_classes
        self.device = device

        # Initilize the density function
        self.pdfs = [MultivariateNormal(loc=self.means[i], covariance_matrix=self.covs[i]) for i in range(self.n_classes)]


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
        x = x.to(self.device)
        batch_size = x.shape[0]  # batch_size 
        with torch.no_grad():
            logits = self.classifier(x)
            model_pred = torch.argmax(logits, dim=1, keepdim=True)

                # 2) compute unnormalized log-posteriors: log w_i + log p_i(x)
        x = x.view(batch_size, -1) 
        log_data_probs = torch.stack([
            self.pdfs[i].log_prob(x) + self.log_weights[i]
            for i in range(len(self.pdfs))
        ], dim=1) 

        #  3) log-denominator = logsumexp over classes
        log_den = torch.logsumexp(log_data_probs, dim=1, keepdim=True)  # [1,1]

        #  4) log-posterior per class
        log_post = log_data_probs - log_den 

        # 5) Bayes error probability = 1 − posterior_of_predicted
        log_post_pred = log_post.gather(1, model_pred)            # [1,1]
        error_proba    = 1.0 - torch.exp(log_post_pred)      # [1,1]

        # Normalize the probabilities
        return error_proba


    def __call__(self, x):
    
        return self.func_proba_error(x)


@register_detector("gini")
class GiniDetector:
    def __init__(self, model, temperature=1, normalize_gini=False, device=torch.device('cpu'), **kwargs):
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
        self.model = model
        self.temperature = temperature
        self.device = device
        self.normalize = normalize_gini



    def __call__(self, inputs=None, logits=None):
        if logits is None:
            if inputs is None:
                raise ValueError("Either logits or inputs must be provided")
            inputs = inputs.to(self.device)
            logits = self.model(inputs)

        return gini(logits, temperature=self.temperature, normalize=self.normalize)
    

@register_detector("max_proba")
class MaxProbaDetector:
    def __init__(self, model, temperature, device=torch.device('cpu'), **kwargs):
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
        self.model = model
        self.temperature = temperature
        self.device = device



    def __call__(self, inputs=None, logits=None):
        if logits is None:
            if inputs is None:
                raise ValueError("Either logits or inputs must be provided")
            logits = self.model(inputs)

        # return -torch.softmax(logits / self.temperature, dim=1).max(dim=1, keepdim=True)[0]  # [batch_size]
        return -torch.softmax(logits / self.temperature, dim=1).max(dim=1)[0]  # [batch_size]