import torch
from torch.distributions import MultivariateNormal
from tqdm import tqdm
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from torchvision.models.feature_extraction import create_feature_extractor
import umap
from sklearn.decomposition import PCA
from .clustering.models import BregmanHard
from .clustering.divergences import (
    euclidean,
    kullback_leibler,
    itakura_saito,
    alpha_divergence_factory,
)

def gini(logits, temperature=1.0, normalize=False):
    g =torch.sum(torch.softmax(logits / temperature, dim=1) ** 2, dim=1, keepdim=True)
    if normalize:
        return  (1 - g) / g 
    else:
        return 1 - g


class PartitionDetector:
    def __init__(
            self, model, weights=None, means=None, covs=None,
            n_cluster=100, alpha=0.05, method="uniform", device=torch.device('cpu'),
            n_classes=7, kmeans_seed=0, init_scheme="k-means++", # "random" or "k-means++", 
            n_init=1, # Number of initializations for k-means
            partionning_space="true_proba_error", temperature=1.0, cov_type = None,
            reducer=None, # For dimensionality reduction
            reduction_dim=2, n_neighbors=15, reducer_seed=0, # For UMAP
            normalize_gini=False, # Whether to normalize the Gini coefficient
            divergence=None, # For BregmanHard clustering
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
        # self.weights = weights      # torch.Tensor, shape: [n_classes]
        # self.means = means          # torch.Tensor, shape: [n_classes, dim]
        # self.stds = stds            # torch.Tensor, shape: [n_classes, dim]

        self.method = method
        self.device = device
        self.n_classes = n_classes
        self.kmeans_seed = kmeans_seed
        self.init_scheme = init_scheme
        self.n_init = n_init
        self.partionning_space = partionning_space
        self.cov_type = cov_type
        self.temperature = temperature
        self.divergence = divergence

        self.normalize_gini = normalize_gini

        self.reducer = reducer
        self.reducing_dim = reduction_dim
        self.n_neighbors = n_neighbors
        self.reducer_seed = reducer_seed
        if self.reducer is not None:
            print("Using dimensionality reduction with", self.reducer)
            if reducer == "umap":
                self.reducer = umap.UMAP(n_components=self.reducing_dim,
                                         n_neighbors= self.n_neighbors, 
                                        #  random_state=self.reducer_seed
                                         )
            elif reducer == "pca":
                self.reducer = PCA(n_components=self.reducing_dim, random_state=self.kmeans_seed)

        # Initilize the density function
        if partionning_space == "true_proba_error":
            import os
            import numpy as np
            param_path = os.path.join("checkpoints", "ce", 
                                  f"resnet34_synth_dim-3072_classes-10",
                                  "data_parameters.npz")
            params = np.load(param_path)
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
        # elif self.method == "spectral_clustering":
        
        #     self.clustering_algo = SpectralClustering(n_components=self.n_cluster, 
        #                                            random_state=self.kmeans_seed,
        #                                            assign_labels= self.assign_labels,
        #                                            affinity=self.affinity,
        #                                            )
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

        
        # Initialize Feature extractor
        if self.partionning_space == "gini":
            # self.feature_extractor = lambda x: gini(self.model(x)[0], temperature=self.temperature)
            self.feature_extractor = lambda x: gini(self.model(x), temperature=self.temperature, normalize=self.normalize_gini)
        elif self.partionning_space == "true_proba_error":
            self.feature_extractor = lambda x: self.func_proba_error(x)
        elif self.partionning_space == "probits":
            self.feature_extractor = lambda x: torch.softmax(self.model(x) / temperature, dim=1) 
        else: 
            self.extractor = create_feature_extractor(self.model, {partionning_space : partionning_space})
            self.feature_extractor = lambda x: self.extractor(x)[partionning_space].flatten(1)
    
    # def func_proba_error(self, x):
    #     """
    #     Compute an error probability for a given input x.
    #     A simple proxy is: error probability = 1 - max(predicted probability)
        
    #     Args:
    #         x (np.ndarray): A 1D NumPy array of shape [dim].
    #         classifier (nn.Module): The classifier.
    #         device (torch.device): Device to use.
            
    #     Returns:
    #         proba_error (float): The error probability.
    #     """
    #     with torch.no_grad():
    #         # classifier returns (logits, probs)
    #         _, model_probs = self.model(x)
    #         model_pred = torch.argmax(model_probs, dim=1, keepdim=True)
    #     data_probs = [self.weights[i] *  torch.exp(self.pdfs[i].log_prob(x)) for i in range(self.n_classes)]
 
    #     data_probs = torch.stack(data_probs, dim=1) # [batch_size, n_classes]
    #     pred_prob = data_probs.gather(1, model_pred) # [batch_size, 1]
    #     # Normalize the probabilities
    #     return 1 - pred_prob / data_probs.sum(dim=1, keepdim=True)  # [batch_size, n_classes]
    
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


    def predict_clusters(self, x):

       
        if self.method == "uniform":
            embs = self.feature_extractor(x)
            cluster = torch.floor(embs * self.n_cluster).long()
            # Handle edge case when proba_error == 1
            cluster[cluster == self.n_cluster] = self.n_cluster - 1
            return cluster
        elif self.method in ["kmeans", "soft-kmeans", "bregman-hard"]:
            embs = self.feature_extractor(x)
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
                # print(targets[:15])

                # logits, _ = self.model(inputs)  # logits: [batch_size, num_classes]
                logits = self.model(inputs)  # logits: [batch_size, num_classes]
                model_preds = torch.argmax(logits, dim=1)  # [batch_size]

                detector_labels = (model_preds != targets).float()

                if self.method == "uniform":
                    clusters = self.predict_clusters(inputs)
                    all_clusters.append(clusters)
                elif self.method in ["kmeans", "soft-kmeans", "bregman-hard"]:
                    embs = self.feature_extractor(inputs)
                    all_embs.append(embs)

                all_model_preds.append(model_preds)
                all_detector_labels.append(detector_labels)
        
        
        if self.method == "uniform":
            clusters = torch.cat(all_clusters, dim=0)
        else:
            all_embs = torch.cat(all_embs, dim=0)
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

class MetricLearningLagrange:
    def __init__(self, model, lbd=0.5, temperature=1, **kwargs):
        self.model = model
        self.device = next(model.parameters()).device
        self.lbd = lbd
        self.temperature = temperature
        self.params = None

    def fit(self, train_dataloader, *args, **kwargs):
        # get train logits
        train_logits = []
        train_labels = []
        for data, labels in tqdm(train_dataloader, desc="Fitting metric"):
  
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

        train_probs = torch.softmax(train_logits / self.temperature, dim=1)

        train_probs_pos = train_probs[train_labels == 0]
        train_probs_neg = train_probs[train_labels == 1]
        # print("train_probs_pos", train_probs_pos[0])

        self.params = -(1 - self.lbd) * torch.einsum("ij,ik->ijk", train_probs_pos, train_probs_pos).mean(dim=0).to(
            self.device
        ) + self.lbd * torch.einsum("ij,ik->ijk", train_probs_neg, train_probs_neg).mean(dim=0).to(self.device)
        self.params = torch.tril(self.params, diagonal=-1)
        self.params = self.params + self.params.T
        self.params = torch.relu(self.params)
        if torch.all(self.params <= 0):
            # default to gini
            self.params = torch.ones(self.params.size()).to(self.device)
            self.params = torch.tril(self.params, diagonal=-1)
            self.params = self.params + self.params.T
        self.params = self.params / self.params.norm()

    def __call__(self, inputs, *args, **kwds):
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

class BayesDetector:
    def __init__(self, classifier, weights, means, covs, n_classes, device=torch.device('cpu')):
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
        self.log_weights = torch.log(weights.to(device))      # torch.Tensor, shape: [n_classes]
        self.means = means.to(device)          # torch.Tensor, shape: [n_classes, dim]
        self.covs = covs.to(device)            # torch.Tensor, shape: [n_classes, dim]
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
    
# def gini(logits, temperature=1.0, normalize=False):
#     g = torch.sum(torch.softmax(logits / temperature, dim=1) ** 2, dim=1, keepdim=True)
#     if normalize:
#         return  (1 - g) / g 
#     else:
#         return 1 - g

class GiniDetector:
    def __init__(self, classifier, temperature, normalize, device=torch.device('cpu')):
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
        self.temperature = temperature
        self.device = device
        self.normalize = normalize



    def __call__(self, x):
        logits = self.classifier(x)
        return gini(logits, temperature=self.temperature, normalize=self.normalize)
    

class MaxProbaDetector:
    def __init__(self, model, temperature, device=torch.device('cpu')):
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



    def __call__(self, x):
        logits = self.model(x)
        return -torch.softmax(logits / self.temperature, dim=1).max(dim=1, keepdim=True)[0]  # [batch_size]