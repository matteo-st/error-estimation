from functools import partial
from typing import Any, Tuple, Dict, List
import torch
import torch.utils.data
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
import copy
from src.utils.eval import evaluate_classification, get_classification_metrics
from synthetic_code.utils.dataset import load_data_config, load_generating_params, bayes_proba
import numpy as np
import pandas as pd
from torch.autograd import Variable


def g(logits, temperature=1.0):
    return torch.sum(torch.softmax(logits / temperature, dim=1) ** 2, dim=1, keepdim=True)


def doctor(logits: torch.Tensor, temperature: float = 1.0, **kwargs):
    g_out = g(logits=logits, temperature=temperature)
    return (1 - g_out) / g_out


def odin(logits: torch.Tensor, temperature: float = 1.0, **kwargs):
    return -torch.softmax(logits / temperature, dim=1).max(dim=1)[0]


def msp(logits: torch.Tensor, **kwargs):
    return -torch.softmax(logits, dim=1).max(dim=1)[0]


def entropy(logits: torch.Tensor, **kwargs):
    probs = torch.softmax(logits, dim=1)
    return -torch.sum(probs * torch.log(probs), dim=1)


def enable_dropout(model):
    """Function to enable the dropout layers during test-time."""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def add_dropout_layer(model, dropout_p=0.5):
    """Function to add dropout layers to the model.

    replace model.linear by a sequential model with dropout and the same linear layer
    """
    # get the last layer
    last_layer = model.linear
    # remove it
    model.linear = torch.nn.Sequential()
    # add dropout
    model.linear.add_module("dropout", torch.nn.Dropout(dropout_p))
    # add the last layer
    model.linear.add_module("linear", last_layer)


class BayesDetector:
    def __init__(self, model, dataset, *args, **kwargs):
        self.model = model
        self.device = next(model.parameters()).device
        self.dataset = dataset
        self.dim = dataset["dim"]
        self.num_classes = dataset["num_classes"]
        self.data_folder = f"data/synthetic/dim-{self.dim}_classes-{self.num_classes}"
        self.means, self.stds = load_generating_params(self.data_folder)

    def evaluate(self, dataloader: torch.utils.data.DataLoader, name_save_file: str = "test"):
        self.model.eval()
        test_preds, test_targets, test_scores = [], [], []
        
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, total=len(dataloader), desc="Evaluating"):
                inputs = inputs.to(self.device)
                logits = self.model(inputs)
                preds = torch.argmax(logits, dim=1)
                data_cond_probs = bayes_proba(inputs.cpu().numpy(), self.means, self.stds)
                scores = 1 - data_cond_probs[np.arange(len(inputs)), preds.cpu().numpy()]
        
                test_preds.append(preds.cpu().numpy())
                test_targets.append(targets.cpu().numpy())
                test_scores.append(scores)

            test_preds = np.concat(test_preds, axis=0)
            test_targets = np.concat(test_targets, axis=0)
            test_scores = np.concat(test_scores, axis=0)

        return test_preds, test_targets, test_scores

   
        

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
            with torch.no_grad():
                logits = self.model(data).cpu()
            if logits.shape[1] % 2 == 1:  # openmix
                logits = logits[:, :-1]
            train_logits.append(logits)
            train_labels.append(labels)
        train_logits = torch.cat(train_logits, dim=0)
        train_pred = train_logits.argmax(dim=1)
        train_labels = torch.cat(train_labels, dim=0)
        train_labels = (train_labels != train_pred).int()

        train_probs = torch.softmax(train_logits / self.temperature, dim=1)

        train_probs_pos = train_probs[train_labels == 0]
        train_probs_neg = train_probs[train_labels == 1]

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

    def __call__(self, logits, *args, **kwds):
        probs = torch.softmax(logits / self.temperature, dim=1)
        params = torch.tril(self.params, diagonal=-1)
        params = params + params.T
        params = params / params.norm()
        return torch.diag(probs @ params @ probs.T)

    def export_matrix(self):
        return self.params.cpu()


class MLP(torch.nn.Module):
    def __init__(self, num_classes, hidden_size=128, num_hidden_layers=1, dropout_p=0, *args, **kwargs) -> None:
        super().__init__()
        self.layer0 = torch.nn.Linear(num_classes, hidden_size)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.classifier = torch.nn.Linear(hidden_size, 1)
        self.hidden_layers = None
        if num_hidden_layers > 0:
            self.hidden_layers = torch.nn.Sequential(
                *(
                    [
                        torch.nn.Linear(hidden_size, hidden_size),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(dropout_p),
                    ]
                    * num_hidden_layers
                ),
            )

    def forward(self, x):
        x = torch.relu(self.layer0(x))
        x = self.dropout(x)
        if self.hidden_layers is not None:
            x = self.hidden_layers(x)
        return torch.sigmoid(self.classifier(x))


class MLPTrainer:
    def __init__(self, model, num_classes, epochs=100, hidden_size=128, num_hidden_layers=2, *args, **kwargs) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        print("num_classes", num_classes)
        self.net = MLP(num_classes, hidden_size, num_hidden_layers)
        self.net = self.net.to(self.device)
        self.criterion = torch.nn.BCELoss()
        self.epochs = epochs

    def fit(self, train_dataloader, val_dataloader, *args, **kwargs):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        best_acc = 0
        best_fpr = 1
        best_auc = 0
        loss = torch.inf
        best_weights = copy.deepcopy(self.net.to("cpu").state_dict())
        self.net = self.net.to(self.device)
        progress_bar = tqdm(range(self.epochs), desc="Fit", unit="e")
        for e in progress_bar:
            # train step
            self.net.train()
            for data, labels in train_dataloader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                with torch.no_grad():
                    logits = self.model(data)
                model_pred = logits.argmax(dim=1)
                bin_labels = (model_pred != labels).float()
                y_pred = self.net(logits)
                loss = self.criterion(y_pred.view(-1), bin_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss = loss.item()

            # eval
            self.net.eval()
            scores = []
            targets = []
            preds = []
            for data, labels in val_dataloader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                with torch.no_grad():
                    logits = self.model(data)
                    model_pred = logits.argmax(dim=1)
                    bin_labels = (model_pred != labels).int()
                    y_pred = self.net(logits)
                preds.append(y_pred.round())
                targets.append(bin_labels.view(-1))
                scores.append(y_pred.view(-1))
            targets = torch.cat(targets)
            scores = torch.cat(scores)
            preds = torch.cat(preds)
            acc, roc_auc, fpr, var, tpr, aurc = get_classification_metrics(preds, targets, scores)
            print
            # if acc > best_acc:
            #     best_acc = acc
            #     best_weights = copy.deepcopy(self.net.to("cpu").state_dict())
            #     self.net = self.net.to(self.device)
            if fpr < best_fpr:
                best_fpr = fpr
                best_auc = roc_auc
                best_aurc = aurc
                best_weights = copy.deepcopy(self.net.to("cpu").state_dict())
                self.net = self.net.to(self.device)
            # if roc_auc > best_auc:
            #     best_auc = roc_auc
            #     best_weights = copy.deepcopy(self.net.to("cpu").state_dict())
            #     self.net = self.net.to(self.device)

            progress_bar.set_postfix(l=loss, acc=acc, fpr=fpr, b_auc=best_auc, b_fpr=best_fpr, auc=roc_auc)

        self.net.load_state_dict(best_weights)
        self.net = self.net.to(self.device)

    def __call__(self, logits, *args: Any, **kwds: Any) -> Any:
        logits_device = logits.device
        logits = logits.to(self.device)
        self.net.eval()
        return self.net(logits).to(logits_device)


import os
import numpy as np
from tqdm import tqdm
import torch
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.cluster import KMeans

import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.cluster import KMeans
from torch.autograd import Variable
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

class EmbeddingKMeans:
    """
    Clusters the embedding space (extracted from the model) using different clustering strategies.
    In addition to classical KMeans (using scikit-learn) or uniform partitioning, a custom 
    k-means algorithm (kmeans_grad) is implemented in which the centroid update is computed via gradient descent.
    
    The divergence used in the optimization is selected via a string parameter (f_divergence) which 
    can be one of:
       - 'kl': Kullback-Leibler divergence
       - 'js': Jensen-Shannon divergence
       - 'chi2': Chi-square divergence
       - 'reverse_kl': Reverse Kullback-Leibler divergence
       - 'hellinger': Squared Hellinger distance
       - 'tv': Total variation distance
    """
    def __init__(self, model, n_cluster=10, clustering_space=None,  
                 alpha=0.05, results_folder=None, model_name="resnet34",
                 temperature=1, clustering_method="kmeans", magnitude=0,
                 grad_n_iters=100, grad_lr=0.1, kmens_n_iter=100,
                 f_divergence="kl", init_scheme="kmeans++",
                 kmeans_seed=None,  # New parameter for initialization seed.
                 *args, **kwargs) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        self.n_cluster = n_cluster
        self.temperature = temperature
        self.alpha = alpha
        self.results_folder = results_folder
        self.clustering_method = clustering_method  # "kmeans", "kmeans_grad", or "uniform"
        self.magnitude = magnitude
        self.f_divergence = f_divergence.lower()
        self.grad_n_iters = grad_n_iters
        self.grad_lr = grad_lr      
        self.kmens_n_iter = kmens_n_iter
        # New parameter: initialization scheme for kmeans ("random" or "kmeans++")
        self.init_scheme = init_scheme.lower()
        # New parameter: seed for kmeans initialization.
        self.kmeans_seed = kmeans_seed
        
        # Select the f-divergence function.
        if self.f_divergence == 'kl':
            self.f_divergence_fn = lambda p, q, eps=1e-10: torch.sum(p * (torch.log(p+eps) - torch.log(q+eps)), dim=-1)
        elif self.f_divergence == 'js':
            def js_div(p, q, eps=1e-10):
                m = 0.5 * (p + q)
                return 0.5 * torch.sum(p * (torch.log(p+eps) - torch.log(m+eps)), dim=-1) + \
                       0.5 * torch.sum(q * (torch.log(q+eps) - torch.log(m+eps)), dim=-1)
            self.f_divergence_fn = js_div
        elif self.f_divergence == 'eucli':
            self.f_divergence_fn = lambda p, q: torch.sum((p - q)**2, dim=-1)
        elif self.f_divergence == 'chi2':
            self.f_divergence_fn = lambda p, q, eps=1e-10: torch.sum((p - q)**2 / (q+eps), dim=-1)
        elif self.f_divergence == 'reverse_kl':
            self.f_divergence_fn = lambda p, q, eps=1e-10: torch.sum(q * (torch.log(q+eps) - torch.log(p+eps)), dim=-1)
        elif self.f_divergence == 'hellinger':
            self.f_divergence_fn = lambda p, q, eps=1e-10: torch.sum((torch.sqrt(p+eps) - torch.sqrt(q+eps))**2, dim=-1) / 2
        elif self.f_divergence == 'tv':
            self.f_divergence_fn = lambda p, q, eps=1e-10: 0.5 * torch.sum(torch.abs(p - q), dim=-1)
        else:
            raise ValueError("Unsupported f divergence: choose among 'kl', 'js', 'chi2', 'reverse_kl', 'hellinger', 'tv'")

        # Configure the feature extractor for the clustering space.
        if model_name == "resnet34":
            nodes_registry = {"last_layer": "view", "logits": "linear"}
            if clustering_space == "probs":
                self.return_nodes = {nodes_registry["logits"]: "clustering_space"}
                feature_extractor = create_feature_extractor(model, self.return_nodes)
                self.feature_extractor = lambda x: F.softmax(feature_extractor(x)["clustering_space"] / self.temperature, dim=1)
            elif clustering_space == "gini":
                self.return_nodes = {nodes_registry["logits"]: "clustering_space"}
                feature_extractor = create_feature_extractor(model, self.return_nodes)
                self.feature_extractor = lambda x: doctor(feature_extractor(x)["clustering_space"], self.temperature)
            elif clustering_space in nodes_registry.keys(): 
                self.return_nodes = {nodes_registry[clustering_space]: "clustering_space"}
                feature_extractor = create_feature_extractor(model, self.return_nodes)
                self.feature_extractor = lambda x: feature_extractor(x)["clustering_space"]
            else:
                raise ValueError(f"Nodes registry is not implemented for model {model}")

        self.clustering = None
        self.centroids = None

        # Histories for convergence and overall loss.
        self.centroid_convergence_history = []
        self.kmeans_loss_history = []

    def _init_centroids_kmeanspp(self, data, eps=1e-10):
        """
        Initializes centroids using the KMeans++ algorithm.
        The distance between a point and a centroid is computed using the f_divergence_fn.
        
        Args:
            data (torch.Tensor): Tensor of shape (n_samples, K) containing the data points.
            eps (float): Small constant for numerical stability.
        
        Returns:
            torch.Tensor: Initialized centroids of shape (n_cluster, K).
        """
        n_samples = data.shape[0]
        centroids = []
        # Create a generator if a specific seed is provided.
        if self.kmeans_seed is not None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.kmeans_seed)
        else:
            gen = None

        # Choose the first centroid uniformly at random.
        first_idx = torch.randint(0, n_samples, (1,), generator=gen)
        centroids.append(data[first_idx].clone())
        # Initialize distances: divergence from the first centroid.
        distances = self.f_divergence_fn(data, centroids[0])
        for _ in range(1, self.n_cluster):
            # Compute probabilities proportional to distances.
            prob = distances / torch.sum(distances)
            # Sample next centroid index.
            next_idx = torch.multinomial(prob, 1, generator=gen)
            next_centroid = data[next_idx].clone()
            centroids.append(next_centroid)
            # Update distances: for each point, take the minimum divergence so far.
            new_distances = self.f_divergence_fn(data, next_centroid)
            distances = torch.min(distances, new_distances)
        return torch.stack(centroids, dim=0)

    def _compute_centroid(self, data: torch.Tensor, eps=1e-10):
        """
        Computes the centroid of a set of distributions via gradient descent.
        We parameterize x = softmax(y) and minimize:
            L(y) = sum_i f_divergence(p_i || softmax(y))
        Returns a tuple: (final centroid, convergence_history)
        where convergence_history is a list of loss values for each inner iteration.
        """
        K = data.shape[1]
        y = torch.zeros(K, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([y], lr=self.grad_lr)
        convergence_history = []
        for _ in tqdm(range(self.grad_n_iters), desc="Centroid computation", leave=False):
            optimizer.zero_grad()
            centroid = F.softmax(y, dim=0)
            loss = torch.sum(self.f_divergence_fn(data, centroid))
            loss.backward()
            optimizer.step()
            convergence_history.append(loss.item())
        return F.softmax(y, dim=0).detach(), convergence_history

    def _custom_kmeans(self, embs: np.ndarray, eps=1e-10):
        """
        Custom k-means algorithm:
         - Initializes centroids using the selected initialization scheme.
         - For each outer iteration, assigns each point to the closest centroid (w.r.t. f_divergence_fn).
         - Updates each centroid via _compute_centroid.
         - Records the convergence history for each centroid update.
         - Also records the overall k-means loss (sum of divergences over points) at each outer iteration.
         
        Returns:
            cluster_assignments: numpy array of cluster indices.
            centroids: final centroids as a numpy array.
        """
        data = torch.tensor(embs, dtype=torch.float32, device=self.device)
        n_samples, K = data.shape

        # Initialization based on the chosen scheme.
        if self.init_scheme == "random":
            if self.kmeans_seed is not None:
                gen = torch.Generator(device="cpu")
                gen.manual_seed(self.kmeans_seed)
                indices = torch.randperm(n_samples, generator=gen)[:self.n_cluster]
            else:
                indices = torch.randperm(n_samples)[:self.n_cluster]
            centroids = data[indices].clone()
        elif self.init_scheme == "kmeans++":
            centroids = self._init_centroids_kmeanspp(data)
        else:
            raise ValueError("Unsupported initialization scheme. Choose 'random' or 'kmeans++'.")

        # Reset histories.
        self.centroid_convergence_history = []
        self.kmeans_loss_history = []
        
        for outer_it in tqdm(range(self.kmens_n_iter), desc="KMeans_grad iterations"):
            D = torch.zeros(n_samples, self.n_cluster, device=self.device)
            for j in range(self.n_cluster):
                D[:, j] = self.f_divergence_fn(data, centroids[j])
            cluster_assignments = torch.argmin(D, dim=1)
            # Compute overall k-means loss for this iteration.
            loss_km = torch.sum(D.gather(1, cluster_assignments.unsqueeze(1)))
            self.kmeans_loss_history.append(loss_km.item())
            
            new_centroids = []
            iter_history = {}  # to store convergence history for each cluster at this iteration.
            for j in range(self.n_cluster):
                cluster_data = data[cluster_assignments == j]
                if cluster_data.shape[0] == 0:
                    new_centroids.append(data[torch.randint(0, n_samples, (1,))].squeeze(0))
                    iter_history[j] = []
                else:
                    new_centroid, centroid_history = self._compute_centroid(cluster_data, eps=eps)
                    new_centroids.append(new_centroid)
                    iter_history[j] = centroid_history
            self.centroid_convergence_history.append(iter_history)
            new_centroids = torch.stack(new_centroids, dim=0)
            diff = torch.norm(new_centroids - centroids)
            centroids = new_centroids
            if diff < 1e-4:
                break
        return cluster_assignments.cpu().numpy(), centroids.cpu().detach().numpy()
    
    def save_centroid_convergence_history_csv(self, name_save_file):
        """
        Saves the convergence history of each centroid update to a CSV file.
        Each row contains: outer_iteration, cluster, inner_iteration, loss.
        """
        records = []
        for outer_it, iter_history in enumerate(self.centroid_convergence_history):
            for cluster, loss_history in iter_history.items():
                for inner_it, loss_value in enumerate(loss_history):
                    records.append({
                        "outer_iteration": outer_it,
                        "cluster": cluster,
                        "inner_iteration": inner_it,
                        "loss": loss_value
                    })
        df = pd.DataFrame(records)
        file_path = os.path.join(self.results_folder, f"{name_save_file}_centroid_convergence.csv")
        df.to_csv(file_path, index=False)
        print(f"Centroid convergence history saved to {file_path}")

    def save_kmeans_loss_history_csv(self, name_save_file):
        """
        Saves the overall k-means loss (inertia) history to a CSV file.
        Each row contains: outer_iteration, kmeans_loss.
        """
        records = []
        for outer_it, loss_value in enumerate(self.kmeans_loss_history):
            records.append({
                "outer_iteration": outer_it,
                "kmeans_loss": loss_value
            })
        df = pd.DataFrame(records)
        file_path = os.path.join(self.results_folder, f"{name_save_file}_kmeans_loss_history.csv")
        df.to_csv(file_path, index=False)
        print(f"KMeans loss history saved to {file_path}")

    def fit(self, train_dataloader: torch.utils.data.DataLoader, *args, name_save_file: str = "train", save_tensors: bool = False, **kwargs) -> None:
        """
        Extracts embeddings from the training dataloader, performs clustering according to the selected method,
        computes cluster-level error metrics, and saves the results.
        
        For clustering_method "kmeans_grad", the custom k-means algorithm is executed and the convergence histories
        (both for centroid updates and the overall k-means loss) are recorded and saved.
        """
        self.model.eval()
        embs_list, preds_list, targets_list, errors_list = [], [], [], []
        
        with torch.no_grad():
            for inputs, targets in tqdm(train_dataloader, total=len(train_dataloader), desc="Training"):
                inputs = inputs.to(self.device)
                if self.magnitude > 0 and self.clustering_method == "uniform":
                    inputs = Variable(inputs, requires_grad=True)
                    gini = self.feature_extractor(inputs)
                    gini = torch.log(gini)
                    gini.sum().backward()
                    inputs = inputs - self.magnitude * torch.sign(-inputs.grad)
                    inputs = Variable(inputs, requires_grad=False)
                
                emb = self.feature_extractor(inputs)
                emb = emb.cpu().numpy()
                
                logits = self.model(inputs)
                preds = logits.argmax(dim=1)
                errors = (preds != targets.to(self.device)).float().cpu().numpy()
                
                embs_list.append(emb)
                preds_list.append(preds.cpu().numpy())
                targets_list.append(targets.cpu().numpy())
                errors_list.append(errors)
        
        embs = np.concatenate(embs_list, axis=0)
        preds = np.concatenate(preds_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        errors = np.concatenate(errors_list, axis=0)
        
        if self.clustering_method == "kmeans":
            # random_state = 42
            self.clustering = KMeans(n_clusters=self.n_cluster, random_state=self.kmeans_seed, init=self.init_scheme, verbose=0)
            clusters = self.clustering.fit_predict(embs)
        elif self.clustering_method == "kmeans_grad":
            clusters, centroids = self._custom_kmeans(embs)
            self.centroids = centroids
            # Save the histories in CSV files.
            if self.results_folder is not None:
                self.save_centroid_convergence_history_csv(name_save_file)
                self.save_kmeans_loss_history_csv(name_save_file)
        elif self.clustering_method == "uniform":
            clusters = self._predict_uniform_clusters(embs)
        else:
            raise ValueError("Unsupported clustering method: choose 'kmeans', 'kmeans_grad' or 'uniform'.")
        
        self.fit_concentration(clusters, errors, name_save_file)
        scores = np.array([self.cluster_intervals[i][1] for i in clusters])
        
        if save_tensors:
            self.save_tensors(embs, clusters, scores, preds, targets, errors, name_save_file)
        return preds, targets, scores

    def fit_concentration(self, clusters: np.ndarray, errors: np.ndarray, name_save_file: str = "train") -> None:
        self.cluster_counts, self.cluster_error_means, self.cluster_error_vars, self.cluster_intervals = [], [], [], []
        
        for i in range(self.n_cluster):
            idx = np.where(clusters == i)[0]
            count = idx.size
            self.cluster_counts.append(count)
            if count > 0:
                error_mean = np.mean(errors[idx])
                self.cluster_error_means.append(error_mean)
                error_vars = np.var(errors[idx])
                self.cluster_error_vars.append(error_vars)
                half_width = np.sqrt(np.log(2 / self.alpha) / (2 * count))
                lower_bound = max(0.0, error_mean - half_width)
                upper_bound = min(1.0, error_mean + half_width)
                self.cluster_intervals.append((lower_bound, upper_bound))
            else:
                self.cluster_error_means.append(0.0)
                self.cluster_error_vars.append(0.0)
                self.cluster_intervals.append((0, 1))
                
        metrics_dict = {
            "cluster_counts": self.cluster_counts,
            "cluster_means": self.cluster_error_means,
            "cluster_vars": self.cluster_error_vars,
            "cluster_intervals": self.cluster_intervals
        }
        csv_path = os.path.join(self.results_folder, name_save_file + "_cluster_results.csv")
        results_df = pd.DataFrame(metrics_dict)
        if not os.path.isfile(csv_path):
            results_df.to_csv(csv_path, index=False, header=True)
        else:
            results_df.to_csv(csv_path, mode="a", index=False, header=False)
        print(f"Cluster evaluation results saved to {csv_path}")

        # Save Inertie
        csv_path = os.path.join(self.results_folder, name_save_file + "_cluster_inertia.csv") 
        pd.DataFrame({"inertie" : [self.clustering.inertia_]}).to_csv(csv_path, index=False, header=True)

        return 

    def evaluate(self, dataloader: torch.utils.data.DataLoader, name_save_file: str = "test", evaluate_clusters: bool = True, save_tensors: bool = False):
        self.model.eval()
        embs_list, clusters_list, scores_list = [], [], []
        preds_list, targets_list, errors_list = [], [], []
        
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, total=len(dataloader), desc="Evaluating"):
                inputs = inputs.to(self.device)
                if self.magnitude > 0 and self.clustering_method == "uniform":
                    inputs = Variable(inputs, requires_grad=True)
                    gini = self.feature_extractor(inputs)
                    gini = torch.log(gini)
                    gini.sum().backward()
                    inputs = inputs - self.magnitude * torch.sign(-inputs.grad)
                    inputs = Variable(inputs, requires_grad=False)
                emb = self.feature_extractor(inputs)
                emb = emb.cpu().numpy()
                
                if self.clustering_method == "kmeans":
                    clusters = self.clustering.predict(emb)
                elif self.clustering_method == "kmeans_grad":
                    data = torch.tensor(emb, dtype=torch.float32, device=self.device)
                    n_samples = data.shape[0]
                    eps = 1e-10
                    D = torch.zeros(n_samples, self.n_cluster, device=self.device)
                    centroids = torch.tensor(self.centroids, dtype=torch.float32, device=self.device)
                    for j in range(self.n_cluster):
                        D[:, j] = self.f_divergence_fn(data, centroids[j])
                    clusters = torch.argmin(D, dim=1).cpu().numpy()
                elif self.clustering_method == "uniform":
                    clusters = self._predict_uniform_clusters(emb)
                else:
                    raise ValueError("Unsupported clustering method")
                scores = np.array([self.cluster_intervals[i][1] for i in clusters])
                logits = self.model(inputs)
                preds = logits.argmax(dim=1)
                errors = (preds != targets.to(self.device)).float().cpu().numpy()
                
                embs_list.append(emb)
                clusters_list.append(clusters)
                scores_list.append(scores)
                preds_list.append(preds.cpu().numpy())
                targets_list.append(targets.cpu().numpy())
                errors_list.append(errors)
        
        embs = np.concatenate(embs_list, axis=0)
        clusters = np.concatenate(clusters_list, axis=0)
        scores = np.concatenate(scores_list, axis=0)
        preds = np.concatenate(preds_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        errors = np.concatenate(errors_list, axis=0)
        
        if save_tensors:
            self.save_tensors(embs, clusters, scores, preds, targets, errors, name_save_file)
        if evaluate_clusters:
            self.evaluate_clusters(clusters, errors, name_save_file)
        return preds, targets, scores

    def evaluate_clusters(self, clusters: np.ndarray, errors: np.ndarray, name_save_file: str = "train"):
        counts_list, error_means_list, error_vars_list = [], [], []
        for i in range(self.n_cluster):
            idx = np.where(clusters == i)[0]
            count = idx.size
            counts_list.append(count)
            if count > 0:
                error_means = np.mean(errors[idx])
                error_means_list.append(error_means)
                error_vars = np.var(errors[idx])
                error_vars_list.append(error_vars)
            else:
                error_means_list.append(0.0)
                error_vars_list.append(0.0)
        metrics_dict = {
            "cluster_counts": counts_list,
            "cluster_means": error_means_list,
            "cluster_vars": error_vars_list
        }
        csv_path = os.path.join(self.results_folder, name_save_file + "_cluster_results.csv")
        results_df = pd.DataFrame(metrics_dict)
        if not os.path.isfile(csv_path):
            results_df.to_csv(csv_path, index=True, header=True)
        else:
            results_df.to_csv(csv_path, mode="a", index=True, header=False)
        print(f"Cluster evaluation results saved to {csv_path}")
        return metrics_dict

    def save_tensors(self, embs, clusters, scores, preds, targets, errors, name_save_file):
        save_path = os.path.join(self.results_folder, "tensors_" + name_save_file + ".npz")
        save_obj = {
            "embs": embs,
            "clusters": clusters,
            "scores": scores,
            "preds": preds,
            "targets": targets,
            "errors": errors,
        }
        np.savez(save_path, **save_obj)
        print(f"Tensors saved to {save_path}")

    # ... (rest of the class remains unchanged)


# class EmbeddingKMeans:
#     """
#     Clusters the embedding space (extracted from the model) using KMeans and computes the cluster-wise 
#     misclassification error along with a confidence interval based on a Chernoff/Hoeffding bound.

#     In addition, it computes empirical measures (variance and L1 distance) to quantify how constant the 
#     conditional probability of error is within each cluster.
    
#     For new inputs, the method extracts their embeddings, assigns them to a cluster, and returns the 
#     corresponding error uncertainty score.
#     """

#     def __init__(self, model, n_cluster = 10, clustering_space=None,  
#                 alpha=0.05, results_folder=None, model_name="resnet34",
#                 temperature = 1, clustering_method = "kmeans", magnitude=0, *args, **kwargs) -> None:
#         """
#         Clusters the embedding space using KMeans and computes cluster-level metrics.
#         """
#         self.model = model
#         self.device = next(model.parameters()).device
#         self.n_cluster = n_cluster
#         self.temperature = temperature
#         self.alpha = alpha
#         self.results_folder = results_folder
#         self.clustering_method = clustering_method
#         self.magnitude = magnitude
#         if model_name == "resnet34":
#             nodes_registry = {"last_layer": "view", "logits": "linear"}

#             if clustering_space == "probs":
#                 self.return_nodes = {nodes_registry["logits"]: "clustering_space"}
#                 feature_extractor = create_feature_extractor(model, self.return_nodes)
#                 self.feature_extractor = lambda x : F.softmax(feature_extractor(x)["clustering_space"] / self.temperature, dim=1)
            
#             elif clustering_space == "gini":
#                 self.return_nodes = {nodes_registry["logits"]: "clustering_space"}
#                 feature_extractor = create_feature_extractor(model, self.return_nodes)
#                 self.feature_extractor = lambda x : doctor(feature_extractor(x)["clustering_space"], self.temperature)

                
#             elif clustering_space in nodes_registry.keys(): 
#                 self.return_nodes = {nodes_registry[clustering_space]: "clustering_space"}
#                 feature_extractor = create_feature_extractor(model, self.return_nodes)
#                 self.feature_extractor = lambda x : feature_extractor(x)["clustering_space"]
#             else:
#                 raise ValueError(f"Nodes registry is not implemented for model {model}")
        
#         self.clustering = None

#     def _predict_uniform_clusters(self, embs: np.ndarray) -> np.ndarray:
#         """
#         Helper method to predict clusters uniformly by partitioning [0,1] into self.n_cluster bins.
#         If the embeddings are multi-dimensional, a simple projection (mean) is used.
#         """
#         # if embs.ndim > 1 and embs.shape[1] != 1:
#         #     embs_1d = embs.mean(axis=1)
#         # else:
#         #     embs_1d = embs.flatten()
#         clusters = np.floor(embs * self.n_cluster).astype(int)
#         # Handle edge case where an embedding equals 1
#         clusters[clusters == self.n_cluster] = self.n_cluster - 1
#         return clusters

#     def fit(self, train_dataloader: torch.utils.data.DataLoader, *args,  name_save_file: str = "train", save_tensors: bool = True, **kwargs) -> None:

#         """
#         Evaluates the model and clustering on the provided dataloader.
#         Extracts embeddings, predicts clusters and uncertainty scores,
#         collects predictions, targets, and error indicators, and saves the results as an NPZ file.
        
#         Optionally, it evaluates cluster-level metrics and saves them as CSV.
        
#         Returns:
#             preds, targets, scores (as torch tensors)
#         """


#         self.model.eval()
        
#         embs_list, clusters_list, scores_list = [], [], []
#         preds_list, targets_list, errors_list = [], [], []
        
#         with torch.no_grad():
#             for inputs, targets in tqdm(train_dataloader, total=len(train_dataloader), desc="Training"):
#                 inputs = inputs.to(self.device)
#                 if self.magnitude > 0 and self.clustering_method == "uniform":
#                     inputs = Variable(inputs, requires_grad=True)
#                     # compute perturbation
#                     gini =  self.feature_extractor(inputs)
#                     gini = torch.log(gini)
#                     gini.sum().backward()
#                     inputs = inputs - self.magnitude * torch.sign(-inputs.grad)
#                     # inputs = torch.clamp(inputs, 0, 1)
#                     inputs = Variable(inputs, requires_grad=False)
                
#                 # Extract embeddings from the designated layer.
#                 emb = self.feature_extractor(inputs)
#                 emb = emb.cpu().numpy()
                
#                 # Obtain model predictions and error indicators.
#                 logits = self.model(inputs)
#                 preds = logits.argmax(dim=1)
#                 errors = (preds != targets.to(self.device)).float().cpu().numpy()
                
#                 # Append results.
#                 embs_list.append(emb)
#                 preds_list.append(preds.cpu().numpy())
#                 targets_list.append(targets.cpu().numpy())
#                 errors_list.append(errors)

#         # Concatenate results from all batches.
#         embs = np.concatenate(embs_list, axis=0)
#         preds = np.concatenate(preds_list, axis=0)
#         targets = np.concatenate(targets_list, axis=0)
#         errors = np.concatenate(errors_list, axis=0)

#                 # Fit KMeans clustering on the embeddings.
#         if self.clustering_method == "kmeans":
#             self.clustering = KMeans(n_clusters=self.n_cluster, random_state=42)
#             # if embs.ndim ==1:
#             #     embs = embs.reshape(-1,1)
#             clusters = self.clustering.fit_predict(embs)
            
#         elif self.clustering_method == "uniform":
#             clusters = self._predict_uniform_clusters(embs)
   
#         self.fit_concentration(clusters, errors, name_save_file)
#         scores = np.array([self.cluster_intervals[i][1] for i in clusters])

#         # Save overall test scores as an NPZ file.
#         if save_tensors:
#             self.save_tensors(embs, clusters, scores, preds, targets, errors, name_save_file)

#         return preds, targets, scores
   
#     def fit_concentration(self, clusters: np.ndarray, errors: np.ndarray, name_save_file: str = "train") -> None:

#         self.cluster_counts, self.cluster_error_means, self.cluster_error_vars, self.cluster_intervals = [], [], [], []
        
#         for i in range(self.n_cluster):
#             idx = np.where(clusters == i)[0]
#             count = idx.size
#             self.cluster_counts.append(count)
#             if count > 0:
#                 error_mean = np.mean(errors[idx])
#                 self.cluster_error_means.append(error_mean)
#                 error_vars = np.var(errors[idx])
#                 self.cluster_error_vars.append(error_vars)
                
#                 half_width = np.sqrt(np.log(2 / self.alpha) / (2 * count))
#                 lower_bound = max(0.0, error_mean - half_width)
#                 upper_bound = min(1.0, error_mean + half_width)
#                 self.cluster_intervals.append((lower_bound, upper_bound))
#             else:
#                 self.cluster_error_means.append(0.0)
#                 self.cluster_error_vars.append(0.0)
#                 self.cluster_intervals.append((0, 1))
                
#         metrics_dict = {
#             "cluster_counts": self.cluster_counts,
#             "cluster_means": self.cluster_error_means,
#             "cluster_vars": self.cluster_error_vars,
#             "cluster_intervals": self.cluster_intervals
#         }
#         csv_path = os.path.join(self.results_folder, name_save_file + "_cluster_results.csv")
#         results_df = pd.DataFrame(metrics_dict)
#         if not os.path.isfile(csv_path):
#             results_df.to_csv(csv_path, index=False, header=True)
#         else:
#             results_df.to_csv(csv_path, mode="a", index=False, header=False)
#         print(f"Cluster evaluation results saved to {csv_path}")
#         # print(metrics_dict)
#         return 

#     def evaluate(
#                 self, dataloader: torch.utils.data.DataLoader, name_save_file: str = "test", evaluate_clusters: bool = True, save_tensors: bool = True
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Evaluates the model and clustering on the provided dataloader.
#         Extracts embeddings, predicts clusters and uncertainty scores,
#         collects predictions, targets, and error indicators, and saves the results as an NPZ file.

#         Optionally, it evaluates cluster-level metrics and saves them as CSV.

#         Returns:
#             preds, targets, scores (as torch tensors)
#         """
#         self.model.eval()

#         embs_list, clusters_list, scores_list = [], [], []
#         preds_list, targets_list, errors_list = [], [], []

#         with torch.no_grad():
#             for inputs, targets in tqdm(dataloader, total=len(dataloader), desc="Evaluating"):
#                 inputs = inputs.to(self.device)
#                 # Extract embeddings from the designated layer.
#                 if self.magnitude > 0 and self.clustering_method == "uniform":
#                     inputs = Variable(inputs, requires_grad=True)
#                     # compute perturbation
#                     gini =  self.feature_extractor(inputs)
#                     gini = torch.log(gini)
#                     gini.sum().backward()
#                     inputs = inputs - self.magnitude * torch.sign(-inputs.grad)
#                     # inputs = torch.clamp(inputs, 0, 1)
#                     inputs = Variable(inputs, requires_grad=False)

#                 emb = self.feature_extractor(inputs)
#                 emb = emb.cpu().numpy()
                
#                 # Predict clusters using the chosen clustering method.
#                 if self.clustering_method == "kmeans":
#                     clusters = self.clustering.predict(emb)
#                 elif self.clustering_method == "uniform":
#                     clusters = self._predict_uniform_clusters(emb)
#                 else:
#                     raise ValueError("Unsupported clustering method: choose 'kmeans' or 'uniform'.")
#                 # Compute uncertainty scores using the upper bound of the confidence interval.
#                 scores = np.array([self.cluster_intervals[i][1] for i in clusters])
                
#                 # Obtain model predictions and error indicators.
#                 logits = self.model(inputs)
#                 preds = logits.argmax(dim=1)
#                 errors = (preds != targets.to(self.device)).float().cpu().numpy()
                
#                 # Append results.
#                 embs_list.append(emb)
#                 clusters_list.append(clusters)
#                 scores_list.append(scores)
#                 preds_list.append(preds.cpu().numpy())
#                 targets_list.append(targets.cpu().numpy())
#                 errors_list.append(errors)

#         # Concatenate results from all batches.
#         embs = np.concatenate(embs_list, axis=0)
#         clusters = np.concatenate(clusters_list, axis=0)
#         scores = np.concatenate(scores_list, axis=0)
#         preds = np.concatenate(preds_list, axis=0)
#         targets = np.concatenate(targets_list, axis=0)
#         errors = np.concatenate(errors_list, axis=0)

#         # Save overall test scores as an NPZ file.
#         if save_tensors:
#             self.save_tensors(embs, clusters, scores, preds, targets, errors, name_save_file)

#         # Optionally evaluate cluster-level metrics and save to CSV.
#         if evaluate_clusters:
#             self.evaluate_clusters(clusters, errors, name_save_file)
                    
#         # Return predictions, targets, and scores as torch tensors.
#         # preds = torch.from_numpy(preds)
#         # targets = torch.from_numpy(targets)
#         # scores = torch.from_numpy(scores)
#         return preds, targets, scores


#     def evaluate_clusters(self, clusters: np.ndarray, errors: np.ndarray, name_save_file: str = "train") -> Dict[str, List[float]]:
        
#         counts_list, error_means_list, error_vars_list = [], [], []
        
#         for i in range(self.n_cluster):
#             idx = np.where(clusters == i)[0]
#             count = idx.size
#             counts_list.append(count)
#             if count > 0:
#                 error_means = np.mean(errors[idx])
#                 error_means_list.append(error_means)
#                 error_vars = np.var(errors[idx])
#                 error_vars_list.append(error_vars)
#             else:
#                 error_means_list.append(0.0)
#                 error_vars_list.append(0.0)
                
#         metrics_dict = {
#             "cluster_counts": counts_list,
#             "cluster_means": error_means_list,
#             "cluster_vars": error_vars_list
#         }
#         csv_path = os.path.join(self.results_folder, name_save_file + "_cluster_results.csv")
#         results_df = pd.DataFrame(metrics_dict)
#         if not os.path.isfile(csv_path):
#             results_df.to_csv(csv_path, index=True, header=True)
#         else:
#             raise ValueError("dd")
#             results_df.to_csv(csv_path, mode="a", index=True, header=False)
#         print(f"Cluster evaluation results saved to {csv_path}")
#         return metrics_dict
    
#     def save_tensors(self, embs, clusters, scores, preds, targets, errors, name_save_file):
#                 # Save overall test scores as an NPZ file.
#         save_path = os.path.join(self.results_folder, "tensors_" + name_save_file + ".npz")
#         save_obj = {
#             "embs": embs,
#             "clusters": clusters,
#             "scores": scores,
#             "preds": preds,
#             "targets": targets,
#             "errors": errors,
#         }
#         np.savez(save_path, **save_obj)
#         print(f"Tensors saved to {save_path}")

        


class Wrapper:
    def __init__(self, method, model, *args, **kwargs):
        self.method = method
        self.model = model
        self.device = next(model.parameters()).device

    def fit(self, train_dataloader, val_dataloader, **kwargs):
        if hasattr(self.method, "fit"):
            return self.method.fit(train_dataloader, val_dataloader, **kwargs)
        else:
            return self
            

    def __call__(self, x):
        return self.method(x)
    
    def evaluate(self, test_dataloader, *args, **kwargs):
        if hasattr(self.method, "evaluate"):
            return self.method.evaluate(test_dataloader, *args, **kwargs)
        else:
            test_preds, test_targets, test_scores = [], [], []

            for inputs, targets in tqdm(test_dataloader, total=len(test_dataloader)):
                inputs = inputs.to(self.device)
                with torch.no_grad():
                  
                    logits = self.model(inputs)
                    scores = self.method(logits)
                    pred = torch.argmax(logits, dim=1)

                test_preds.append(pred.cpu().numpy())
                test_targets.append(targets.cpu().numpy())
                test_scores.append(scores.cpu().numpy())

            test_preds = np.concat(test_preds, axis=0)
            test_targets = np.concat(test_targets, axis=0)
            test_scores = np.concat(test_scores, axis=0)
        return test_preds, test_targets, test_scores


    def export_matrix(self):
        return self.method.export_matrix()






def get_method(method_name: str, model, *args, **kwargs) -> Wrapper:
    if method_name == "doctor":
        return Wrapper(partial(doctor, *args, **kwargs), model)
    if method_name == "odin":
        return Wrapper(partial(odin, *args, **kwargs))
    if method_name == "msp":
        return Wrapper(msp)
    if method_name == "metric_lagrange":
        return Wrapper(MetricLearningLagrange(*args, **kwargs))
    if method_name == "mlp":
        return Wrapper(MLPTrainer(model, kwargs["dataset"]["num_classes"], *args, **kwargs), model)
    if method_name == "mc_dropout":
        return Wrapper(entropy)
    if method_name == "ensemble":
        return Wrapper(msp)
    if method_name == "bayes":
        return Wrapper(BayesDetector(model, *args, **kwargs), model)
    elif method_name == "conformal":
        return Wrapper(EmbeddingKMeans(model, *args, **kwargs), model)
    raise ValueError(f"Method {method_name} not supported")
