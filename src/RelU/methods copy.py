from functools import partial
from typing import Any, Tuple, Dict, List
import torch
import torch.utils.data
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
import copy
from src.utils.eval import evaluate_classification
import numpy as np
import pandas as pd


def g(logits, temperature=1.0):
    return torch.sum(torch.softmax(logits / temperature, dim=1) ** 2, dim=1)


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
            acc, roc_auc, fpr, aurc = evaluate_classification(preds, targets, scores)
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

class EmbeddingKMeans:
    """
    Clusters the embedding space (extracted from the model) using KMeans and computes the cluster-wise 
    misclassification error along with a confidence interval based on a Chernoff/Hoeffding bound.

    In addition, it computes empirical measures (variance and L1 distance) to quantify how constant the 
    conditional probability of error is within each cluster.
    
    For new inputs, the method extracts their embeddings, assigns them to a cluster, and returns the 
    corresponding error uncertainty score.
    """

    def __init__(self, model, n_cluster = 10, clustering_space=None,  
                alpha=0.05, results_folder=None, model_name="resnet34",
                temperature = 1,*args, **kwargs) -> None:
        """
        Clusters the embedding space using KMeans and computes cluster-level metrics.
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.n_cluster = n_cluster
        self.temperature = temperature
        self.alpha = alpha
        self.results_folder = results_folder
        if model_name == "resnet34":
            nodes_registry = {"last_layer": "view", "logits": "linear"}

            if clustering_space == "probs":
                self.return_nodes = {nodes_registry["logits"]: "clustering_space"}
                feature_extractor = create_feature_extractor(model, self.return_nodes)
                self.feature_extractor = lambda x : F.softmax(feature_extractor(x)["clustering_space"] / self.temperature, dim=1)
            
            elif clustering_space == "gini":
                self.return_nodes = {nodes_registry["logits"]: "clustering_space"}
                feature_extractor = create_feature_extractor(model, self.return_nodes)
                self.feature_extractor = lambda x : doctor(feature_extractor(x)["clustering_space"], self.temperature)

                
            elif clustering_space in nodes_registry.keys(): 
                self.return_nodes = {nodes_registry[clustering_space]: "clustering_space"}
                feature_extractor = create_feature_extractor(model, self.return_nodes)
                self.feature_extractor = lambda x : feature_extractor(x)["clustering_space"]
            else:
                raise ValueError(f"Nodes registry is not implemented for model {model}")
        
        self.clustering = None

    def fit(self, train_dataloader: torch.utils.data.DataLoader, *args,  name_save_file: str = "train", save_tensors: bool = False, **kwargs) -> None:

        """
        Evaluates the model and clustering on the provided dataloader.
        Extracts embeddings, predicts clusters and uncertainty scores,
        collects predictions, targets, and error indicators, and saves the results as an NPZ file.
        
        Optionally, it evaluates cluster-level metrics and saves them as CSV.
        
        Returns:
            preds, targets, scores (as torch tensors)
        """


        self.model.eval()
        
        embs_list, clusters_list, scores_list = [], [], []
        preds_list, targets_list, errors_list = [], [], []
        
        with torch.no_grad():
            for inputs, targets in tqdm(train_dataloader, total=len(train_dataloader), desc="Training"):
                inputs = inputs.to(self.device)
                # Extract embeddings from the designated layer.
                emb = self.feature_extractor(inputs)
                emb = emb.cpu().numpy()
                
                # Obtain model predictions and error indicators.
                logits = self.model(inputs)
                preds = logits.argmax(dim=1)
                errors = (preds != targets.to(self.device)).float().cpu().numpy()
                
                # Append results.
                embs_list.append(emb)
                preds_list.append(preds.cpu().numpy())
                targets_list.append(targets.cpu().numpy())
                errors_list.append(errors)

        # Concatenate results from all batches.
        embs = np.concatenate(embs_list, axis=0)
        preds = np.concatenate(preds_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        errors = np.concatenate(errors_list, axis=0)

                # Fit KMeans clustering on the embeddings.
        if self.clustering_method == "kmeans":
            self.clustering = KMeans(n_clusters=self.n_cluster, random_state=42)
        elif self.clustering_method == "trivial":
            clusters = self.clustering.fit_predict(embs)
        self.fit_concentration(clusters, errors, name_save_file)
        scores = np.array([self.cluster_intervals[i][1] for i in clusters])

        # Save overall test scores as an NPZ file.
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
            "cluster_vars": self.cluster_error_vars
        }
        csv_path = os.path.join(self.results_folder, name_save_file + "_cluster_results.csv")
        results_df = pd.DataFrame(metrics_dict)
        if not os.path.isfile(csv_path):
            results_df.to_csv(csv_path, index=False, header=True)
        else:
            results_df.to_csv(csv_path, mode="a", index=False, header=False)
        print(f"Cluster evaluation results saved to {csv_path}")
        # print(metrics_dict)
        return 

    def evaluate(
                self, dataloader: torch.utils.data.DataLoader, name_save_file: str = "test", evaluate_clusters: bool = True, save_tensors: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates the model and clustering on the provided dataloader.
        Extracts embeddings, predicts clusters and uncertainty scores,
        collects predictions, targets, and error indicators, and saves the results as an NPZ file.

        Optionally, it evaluates cluster-level metrics and saves them as CSV.

        Returns:
            preds, targets, scores (as torch tensors)
        """
        self.model.eval()

        embs_list, clusters_list, scores_list = [], [], []
        preds_list, targets_list, errors_list = [], [], []

        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, total=len(dataloader), desc="Evaluating"):
                inputs = inputs.to(self.device)
                # Extract embeddings from the designated layer.
                emb = self.feature_extractor(inputs)
                emb = emb.cpu().numpy()
                
                # Predict clusters using KMeans.
                clusters = self.clustering.predict(emb)
                # Compute uncertainty scores using the upper bound of the confidence interval.
                scores = np.array([self.cluster_intervals[i][1] for i in clusters])
                
                # Obtain model predictions and error indicators.
                logits = self.model(inputs)
                preds = logits.argmax(dim=1)
                errors = (preds != targets.to(self.device)).float().cpu().numpy()
                
                # Append results.
                embs_list.append(emb)
                clusters_list.append(clusters)
                scores_list.append(scores)
                preds_list.append(preds.cpu().numpy())
                targets_list.append(targets.cpu().numpy())
                errors_list.append(errors)

        # Concatenate results from all batches.
        embs = np.concatenate(embs_list, axis=0)
        clusters = np.concatenate(clusters_list, axis=0)
        scores = np.concatenate(scores_list, axis=0)
        preds = np.concatenate(preds_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        errors = np.concatenate(errors_list, axis=0)

        # Save overall test scores as an NPZ file.
        if save_tensors:
            self.save_tensors(embs, clusters, scores, preds, targets, errors, name_save_file)

        # Optionally evaluate cluster-level metrics and save to CSV.
        if evaluate_clusters:
            self.evaluate_clusters(clusters, errors, name_save_file)
                    
        # Return predictions, targets, and scores as torch tensors.
        # preds = torch.from_numpy(preds)
        # targets = torch.from_numpy(targets)
        # scores = torch.from_numpy(scores)
        return preds, targets, scores


    def evaluate_clusters(self, clusters: np.ndarray, errors: np.ndarray, name_save_file: str = "train") -> Dict[str, List[float]]:
        
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
            raise ValueError("dd")
            results_df.to_csv(csv_path, mode="a", index=True, header=False)
        print(f"Cluster evaluation results saved to {csv_path}")
        return metrics_dict
    
    def save_tensors(self, embs, clusters, scores, preds, targets, errors, name_save_file):
                # Save overall test scores as an NPZ file.
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

        


class Wrapper:
    def __init__(self, method, *args, **kwargs):
        self.method = method

    def fit(self, train_dataloader, val_dataloader, **kwargs):
        if hasattr(self.method, "fit"):
            print("kwargs", kwargs)
            return self.method.fit(train_dataloader, val_dataloader, **kwargs)
        else:
            raise ValueError(f"fit method not implemented")
            

    def __call__(self, x):
        return self.method(x)
    
    def evaluate(self, *args, **kwargs):
        return self.method.evaluate(*args, **kwargs)

    def export_matrix(self):
        return self.method.export_matrix()


def get_method(method_name: str, *args, **kwargs) -> Wrapper:
    if method_name == "doctor":
        return Wrapper(partial(doctor, *args, **kwargs))
    if method_name == "odin":
        return Wrapper(partial(odin, *args, **kwargs))
    if method_name == "msp":
        return Wrapper(msp)
    if method_name == "metric_lagrange":
        return Wrapper(MetricLearningLagrange(*args, **kwargs))
    if method_name == "mlp":
        return Wrapper(MLPTrainer(*args, **kwargs))
    if method_name == "mc_dropout":
        return Wrapper(entropy)
    if method_name == "ensemble":
        return Wrapper(msp)
    elif method_name == "conformal":
        return Wrapper(EmbeddingKMeans(*args, **kwargs))
    raise ValueError(f"Method {method_name} not supported")
