import torch
import numpy as np
from torch.distributions import MultivariateNormal
from tqdm import tqdm
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from torchvision.models.feature_extraction import create_feature_extractor
import umap
import os
import joblib
from torch.autograd import Variable
from sklearn.decomposition import PCA
from code.utils.helper import _prepare_config_for_results, append_results_to_file
from code.utils.clustering.models import BregmanHard
from code.utils.eval import  MultiDetectorEvaluator
from code.utils.clustering.divergences import (
    euclidean,
    kullback_leibler,
    itakura_saito,
    alpha_divergence_factory,
)
from code.utils.clustering.kmeans import KMeans as TorchKMeans
from code.utils.clustering.soft_kmeans import SoftKMeans as TorchSoftKMeans

import pandas as pd
from code.utils.metrics import compute_all_metrics
from code.utils.detection.registry import register_detector
from sklearn.model_selection import StratifiedKFold

def gini(logits, temperature=1.0, normalize=False):
    g =torch.sum(torch.softmax(logits / temperature, dim=1) ** 2, dim=1, keepdim=True)
    if normalize:
        return  (1 - g) / g 
    else:
        return 1 - g




class HyperparameterSearch:
    def __init__(
            self, 
            detectors, 
            model, 
            train_loader,
            val_loader,
            device,
            base_config, 
            list_configs=None, 
            metric='fpr', 
            result_folder="results/"
            ):

        """
        Args:
            detectors (list): List of detector instances.
        """
        self.detectors = detectors
        self.model = model
        self.device = device
        self.base_config = base_config
        self.method_name = base_config.get("method_name")
        self.list_configs = list_configs
        self.n_splits = base_config["data"]["n_splits"]
        self.result_folder = result_folder
        self.n_epochs = base_config["data"]["n_epochs"]
        self.root = f"storage_latent/{base_config['data']['name']}_{base_config['model']['name']}_r-{base_config['data']['r']}_seed-split-{base_config['data']['seed_split']}/"

        if  (base_config['method_name'] == "clustering") & (base_config['clustering']['space'] == "classifier"):
            self.latent_path = self.root + f"{base_config['clustering']['space']}_train_n-epochs{self.n_epochs}_transform-{base_config['data']['transform']}.pt"
        else:
            self.latent_path = self.root + f"logits_train_n-epochs{self.n_epochs}_transform-{base_config['data']['transform']}.pt"
     

        self.metric = metric
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.evaluator_train = MultiDetectorEvaluator(
            self.model, self.train_loader, device=self.device, suffix="train", base_config=self.base_config,
          
            )
        self.evaluator_test = MultiDetectorEvaluator(
            self.model, self.val_loader, device=self.device, suffix="val", base_config=self.base_config,
        )
        self.evaluator_cross = MultiDetectorEvaluator(
            self.model, self.val_loader, device=self.device, suffix="cross", base_config=self.base_config,
        )

        self.run()

    def get_values(self, train_dataloader):

        

        # all_model_preds = []
        if os.path.exists(self.latent_path):
            pkg = torch.load(self.latent_path, map_location="cpu")
            all_logits = pkg["logits"].to(torch.float32)        # (N, C)
            all_labels = pkg["labels"]              # (N,)
            all_model_preds  = pkg["model_preds"]# (N,)
            all_detector_labels = (all_model_preds != all_labels).float()
        
        else:
                        
            # def _invert_normalize(x, mean, std):
            #     """Invert normalization on a BCHW tensor in-place-safe way."""
            #     if mean is None or std is None:
            #         return x
            #     mean = torch.tensor(mean, device=x.device).view(1, -1, 1, 1)
            #     std  = torch.tensor(std,  device=x.device).view(1, -1, 1, 1)
            #     return x * std + mean

            # def save_aug_grid(inputs, save_path, mean=None, std=None, nrow=8, clamp=True):
            #     """
            #     inputs: tensor [B,C,H,W] as it comes from the DataLoader (already augmented/normalized)
            #     Saves a PNG grid after inverting Normalize.
            #     """
            #     from torchvision import transforms, utils as vutils
            #     from PIL import Image
            #     x = inputs.detach().cpu()
            #     if mean is not None and std is not None:
            #         x = _invert_normalize(x, mean, std).cpu()
            #     if clamp:
            #         x = torch.clamp(x, 0.0, 1.0)  # safe if transforms put values in [0,1] after inverse
            #     grid = vutils.make_grid(x, nrow=nrow, padding=2)  # [3,H',W'] in [0,1]
            #     nd = (grid.numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            #     Image.fromarray(nd).save(save_path)
            self.model.to(self.device)
            self.model.eval()

            all_model_preds = []
            all_labels = []
            all_logits = []
            # os.makedirs("debug_aug", exist_ok=True)
            for epoch in range(self.n_epochs):
                with torch.no_grad():
                    for batch, (inputs, targets) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Getting Training Logits", disable=False):

                        inputs = inputs.to(self.device)
                        # targets = targets.to(self.device)
                    
                        logits = self.model(inputs).cpu()  # logits: [batch_size, num_classes]
                        model_preds = torch.argmax(logits, dim=1)

                        # detector_labels = (model_preds != targets).float()
                        # # all_model_preds.append(model_preds)
                        # all_detector_labels.append(detector_labels)
                        all_logits.append(logits)
                        all_labels.append(targets.cpu())
                        all_model_preds.append(model_preds)

            
            
            # all_model_preds = torch.cat(all_model_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            all_model_preds = torch.cat(all_model_preds, dim=0)
            all_detector_labels = (all_model_preds != all_labels).float()
            all_logits = torch.cat(all_logits, dim=0)

            # AFTER (robust)
            parent = os.path.dirname(self.latent_path)
            os.makedirs(parent, exist_ok=True)

            tmp = self.latent_path + ".tmp"
            torch.save(
                {
                    "logits": all_logits.cpu(),     # compact on disk
                    "labels": all_labels.cpu().to(torch.int64),
                    "model_preds": all_model_preds.cpu().to(torch.int64),
                },
                tmp,
            )
            os.replace(tmp, self.latent_path)  # atomic rename
            
        self.values = {"logits": all_logits, "detector_labels": all_detector_labels}


    def prepare_configs_group(self):
        groups = {}
        order = []
        
        for i, cfg in enumerate(self.list_configs):
            key = tuple((k, v) for k, v in cfg[self.method_name].items() if k != "magnitude")
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(i)
        self.config_groups = [groups[k] for k in order]

            

    def aggregate_cv_over_folds(self, per_fold_results: list[list[pd.DataFrame]]) -> list[pd.DataFrame]:
        """
        Args
        ----
        per_fold_results : list over folds [
            # fold k (1..K)
            [ df_det0_k, df_det1_k, ..., df_det{D-1}_k ]   # each df is 1 row with config + metrics_{cross_fold-k}
        ]

        Returns
        -------
        per_detector_agg : list of length D
            Each element is a 1-row DataFrame with config columns first,
            then {metric}_cross_mean and {metric}_cross_std.
        """
        import re

        METRICS = ["fpr","tpr","thr","roc_auc","model_acc","aurc","aupr_err","aupr_success"]

        n_folds = len(per_fold_results)
        D = len(per_fold_results[0]) if n_folds > 0 else 0
        

        per_detector_agg = []

        for det_idx in range(D):
            # Stack the 1-row DataFrames for this detector across folds (rows become n_folds)
            df_all = pd.concat([per_fold_results[k][det_idx] for k in range(n_folds)],
                            axis=0, ignore_index=True)

            # Identify config columns (everything that is NOT suffixed with _val_cross_fold-<num>)
            fold_suffix_re = re.compile(r"_val_cross_fold-\d+$")
            cfg_cols = [c for c in df_all.columns if not fold_suffix_re.search(c)]

            # Take config from the first row (identical across folds by construction)
            out = df_all[cfg_cols].iloc[[0]].copy()   # keep as 1-row DataFrame

            # For each metric, collect all fold-specific columns and aggregate
            for m in METRICS:
                pat = re.compile(rf"^{re.escape(m)}_val_cross_fold-\d+$")
                mcols = [c for c in df_all.columns if pat.match(c)]
                if not mcols:
                    out[f"{m}_val_cross_mean"] = np.nan
                    out[f"{m}_val_cross_std"]  = np.nan
                    continue

              
                stacked = df_all[mcols].stack(future_stack=True)  # new implementation, no dropna
                stacked = stacked.dropna()  
                vals = pd.to_numeric(stacked, errors="coerce").to_numpy()
                cnt = np.isfinite(vals).sum()
                mean = float(np.nanmean(vals)) if cnt else np.nan
                std  = float(np.nanstd(vals, ddof=1)) if cnt > 1 else 0.0

                out[f"{m}_val_cross_mean"] = mean
                out[f"{m}_val_cross_std"]  = std

            per_detector_agg.append(out)

        return per_detector_agg

    def cross_validation_magnitude(self):
        

        list_results = []
        list_magnitudes = [config[self.method_name]["magnitude"] for config in self.list_configs]
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=False)
        

         

        # for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros_like(self.values["detector_labels"]), self.values["detector_labels"]), 1):
        for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros_like(self.values["detector_labels"]), self.values["detector_labels"]), 1):
            
            
            val_scores = {i : np.zeros(len(va_idx)) for i in range(len(self.list_configs))}
            val_loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(self.train_loader.dataset, va_idx),
                batch_size=self.train_loader.batch_size, shuffle=False,
                num_workers=self.train_loader.num_workers, pin_memory=True
            )

            logits_train = self.values["logits"][tr_idx].to(self.device)
            detector_labels_train = self.values["detector_labels"][tr_idx].to(self.device)
            detector_labels_val = self.values["detector_labels"][va_idx].cpu().numpy()

            for group in tqdm(self.config_groups, total=len(self.config_groups), desc="Group Cross validation", disable=False):

                list_magnitudes = [self.list_configs[cfg_idx][self.method_name]["magnitude"] for cfg_idx in group]
                proto_dec = self.detectors[group[0]]
                proto_dec.fit(logits=logits_train, detector_labels=detector_labels_train)

                write = 0
                for inputs, _ in val_loader:
                  
     
                    bs = inputs.size(0)
                    inputs = inputs.to(self.device).detach().requires_grad_(True)
                    logits = self.model(inputs)
                    score = proto_dec(logits=logits)
                    loss = torch.log(score + 1e-12).sum()
                    grad_inputs, = torch.autograd.grad(loss, inputs, retain_graph=False, create_graph=False)
                    grad_sign = grad_inputs.sign()
                    with torch.no_grad():
                        list_adv_inputs = [inputs + magnitude * grad_sign for magnitude in list_magnitudes]
                    with torch.inference_mode():
                        list_logits_adv = [self.model(adv) for adv in list_adv_inputs]
                        scores_adv = [proto_dec(logits=logits_adv) for logits_adv in list_logits_adv]

                    for cfg_idx, scores in zip(group, scores_adv):
                        val_scores[cfg_idx][write:write+bs] = scores.cpu().numpy()
                    write += bs
            # print("val_scores[cfg_idx]", val_scores[0][:10])
            
            list_results.append(self.evaluator_cross.evaluate(
                list_configs=self.list_configs,
                all_scores= [val_scores[i] for i in range(len(self.list_configs))],
                detector_labels=detector_labels_val,
                suffix=f"val_cross_fold-{fold}"))
        
        list_results = self.aggregate_cv_over_folds(list_results)

        cross_val_results = pd.concat(list_results, axis=0)
        self.crossval_results = cross_val_results


        self.best_idx = np.argmin([np.mean(res[f"{self.metric}_val_cross_mean"].values) for res in list_results])
        self.best_config = self.list_configs[self.best_idx]
        self.best_result = list_results[self.best_idx]
        print(f"Best results: {self.best_result[[col for col in self.best_result.columns if col.startswith(self.method_name)]]}")
        print(f"Best result ({self.metric}): {self.best_result[f'{self.metric}_val_cross_mean'].values}")


        self.save_results(
            result_file=os.path.join(self.result_folder, "hyperparams_results.csv"),
            results=cross_val_results
            )



    def cross_validation(self):

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=False)

        

        # Optional: precompute/cached features/logits here to speed up, if your detectors support it.
        list_results = []

        # if self.method_name == "clustering":
            
        #     train_results = []
        #     for dec_idx, dec in tqdm(enumerate(self.detectors),total=len(self.detectors), desc="Cross validation", disable=False):
        #         for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros_like(self.values["detector_labels"]), self.values["detector_labels"]), 1):

        #             dec.fit(logits=self.values["logits"][tr_idx].to(dec.device), detector_labels=self.values["detector_labels"][tr_idx].to(dec.device))
        #             scores = dec(logits=self.values["logits"][tr_idx].to(dec.device))
        #             self.evaluator_cross.scores = {
        #                 "scores" : scores,
        #                 "detector_labels" : self.values["detector_labels"][tr_idx].to(dec.device)
        #             }
        #             train_results.append(self.evaluator_cross.evaluate([dec], [self.list_configs[dec_idx]])[0])


        




        for dec_idx, dec in tqdm(enumerate(self.detectors),total=len(self.detectors), desc="Cross validation", disable=False):

            tr_metrics = {metric: [] for metric in ["fpr", "tpr", "thr", "roc_auc", "model_acc", "aurc", "aupr_err", "aupr_success"]}
            val_metrics = {metric: [] for metric in ["fpr", "tpr", "thr", "roc_auc", "model_acc", "aurc", "aupr_err", "aupr_success"]}
            for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros_like(self.values["detector_labels"]), self.values["detector_labels"]), 1):

                dec.fit(logits=self.values["logits"][tr_idx], detector_labels=self.values["detector_labels"][tr_idx])
                # Evaluate on validation set

                train_conf = dec(logits=self.values["logits"][tr_idx])
                val_conf = dec(logits=self.values["logits"][va_idx])

                if self.method_name == "metric_learning":

                    self.evaluator_cross

                for split in ["tr_cross", "val_cross"]:
                    if split == "tr_cross":
                        conf = train_conf
                        detector_labels = self.values["detector_labels"][tr_idx]
                        metrics = tr_metrics
                    else:
                        conf = val_conf
                        detector_labels = self.values["detector_labels"][va_idx]
                        metrics = val_metrics
                    fpr, tpr, thr, auroc, accuracy, aurc_value, aupr_in, aupr_out = compute_all_metrics(
                        conf=conf.cpu().numpy(),
                        detector_labels=detector_labels.cpu().numpy(),
                    )
                    metrics["fpr"].append(fpr)
                    metrics["tpr"].append(tpr)
                    metrics["thr"].append(thr)
                    metrics["roc_auc"].append(auroc)
                    metrics["model_acc"].append(accuracy)
                    metrics["aurc"].append(aurc_value)
                    metrics["aupr_err"].append(aupr_in)
                    metrics["aupr_success"].append(aupr_out)

            results = pd.DataFrame([{
                "fpr_tr_cross": np.mean(tr_metrics["fpr"]),
                "fpr_tr_cross_std": np.std(tr_metrics["fpr"]),
                "tpr_tr_cross": np.mean(tr_metrics["tpr"]),
                "tpr_tr_cross_std": np.std(tr_metrics["tpr"]),
                "thr_tr_cross": np.mean(tr_metrics["thr"]),
                "thr_tr_cross_std": np.std(tr_metrics["thr"]),
                "roc_auc_tr_cross": np.mean(tr_metrics["roc_auc"]),
                "roc_auc_tr_cross_std": np.std(tr_metrics["roc_auc"]),
                "model_acc_tr_cross": np.mean(tr_metrics["model_acc"]),
                "model_acc_tr_cross_std": np.std(tr_metrics["model_acc"]),
                "aurc_tr_cross": np.mean(tr_metrics["aurc"]),
                "aurc_tr_cross_std": np.std(tr_metrics["aurc"]),
                "aupr_err_tr_cross": np.mean(tr_metrics["aupr_err"]),
                "aupr_err_tr_cross_std": np.std(tr_metrics["aupr_err"]),
                "aupr_success_tr_cross": np.mean(tr_metrics["aupr_success"]),
                "aupr_success_tr_cross_std": np.std(tr_metrics["aupr_success"]),
                "fpr_val_cross": np.mean(val_metrics["fpr"]),
                "fpr_val_cross_std": np.std(val_metrics["fpr"]),
                "tpr_val_cross": np.mean(val_metrics["tpr"]),
                "tpr_val_cross_std": np.std(val_metrics["tpr"]),
                "thr_val_cross": np.mean(val_metrics["thr"]),
                "thr_val_cross_std": np.std(val_metrics["thr"]),
                "roc_auc_val_cross": np.mean(val_metrics["roc_auc"]),
                "roc_auc_val_cross_std": np.std(val_metrics["roc_auc"]),
                "model_acc_val_cross": np.mean(val_metrics["model_acc"]),
                "model_acc_val_cross_std": np.std(val_metrics["model_acc"]),
                "aurc_val_cross": np.mean(val_metrics["aurc"]),
                "aurc_val_cross_std": np.std(val_metrics["aurc"]),
                "aupr_err_val_cross": np.mean(val_metrics["aupr_err"]),
                "aupr_err_val_cross_std": np.std(val_metrics["aupr_err"]),
                "aupr_success_val_cross": np.mean(val_metrics["aupr_success"]),
                "aupr_success_val_cross_std": np.std(val_metrics["aupr_success"]),
            }])
            
            config = _prepare_config_for_results(self.list_configs[dec_idx])
            config = pd.json_normalize(config, sep="_")
            results = pd.concat([config, results], axis=1)
            list_results.append(results)
        
        cross_val_results = pd.concat(list_results, axis=0)
        self.crossval_results = cross_val_results


        self.best_idx = np.argmin([np.mean(res[f"{self.metric}_val_cross"].values) for res in list_results])
        self.best_config = self.list_configs[self.best_idx]
        self.best_result = list_results[self.best_idx]
        print(f"Best results: {self.best_result[[col for col in self.best_result.columns if col.startswith(self.method_name)]]}")
        print(f"Best result ({self.metric}): {self.best_result[f'{self.metric}_val_cross'].values}")


        self.save_results(
            result_file=os.path.join(self.result_folder, "hyperparams_results.csv"),
            results=cross_val_results
            )


    def save_results(self, result_file, results):

        print(f"Saving results to {result_file}")
        os.makedirs(os.path.dirname(result_file), exist_ok=True)

        if not os.path.isfile(result_file):
            results.to_csv(result_file, header=True, index=False)
        else:
            print(f"Results already exist at {result_file}")
            result_file = result_file.replace(".csv", "_append.csv")
            results.to_csv(result_file, header=True, index=False)


    def search_no_fit(self):

        list_results = self.evaluator_train.evaluate(self.list_configs, self.detectors)
        hyperparam_results = pd.concat(list_results, axis=0)
                   

        self.best_idx = np.argmin([np.mean(res[f"{self.metric}_train"].values) for res in list_results])
        self.best_config = self.list_configs[self.best_idx]
        self.best_result = list_results[self.best_idx]

        print(f"Best Configs: {self.best_result[[col for col in self.best_result.columns if col.startswith(self.method_name)]]}")
        print(f"Best result ({self.metric}): {self.best_result[f'{self.metric}_train'].values}")
        self.train_results = self.best_result

        self.save_results(
            result_file=os.path.join(self.result_folder, "hyperparams_results.csv"),
            results=hyperparam_results
            )


    def run(self):
        """
        Fit all detectors on the training data.
        
        Args:
            train_dataloader (DataLoader): DataLoader for the training data.
        """
        print("Collecting values on training data")
        import time
        t0 = time.time()
        self.get_values(self.train_loader)
        t1 = time.time()
        print(f"Total time: {t1 - t0:.2f} seconds")
        

        if (self.method_name in ["clustering", "random_forest"]) & (self.n_splits >= 2):
        
            print("Performing cross-validation")
            self.cross_validation()

        elif self.method_name == "metric_learning":
            print("Performing cross-validation with magnitude search")
            self.prepare_configs_group()
            self.cross_validation_magnitude()

        elif self.method_name in ["max_prob", "gini"]:
            print("Performing hyperparameter search without fitting")
            t0 = time.time()
            self.search_no_fit()
            t1 = time.time()
            print(f"Total time: {t1 - t0:.2f} seconds")
        else:
            print("No hyperparameter search, using the first detector")
            self.best_idx = 0
            self.best_config = self.list_configs[self.best_idx]

        self.best_detector = self.detectors[self.best_idx]

        if hasattr(self.best_detector, 'fit'):
            print("Fitting best detector on full training data")
            t0 = time.time()
            self.best_detector.fit(
                logits=self.values["logits"].to(self.best_detector.device), 
                detector_labels=self.values["detector_labels"].to(self.best_detector.device)
                )
            t1 = time.time()
            print(f"Total time: {t1 - t0:.2f} seconds")
            print("Evaluating best detector on training data")
            self.train_results = self.evaluator_train.evaluate([self.best_config], [self.best_detector])[0]
            print(f"Train result ({self.metric}): {self.train_results[f'{self.metric}_train'].values}")
        
        print("Evaluating best detector on validation data")
        t0 = time.time()
        self.val_results = self.evaluator_test.evaluate([self.best_config], [self.best_detector])[0]
        t1 = time.time()
        print(f"Val result ({self.metric}): {self.val_results[f'{self.metric}_val'].values}")
        print(f"Total time: {t1 - t0:.2f} seconds")
        self.val_results["experiment_datetime"] = self.train_results["experiment_datetime"]


        self.save_results(
            result_file=os.path.join(self.result_folder, "all_results.csv"),
            results=pd.merge(
                self.train_results, 
                self.val_results,
                how="outer")
                # self.val_results.loc[:, self.val_results.columns.difference(cfg_cols)]
                
            )


class MultiDetectors:
    def __init__(self, detectors, model, device, base_config, seed_split):

        """
        Args:
            detectors (list): List of detector instances.
        """
        self.detectors = detectors
        self.model = model
        self.device = device
        self.base_config = base_config
        self.seed_split = seed_split
        self.latent_path = f"storage_latent/{base_config['data']['name']}_{base_config['model']['name']}_r-{base_config['data']['r']}_seed-split-{seed_split}/logits_train.pt"
        

    def fit(self, train_dataloader):
        """
        Fit all detectors on the training data.
        
        Args:
            train_dataloader (DataLoader): DataLoader for the training data.
        """
        self.model.eval()

        # all_model_preds = []
        if os.path.exists(self.latent_path):
            pkg = torch.load(self.latent_path, map_location="cpu")
            all_logits = pkg["logits"].to(torch.float32)        # (N, C)
            labels = pkg["labels"]              # (N,)
            model_preds  = pkg["model_preds"]# (N,)
            all_detector_labels = (model_preds != labels).float()
        
        else:
            
            all_model_preds = []
            all_labels = []
            all_logits = []
            for epoch in range(self.n_epochs):
                with torch.no_grad():
                    for inputs, targets in tqdm(train_dataloader, total=len(train_dataloader), desc="Getting Training Logits", disable=False):
                        inputs = inputs.to(self.device)
                        # targets = targets.to(self.device)
                    
                        logits = self.model(inputs).cpu()  # logits: [batch_size, num_classes]
                        model_preds = torch.argmax(logits, dim=1)

                        # detector_labels = (model_preds != targets).float()
                        # # all_model_preds.append(model_preds)
                        # all_detector_labels.append(detector_labels)
                        all_logits.append(logits)
                        all_labels.append(targets)
                        all_model_preds.append(model_preds)

            
            
            # all_model_preds = torch.cat(all_model_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            all_model_preds = torch.cat(all_model_preds, dim=0)
            all_detector_labels = (all_model_preds != all_labels).float()
            all_logits = torch.cat(all_logits, dim=0)

            # AFTER (robust)
            parent = os.path.dirname(self.latent_path)
            os.makedirs(parent, exist_ok=True)

            tmp = self.latent_path + ".tmp"
            torch.save(
                {
                    "logits": all_logits.cpu(),     # compact on disk
                    "labels": all_labels.cpu().to(torch.int64),
                    "model_preds": all_model_preds.cpu().to(torch.int64),
                },
                tmp,
            )
            os.replace(tmp, self.latent_path)  # atomic rename
            del all_model_preds, all_labels

        # Saving 

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
            pred_weight=None,
            batch_size=2048,
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
        self.model = model
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
        self.batch_size = batch_size

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

        self.pred_weight = pred_weight

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
                                                    init_params=self.init_scheme,
                                                    n_init=self.n_init, verbose=0, 
                                                    # reg_covar=1e-3
                                                    )
        elif self.method == "kmeans_torch":
            
            self.clustering_algo = TorchKMeans(
                n_clusters=self.n_cluster, 
                seed=self.kmeans_seed, 
                init_method=self.init_scheme,
                num_init=self.n_init, 
                verbose=0, 
                p_norm=2,
                normalize=None,
                # reg_covar=1e-3
            )

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
        elif self.method == "minikmeans":
            self.clustering_algo = MiniBatchKMeans(n_clusters=self.n_cluster, 
                                          random_state=self.kmeans_seed, 
                                            n_init=self.n_init,
                                          init=self.init_scheme, 
                                        batch_size=self.batch_size,
                                          verbose=0)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        
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
            elif self.partionning_space == "logits":
                embs = logits
        else:
            self.model.to(self.device)
            logits = self.model(x)
            self.model.to(torch.device('cpu'))
            if self.partionning_space == "gini":
                embs = gini(logits, temperature=self.temperature, normalize=self.normalize_gini)
            elif self.partionning_space == "probits":
                embs = torch.softmax(logits / self.temperature, dim=1)
            elif self.partionning_space == "logits":
                embs = logits
            else:
                raise ValueError("Unsupported partionning space")


        # Reorder embeddings if needed
        if self.reorder_embs:
            embs = embs.sort(dim=1, descending=True)[0]
        if self.pred_weight is not None:
            preds = torch.argmax(logits, dim=1)
            preds_onehot = torch.nn.functional.one_hot(preds, num_classes=self.n_classes).float()
            scale = embs.detach().abs().amax()
            W = float(self.pred_weight) * (float(scale) + 1e-8)
            embs = torch.cat([embs, W * preds_onehot.to(embs.device)], dim=1)
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
        
        else:
            if self.reducer is not None:
                embs = self.reducer.transform(embs.cpu().numpy())
                cluster = torch.tensor(self.clustering_algo.predict(embs), 
                                    device=self.device)
            else:
                if self.method in ["kmeans_torch", "soft-kmeans_torch"]:
                    embs = embs.to(self.device).float().unsqueeze(0)
                    cluster = self.clustering_algo.predict(embs).squeeze(0).cpu()
                    # print("cluster predict shape", cluster.shape)
                else:
                    cluster = torch.tensor(self.clustering_algo.predict(embs.cpu().numpy()), 
                                   device=self.device)
            return cluster
        # else:
        #     raise ValueError("Unsupported method")

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

        # print("logits shape", logits.shape)
        all_embs = self._extract_embeddings(logits=logits)
    
        if self.reducer is not None:
            # If a reducer is used, fit it on the embeddings
           
            all_embs = torch.tensor(self.reducer.fit_transform(all_embs.cpu().numpy()), device=self.device)
        # self.all_embs = all_embs.cpu().numpy().squeeze(-1)

        # print("dtype", all_embs.cpu().numpy().dtype)
        # print("c contiguous", all_embs.cpu().numpy().flags.c_contiguous)
        if self.method in ["kmeans_torch", "soft-kmeans_torch"]:
            all_embs = all_embs.to(self.device).unsqueeze(0)
            clusters = self.clustering_algo.fit_predict(all_embs)
            clusters = clusters.squeeze(0).cpu()
            # print("clusters fit shape", clusters.shape)
        else:
            all_embs = all_embs.cpu().numpy()
            clusters = self.clustering_algo.fit_predict(all_embs)
            clusters = torch.tensor(clusters)
        # print("clusters fit shape", clusters.shape)


        if self.method == "kmeans":
            self.inertia = self.clustering_algo.inertia_
            self.n_iter = self.clustering_algo.n_iter_
        elif self.method == "soft-kmeans":
            self.inertia = self.clustering_algo.lower_bound_
        elif self.method == "kmeans_torch":
            self.inertia = self.clustering_algo._result.inertia
            self.n_iter = self.clustering_algo.n_iter
        # print("n_iter", self.n_iter)
        # print("inertia", self.inertia)

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


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


from sklearn.ensemble import RandomForestClassifier



# @register_detector("logistic")
# class BaseDetector:
#     def __init__(self, model, device=torch.device('cpu'), regressor=LogisticRegression, 
#                  penalty="l2", C=1, reorder_embs=True, temperature=1, feature_space="probits", **kwargs):
#         self.model = model
#         self.device = device
#         self.penalty = penalty
#         self.C = C
#         self.temperature = temperature
#         self.feature_space = feature_space
#         self.reorder_embs = reorder_embs
#         self.regressor = regressor(
#             penalty=self.penalty, C=self.C, solver="saga") 

    
#     def _extract_embeddings(self, x=None, logits=None):
#         """
#         Extract embeddings from the model.
#         This function is used to create a feature extractor.
#         """

#         if logits is not None:
#             if self.feature_space == "gini":
#                 embs = gini(logits, temperature=self.temperature, normalize=self.normalize_gini)
#             elif self.feature_space == "probits":
#                 embs = torch.softmax(logits / self.temperature, dim=1)
#         else:
#             logits = self.model(x)
#             if self.feature_space == "gini":
#                 embs = gini(logits, temperature=self.temperature, normalize=self.normalize_gini)
#             elif self.feature_space == "probits":
#                 embs = torch.softmax(logits / self.temperature, dim=1)
#             else:
#                 raise ValueError("Unsupported partionning space") 
            
#         if self.reorder_embs:
#             embs = embs.sort(dim=1, descending=True)[0]
#         return embs
         
#     @torch.no_grad()
#     def fit(self, data_loader=None, logits=None, detector_labels=None, verbose=False):

#         self.model.eval()
        
#         if data_loader is not None:
#             all_logits = []
#             all_detector_labels = []
            
#             for inputs, targets in tqdm(data_loader, total=len(data_loader), desc="Getting Training Logits", disable=False):
#                     inputs = inputs.to(self.device)
#                     targets = targets.to(self.device)
                
#                     logits = self.model(inputs).cpu()  # logits: [batch_size, num_classes]
#                     model_preds = torch.argmax(logits, dim=1)

#                     detector_labels = (model_preds != targets.cpu()).float()
#                     # all_model_preds.append(model_preds)
#                     all_detector_labels.append(detector_labels)
#                     all_logits.append(logits)
            
        
#             # all_model_preds = torch.cat(all_model_preds, dim=0)
#             all_detector_labels = torch.cat(all_detector_labels, dim=0)
#             all_logits = torch.cat(all_logits, dim=0)
#         else:
#             assert logits is not None and detector_labels is not None
#             all_logits = logits.cpu()
#             all_detector_labels = detector_labels.cpu()

#         all_embs = self._extract_embeddings(logits=all_logits)
#         self.regressor.fit(all_embs.numpy(), all_detector_labels.numpy())

#     def __call__(self, x=None, logits=None):
#         embs = self._extract_embeddings(x, logits)
#         preds = self.regressor.predict_proba(embs.cpu().numpy())[:, 1]
#         return preds

from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, Optional

# Assume your decorator exists
# from code.utils.detection.registry import register_detector

# Assume your gini() transform exists somewhere in your codebase:
# def gini(logits: torch.Tensor, temperature: float = 1.0, normalize: bool = False) -> torch.Tensor:
#     ...

class BasePostHocDetector(BaseEstimator):
    """
    Base class for post-hoc misclassification detectors that operate on
    embeddings derived from model logits (e.g., softmax^T "probits" or "gini").
    Subclasses must implement _make_estimator(estimator_kwargs) to provide
    a scikit-learn classifier with predict_proba.
    """
    def __init__(
        self,
        model,
        device: torch.device = torch.device("cpu"),
        *,
        space: str = "probits",            # {"probits", "gini"}
        reorder_embs: bool = True,         # sort features decreasingly per sample
        temperature: float = 1.0,
        normalize_gini: bool = False,
        magnitude: float = 0.0,            # for ODIN-like variants (unused here)
        estimator_kwargs: Optional[Dict[str, Any]] = None,  # passed to subclass estimator
    ):
        self.model = model.to(device)
        self.device = device

        # embedding options
        self.space = space
        self.partionning_space = space      # keep legacy attribute name
        self.reorder_embs = reorder_embs
        self.temperature = float(temperature)
        self.normalize_gini = bool(normalize_gini)
        self.magnitude = float(magnitude)

        # estimator
        self._estimator_kwargs = dict(estimator_kwargs or {})
        self.regressor = self._make_estimator(self._estimator_kwargs)

    # ---------- Abstract factory ----------
    def _make_estimator(self, estimator_kwargs: Dict[str, Any]):
        raise NotImplementedError

    # ---------- Embeddings ----------
    def _extract_embeddings(self, x: Optional[torch.Tensor] = None, logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Produce an embedding matrix [N, C] from either provided logits or from x via self.model.
        Supported spaces: "probits", "gini".
        """
        if logits is None:
            if x is None:
                raise ValueError("Either x or logits must be provided.")
            with torch.no_grad():
                logits = self.model(x.to(self.device)).detach().cpu()  # [B, C]
        else:
            logits = logits.detach().cpu()

        if self.space == "probits":
            embs = torch.softmax(logits / self.temperature, dim=1)
        elif self.space == "gini":
            # Provided by your codebase
            embs = gini(logits, temperature=self.temperature, normalize=self.normalize_gini)
        elif self.space == "logits":
            embs = logits
        else:
            raise ValueError(f"Unsupported partionning space: {self.space}")

        if self.reorder_embs:
            embs = embs.sort(dim=1, descending=True)[0]
        return embs  # torch.FloatTensor [N, C]

    # ---------- Fit / Inference ----------
    @torch.no_grad()
    def fit(self, data_loader=None, logits: Optional[torch.Tensor] = None, detector_labels: Optional[torch.Tensor] = None, verbose: bool = False):
        """
        Fit the regressor to P(error | embedding).
        If data_loader is given, compute logits and labels from the model;
        otherwise expects precomputed logits and detector_labels (0/1).
        """
        self.model.eval()

        if data_loader is not None:
            all_logits = []
            all_detector_labels = []
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.cpu()
                l = self.model(inputs).detach().cpu()                   # [B, C]
                preds = torch.argmax(l, dim=1)
                dlab = (preds != targets).float()                       # 1 = error
                all_logits.append(l)
                all_detector_labels.append(dlab)
            logits = torch.cat(all_logits, dim=0)
            detector_labels = torch.cat(all_detector_labels, dim=0)
        else:
            if logits is None or detector_labels is None:
                raise ValueError("Provide either data_loader or (logits and detector_labels).")
            logits = logits.detach().cpu()
            detector_labels = detector_labels.detach().cpu()

        X = self._extract_embeddings(logits=logits).numpy()
        y = detector_labels.numpy().astype(np.int32)

        self.regressor.fit(X, y)
        return self

    def __call__(self, x: Optional[torch.Tensor] = None, logits: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Returns P(error | embedding) as a NumPy array of shape [N].
        """
        X = self._extract_embeddings(x=x, logits=logits).cpu().numpy()
        proba = self.regressor.predict_proba(X)  # [:, 1] is class "1" (error)
        return torch.tensor(proba[:, 1]).to(self.device)

    # ---------- Hyperparameter management ----------
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        sklearn-compatible. Exposes both base params and nested estimator params
        under the prefix 'est__'.
        """
        params = {
            "space": self.space,
            "reorder_embs": self.reorder_embs,
            "temperature": self.temperature,
            "normalize_gini": self.normalize_gini,
            "magnitude": self.magnitude,
        }
        # include estimator params with prefix
        if hasattr(self.regressor, "get_params"):
            est_params = self.regressor.get_params(deep=deep)
            params.update({f"est__{k}": v for k, v in est_params.items()})
        return params

    def set_params(self, **params):
        """
        sklearn-compatible. Keys starting with 'est__' are routed to the estimator.
        """
        est_updates = {k[5:]: v for k, v in params.items() if k.startswith("est__")}
        base_updates = {k: v for k, v in params.items() if not k.startswith("est__")}

        for k, v in base_updates.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown parameter {k}")
            setattr(self, k, v)

        if est_updates:
            if hasattr(self.regressor, "set_params"):
                self.regressor.set_params(**est_updates)
            else:
                raise ValueError("Underlying estimator does not support set_params.")
        return self

    @classmethod
    def default_param_grid(cls) -> Dict[str, Any]:
        """
        Subclasses should override to provide a sensible grid for the estimator.
        """
        return {}


# ------------------- Concrete detectors -------------------

@register_detector("knn")
class KNNDetector(BasePostHocDetector):
    """
    k-NN detector on top of embeddings.
    """
    def __init__(
        self,
        model,
        device: torch.device = torch.device("cpu"),
        *,
        # embedding options
        space: str = "probits",
        reorder_embs: bool = True,
        temperature: float = 1.0,
        normalize_gini: bool = False,
        magnitude: float = 0.0,
        # k-NN hyperparameters
        n_neighbors: int = 50,
        weights: str = "uniform",    # {"uniform", "distance"}
        p: Optional[int] = 2,        # None -> use metric below
        metric: Any = "minkowski",   # or "cosine", etc. (ignored if p is not None and metric="minkowski")
        **kwargs,
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.metric = metric if p is None else "minkowski"

        est_kwargs = dict(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            p=self.p if self.metric == "minkowski" else 2,
            metric=self.metric,
        )
        super().__init__(
            model, device,
            space=space, reorder_embs=reorder_embs, temperature=temperature,
            normalize_gini=normalize_gini, magnitude=magnitude,
            estimator_kwargs=est_kwargs,
        )

    def _make_estimator(self, estimator_kwargs: Dict[str, Any]):
        return KNeighborsClassifier(**estimator_kwargs)

    @classmethod
    def default_param_grid(cls) -> Dict[str, Any]:
        return {
            "est__n_neighbors": [10, 25, 50, 100],
            "est__weights": ["uniform", "distance"],
            # if you want cosine distance: set metric="cosine", p ignored
            "est__p": [1, 2],
        }


@register_detector("random_forest")
class RandomForestDetector(BasePostHocDetector):
    """
    Random-Forest detector on top of embeddings.
    """
    def __init__(
        self,
        model,
        device: torch.device = torch.device("cpu"),
        *,
        # embedding options
        space: str = "probits",
        reorder_embs: bool = True,
        temperature: float = 1.0,
        normalize_gini: bool = False,
        magnitude: float = 0.0,
        # RF hyperparameters
        n_estimators: int = 400,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Any = "sqrt",
        class_weight: Optional[str] = None,   # consider "balanced" if errors are rare
        bootstrap: bool = True,
        max_samples: Optional[float] = None,  # sklearn>=1.2
        n_jobs: int = -1,
        random_state: int = 0,
        oob_score: bool = False,
        **kwargs,
    ):
        est_kwargs = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            oob_score=oob_score,
        )
        if max_samples is not None:
            est_kwargs["max_samples"] = max_samples

        super().__init__(
            model, device,
            space=space, reorder_embs=reorder_embs, temperature=temperature,
            normalize_gini=normalize_gini, magnitude=magnitude,
            estimator_kwargs=est_kwargs,
        )

    def _make_estimator(self, estimator_kwargs: Dict[str, Any]):
        return RandomForestClassifier(**estimator_kwargs)

    @classmethod
    def default_param_grid(cls) -> Dict[str, Any]:
        return {
            "est__n_estimators": [200, 400, 800],
            "est__max_depth": [None, 12, 24],
            "est__min_samples_split": [2, 5, 10],
            "est__min_samples_leaf": [1, 2, 4],
            "est__max_features": ["sqrt", "log2", 0.5],
            "est__class_weight": [None, "balanced"],
            # Optional if supported:
            # "est__max_samples": [None, 0.7, 0.9],
        }


# @register_detector("knn")
# class BaseDetector:
#     def __init__(self, model, device=torch.device('cpu'), regressor=KNeighborsClassifier, 
#                  n_neighbors=50, weights="uniform", p=2, metric=2, magnitude=0, 
#                  space="probits", reorder_embs=True, temperature=1, **kwargs):
#         self.model = model.to(device)
#         self.device = device
#         self.magnitude = magnitude
#         self.partionning_space = space
#         self.reorder_embs = reorder_embs
#         self.n_neighbors = n_neighbors
#         self.weights = weights
#         self.p = p
    
#         if p is None:
#             self.metric = metric
#         else:
#             self.metric = "minkowski"
#         self.temperature = temperature
#         self.regressor = regressor(
#             n_neighbors=self.n_neighbors, 
#             weights=self.weights, 
#             p=self.p, 
#             metric=self.metric) 

    
#     def _extract_embeddings(self, x=None, logits=None):
#         """
#         Extract embeddings from the model.
#         This function is used to create a feature extractor.
#         """

#         if logits is not None:
#             if self.partionning_space == "gini":
#                 embs = gini(logits, temperature=self.temperature, normalize=self.normalize_gini)
#             elif self.partionning_space == "probits":
#                 embs = torch.softmax(logits / self.temperature, dim=1)
#         else:
#             logits = self.model(x)
#             if self.partionning_space == "gini":
#                 embs = gini(logits, temperature=self.temperature, normalize=self.normalize_gini)
#             elif self.partionning_space == "probits":
#                 embs = torch.softmax(logits / self.temperature, dim=1)
#             else:
#                 raise ValueError("Unsupported partionning space") 
            
#         if self.reorder_embs:
#             embs = embs.sort(dim=1, descending=True)[0]
#         return embs
         
#     @torch.no_grad()
#     def fit(self, data_loader=None, logits=None, detector_labels=None, verbose=False):

#         self.model.eval()
        
#         if data_loader is not None:
#             all_logits = []
#             all_detector_labels = []
            
#             for inputs, targets in tqdm(data_loader, total=len(data_loader), desc="Getting Training Logits", disable=False):
#                     inputs = inputs.to(self.device)
#                     targets = targets.to(self.device)
                
#                     logits = self.model(inputs).cpu()  # logits: [batch_size, num_classes]
#                     model_preds = torch.argmax(logits, dim=1)

#                     detector_labels = (model_preds != targets.cpu()).float()
#                     # all_model_preds.append(model_preds)
#                     all_detector_labels.append(detector_labels)
#                     all_logits.append(logits)
            
        
#             # all_model_preds = torch.cat(all_model_preds, dim=0)
#             all_detector_labels = torch.cat(all_detector_labels, dim=0)
#             all_logits = torch.cat(all_logits, dim=0)
#         else:
#             assert logits is not None and detector_labels is not None
#             all_logits = logits.cpu()
#             all_detector_labels = detector_labels.cpu()

#         all_embs = self._extract_embeddings(logits=all_logits)
#         self.regressor.fit(all_embs.numpy(), all_detector_labels.numpy())

#     def __call__(self, x=None, logits=None):
#         embs = self._extract_embeddings(x, logits)
#         preds = self.regressor.predict_proba(embs.cpu().numpy())[:, 1]
#         return preds


# @register_detector("rf")
# class RandomForestDetector(BaseDetector):
#     """
#     Random-Forest detector over embeddings produced by BaseDetector._extract_embeddings.
#     - Embeddings: either "probits" (softmax^T) or "gini" (via your gini() fn), with optional sorting.
#     - Target: binary misclassification indicator (1 = error, 0 = correct).
#     - Output: predict_proba(...)[..., 1] = estimated P(error | embedding).
#     """
#     def __init__(
#         self,
#         model,
#         device=torch.device('cpu'),
#         *,
#         # RF hyperparameters (sane defaults)
#         n_estimators: int = 400,
#         max_depth = None,
#         min_samples_split: int = 2,
#         min_samples_leaf: int = 1,
#         max_features = "sqrt",           # good default for classification
#         class_weight = None,             # consider "balanced" on imbalance
#         bootstrap: bool = True,
#         max_samples = None,              # None -> all bootstrap samples
#         n_jobs: int = -1,
#         random_state: int = 0,
#         oob_score: bool = False,
#         # Embedding / detector options
#         space: str = "probits",
#         reorder_embs: bool = True,
#         temperature: float = 1.0,
#         normalize_gini: bool = False,    # used if space == "gini"
#         magnitude: float = 0,
#         **kwargs,
#     ):
#         # Initialize BaseDetector fields (it expects a regressor; we override it right after)
#         super().__init__(
#             model=model,
#             device=device,
#             regressor=KNeighborsClassifier,   # dummy placeholder (overridden below)
#             n_neighbors=1, weights="uniform", p=2, metric=2,
#             magnitude=magnitude,
#             space=space,
#             reorder_embs=reorder_embs,
#             temperature=temperature,
#             **kwargs,
#         )
#         # Ensure attribute used by _extract_embeddings exists
#         self.normalize_gini = normalize_gini

#         # Build the actual RF classifier
#         rf_kwargs = dict(
#             n_estimators=n_estimators,
#             max_depth=max_depth,
#             min_samples_split=min_samples_split,
#             min_samples_leaf=min_samples_leaf,
#             max_features=max_features,
#             class_weight=class_weight,
#             bootstrap=bootstrap,
#             n_jobs=n_jobs,
#             random_state=random_state,
#             oob_score=oob_score,
#         )
#         # Add only if explicitly set (older sklearns may not support it)
#         if max_samples is not None:
#             rf_kwargs["max_samples"] = max_samples

#         self.regressor = RandomForestClassifier(**rf_kwargs)
    

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

        return gini(logits, temperature=self.temperature, normalize=self.normalize).squeeze()  # [batch_size]
    

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