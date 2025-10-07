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

from .datasets import get_id_ood_dataloader



from .postprocessor import get_postprocessor

def gini(logits, temperature=1.0, normalize=False):
    g =torch.sum(torch.softmax(logits / temperature, dim=1) ** 2, dim=1, keepdim=True)
    if normalize:
        return  (1 - g) / g 
    else:
        return 1 - g




class EvaluatorOOD:
    def __init__(
            self, 
            detectors, 
            model, 
            id_name,
            device,
            postprocessor_name = "msp",
            data_root: str = "./data",
            config_root: str = "./configs",
            APS_mode= True,
            base_config =None, 
            list_configs=None, 
            metric='fpr', 
            result_folder="results/",
            preprocessor =None,
            batch_size: int = 200,
            shuffle: bool = False,
            num_workers: int = 4,
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

        self.APS_mode = APS_mode
        postprocessor = get_postprocessor(config_root, postprocessor_name, id_name)

        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
        }

        dataloader_dict = get_id_ood_dataloader(
            id_name,
            data_root,
            preprocessor,
            postprocessor_name=postprocessor_name,
            **loader_kwargs,
        )


        self.id_name = id_name
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.dataloader_dict = dataloader_dict
        self.metrics = {"id_acc": None, "csid_acc": None, "ood": None, "fsood": None}
        self.scores = {
            "id": {"train": None, "val": None, "test": None},
            "csid": {k: None for k in dataloader_dict["csid"].keys()},
            "ood": {
                "val": None,
                "near": {k: None for k in dataloader_dict["ood"]["near"].keys()},
                "far": {k: None for k in dataloader_dict["ood"]["far"].keys()},
            },
            "id_preds": None,
            "id_labels": None,
            "csid_preds": {k: None for k in dataloader_dict["csid"].keys()},
            "csid_labels": {k: None for k in dataloader_dict["csid"].keys()},
        }


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


    def search_no_fit(self, id_name, ood_name):

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