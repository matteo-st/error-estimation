import torch
import numpy as np
from sklearn.metrics import auc, roc_curve, average_precision_score
from tqdm import tqdm
from torch.autograd import Variable
from torch_uncertainty.metrics.classification import AURC
import pandas as pd

def selective_net_risk(scores, pred, targets, thr: float):

    covered_idx = scores <= thr

    return np.sum(pred[covered_idx] != targets[covered_idx]) / np.sum(covered_idx)

def hard_coverage(scores, thr: float):
    return (scores <= thr).mean()

def risks_coverages_selective_net(scores, pred, targets, sort=True):
    """
    Returns:

        risks, coverages, thrs
    """
    # this function is slow
    risks = []
    coverages = []
    thrs = []
    for thr in np.unique(scores):
        risks.append(selective_net_risk(scores, pred, targets, thr))
        coverages.append(hard_coverage(scores, thr))
        thrs.append(thr)
    risks = np.array(risks)
    coverages = np.array(coverages)
    thrs = np.array(thrs)

    # sort by coverages
    if sort:
        sorted_idx = np.argsort(coverages)
        risks = risks[sorted_idx]
        coverages = coverages[sorted_idx]
        thrs = thrs[sorted_idx]
    return risks, coverages, thrs

class DetectorEvaluator:
    def __init__(self, model, dataloader, device, magnitude = 0, 
                 return_embs=False, return_labels=False, return_model_preds=False,
                 path=None):
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
        self.return_labels = return_labels  # Whether to return labels or not
        self.return_model_preds = return_model_preds
        self.path = path
        self.magnitude = magnitude

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
        all_labels = []
        all_model_preds = []
        aurc = AURC()
        
        for inputs, labels in tqdm(self.dataloader, desc="Evaluating Detector", disable=False):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            # x: [batch_size, dim], labels: [batch_size]


            if self.magnitude > 0:
                
                inputs = Variable(inputs, requires_grad=True)
                logits = self.model(inputs)
                model_preds = torch.argmax(logits, dim=1)
                detector_labels = model_preds != labels

                detector_preds = detector(logits=logits)
                torch.log(detector_preds).sum().backward()
          
                inputs = inputs - self.magnitude * torch.sign(-inputs.grad)
            # inputs = torch.clamp(inputs, 0, 1)
                inputs = Variable(inputs, requires_grad=False)

                with torch.no_grad():
                    detector_preds = detector(inputs)

            else:
                with torch.no_grad():
                    logits = self.model(inputs)  # logits: [batch_size, num_classes]
                    model_preds = torch.argmax(logits, dim=1)  # [batch_size]
            
                    detector_labels = model_preds != labels
                    detector_preds = detector(inputs)
            


            if return_clusters:
                clusters = detector.predict_clusters(inputs)
                # embs = detector.feature_extractor(inputs).squeeze(-1)
            else:
                clusters = torch.tensor([np.nan] * inputs.shape[0], device=self.device) # [np.nan] * inputs.shape[0]
                # embs = torch.tensor([np.nan] * inputs.shape[0], device=self.device) # [np.nan] * inputs.shape[0]
            if self.return_embs:
                embs = detector.feature_extractor(inputs).squeeze(-1)
            else:
                embs = torch.tensor([np.nan] * inputs.shape[0], device=self.device)
            # if not self.return_labels:
            #     labels = torch.tensor([np.nan] * inputs.shape[0], device=self.device)
            # if not self.return_model_preds:
            #     model_preds = torch.tensor([np.nan] * inputs.shape[0], device=self.device)
          
            # aurc.update(detector_preds, detector_labels)
            
        
            all_model_preds.append(model_preds.cpu().numpy())
            all_clusters.append(clusters.cpu().numpy())
            all_embs.append(embs.cpu().numpy())
            all_detector_labels.append(detector_labels.cpu().numpy())
            all_detector_preds.append(detector_preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        all_detector_preds = np.concatenate(all_detector_preds, axis=0)
        all_detector_labels = np.concatenate(all_detector_labels, axis=0)
        all_clusters = np.concatenate(all_clusters, axis=0)
        all_embs = np.concatenate(all_embs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_model_preds = np.concatenate(all_model_preds, axis=0)
        # print("aurc_result", aurc.compute())
        if self.path is not None:
            np.savez_compressed(
                self.path,
                embs=all_embs,
                detector_preds=all_detector_preds,
                detector_labels=all_detector_labels,
                clusters=all_clusters,
                labels=all_labels,
                model_preds=all_model_preds
            )

        fprs, tprs, thrs = roc_curve(all_detector_labels, all_detector_preds)
        # Compute the area under the ROC curve
        roc_auc = auc(fprs, tprs)
        fpr, tpr, thr = self.fpr_at_fixed_tpr(fprs, tprs, thrs, 0.95)
        model_acc = (all_model_preds == all_labels).mean()
        risks, coverages, _ = risks_coverages_selective_net(
            np.squeeze(all_detector_preds),
            np.squeeze(all_model_preds), 
            all_labels)
        aurc = auc(coverages, risks)
    
        aupr_err = average_precision_score(all_detector_labels, all_detector_preds)
        aupr_success = average_precision_score(1 - all_detector_labels, 1 - all_detector_preds)

        return fpr, tpr, thr, roc_auc, model_acc, aurc, aupr_err, aupr_success
        # if self.return_embs:
        #     return fpr, tpr, thr, all_detector_preds, all_detector_labels, all_clusters, all_embs
        # else:
        #     return fpr, tpr, thr, all_detector_preds, all_detector_labels, [None] * len(all_detector_labels), [None] * len(all_detector_labels)



class MultiDetectorEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        magnitudes,
        suffix = "train"
    ):
        """
        Evaluate a single model against multiple post-hoc detectors.

        Args:
            model:                A trained classifier (outputs raw logits).
            dataloader:           DataLoader for the evaluation set.
            device:               torch.device to run model & detectors on.
            magnitude:            If >0, craft a one‐step adversarial example
                                  against each detector as in your original.
        """
        self.model = model.to(device)
        self.loader = dataloader
        self.device = device
        self.magnitude = magnitudes
        self.suffix = suffix

    @staticmethod
    def _fpr_at_tpr(fprs, tprs, thrs, level: float = 0.95):
        # find smallest index where TPR ≥ level
        idxs = np.where(tprs >= level)[0]
        idx = idxs.min() if idxs.size else 0
        return float(fprs[idx]), float(tprs[idx]), float(thrs[idx])

    def evaluate(self, detectors):
        """
        Run the model once per batch, then each detector on those inputs.

        Args:
            detectors:  Either a list of detector‐objects or a dict name→detector.
                        Each detector must be callable as:
                            scores = detector(inputs=..., logits=...)
                        and return a 1‐D tensor of “outlier scores” (higher = more likely error).

        Returns:
            results: dict mapping detector_name → metrics dict, e.g.
                {
                  "ODIN": {
                      "fpr@95tpr": 0.12,
                      "roc_auc": 0.94,
                      "aupr_err": 0.88,
                      "aupr_in": 0.90,
                      "aurc": 0.23,
                      "model_acc": 0.79,  # same for all detectors
                  },
                  …
                }
        """
        # normalize detectors into an ordered dict name→detector

        # storage
        n_det = len(detectors)
        n_samples = len(self.loader.dataset)
        all_scores = [np.zeros(n_samples, dtype=float) for _ in range(n_det)]
        # detector_labels = model_pred != true_label  (same for all detectors)
        all_labels = np.zeros(n_samples, dtype=int)
        all_model_preds = np.zeros(n_samples, dtype=int)
        detector_labels_arr = np.zeros(n_samples, dtype=bool)

        # iterate once over data
        idx = 0
        self.model.eval()
        for inputs, labels in tqdm(self.loader, desc="Getting logits", leave=False):
            bs = inputs.size(0)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # forward pass for model accuracy & labels
            with torch.no_grad():
                logits = self.model(inputs)
                model_preds = torch.argmax(logits, dim=1)

            # which examples are errors?
            det_lab = (model_preds != labels).cpu().numpy()
            detector_labels_arr[idx: idx+bs] = det_lab

            # store global labels & preds
            all_labels[idx: idx+bs] = labels.cpu().numpy()
            all_model_preds[idx: idx+bs] = model_preds.cpu().numpy()

            # now each detector
            for i, det in tqdm(enumerate(detectors), desc="Getting Detectors Scores", leave=False):
                # -- optionally craft 1‑step adv example per detector
                if self.magnitude[i] > 0:
                    # clone input for gradient
                    adv_in = Variable(inputs.clone(), requires_grad=True)
                    adv_logits = self.model(adv_in)
                    scores0 = det(logits=adv_logits)           # initial
                    # backprop on log-score
                    loss = torch.log(scores0 + 1e-12).sum()
                    loss.backward()
                    # step and detach
                    adv_in = (adv_in - self.magnitude[i] * adv_in.grad.sign()).detach()
                    with torch.no_grad():
                        scores = det(inputs=adv_in)
                else:
                    with torch.no_grad():
                        scores = det(logits=logits)

                all_scores[i][idx: idx+bs] = scores.cpu().numpy()

            idx += bs


        # common model accuracy
        model_acc = float((all_model_preds == all_labels).mean())
        list_results = []
        for i, scores in enumerate(all_scores):
            fprs, tprs, thrs = roc_curve(detector_labels_arr, scores)
            roc_auc = float(auc(fprs, tprs))
            fpr95, tpr95, thr95 = self._fpr_at_tpr(fprs, tprs, thrs, 0.95)

            aupr_err     = float(average_precision_score(detector_labels_arr,    scores))
            aupr_success = float(average_precision_score(~detector_labels_arr, 1 - scores))

            # selective risk/coverage → AURC
            risks, coverages, _ = risks_coverages_selective_net(
                scores, all_model_preds, all_labels
            )
            aurc = float(auc(coverages, risks))

            results = pd.DataFrame([{
                "fpr": fpr95,
                "tpr": tpr95,
                "thr": thr95,
                "roc_auc": roc_auc,
                "model_acc": model_acc,
                "aurc": aurc,
                "aupr_err": aupr_err,
                "aupr_success": aupr_success,
            }])
            
            results.columns = [f"{col}_{self.suffix}" for col in results.columns]
            list_results.append(results)

        return list_results