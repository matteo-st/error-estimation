import torch
import numpy as np
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

class DetectorEvaluator:
    def __init__(self, model, dataloader, device, return_embs=False, return_labels=False):
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
                    # embs = detector.feature_extractor(inputs).squeeze(-1)
                else:
                    clusters = torch.tensor([np.nan] * inputs.shape[0], device=self.device) # [np.nan] * inputs.shape[0]
                    # embs = torch.tensor([np.nan] * inputs.shape[0], device=self.device) # [np.nan] * inputs.shape[0]
                if self.return_embs:
                    embs = detector.feature_extractor(inputs).squeeze(-1)
                else:
                    embs = torch.tensor([np.nan] * inputs.shape[0], device=self.device)
                if not self.return_labels:
                    labels = torch.tensor([np.nan] * inputs.shape[0], device=self.device)
                
                    
                detector_preds = detector(inputs)

                # all_model_preds.append(model_preds.cpu().numpy())
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

        fprs, tprs, thrs = roc_curve(all_detector_labels, all_detector_preds)
        # Compute the area under the ROC curve
        fpr, tpr, thr = self.fpr_at_fixed_tpr(fprs, tprs, thrs, 0.95)
        return fpr, tpr, thr, all_detector_preds, all_detector_labels, all_clusters, all_embs, all_labels
        # if self.return_embs:
        #     return fpr, tpr, thr, all_detector_preds, all_detector_labels, all_clusters, all_embs
        # else:
        #     return fpr, tpr, thr, all_detector_preds, all_detector_labels, [None] * len(all_detector_labels), [None] * len(all_detector_labels)