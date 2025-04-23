import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
import numpy as np
from tqdm import tqdm
import os
import json
import datetime
import pandas as pd
from synthetic_code.utils.dataset import GaussianMixtureDataset
from synthetic_code.utils.model import BayesClassifier, ThresholdClassifier
from synthetic_code.detection import DetectorEvaluator
from synthetic_code.utils.detector import BayesDetector



# -------------------------------
# 2. MLP Classifier for Mixture-of-Gaussians
# -------------------------------


# -------------------------------
# 3. Evaluator: Compute Accuracy
# -------------------------------
class Evaluator:
    def __init__(self, model, dataloader, device):
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

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x, labels in tqdm(self.dataloader, desc="Evaluating Classifier"):
                # x: [batch_size, dim], labels: [batch_size]
                x = x.to(self.device)
                labels = labels.to(self.device)
                logits, _ = self.model(x)  # logits: [batch_size, num_classes]
                preds = torch.argmax(logits, dim=1)  # [batch_size]
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        accuracy = correct / total
        return accuracy, torch.cat(all_preds), torch.cat(all_labels)

# -------------------------------
# 5. Main Pipeline: Data, Model, Training and Evaluation
# -------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set random seeds for reproducibility.
    seed = 10
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Mixture-of-Gaussians parameters.
    dim = 1         # Feature dimension: each sample is [10]
    n_classes = 2    # Mixture has 7 components/classes.
    n_samples_train = 20000
    n_samples_val = 500000
    batch_size_train = 32
    batch_size_val = 128
    
    # Generate mixture parameters.
    # means: [7, 10]
    means = torch.tensor([[0.0], [1.0]]).to(device)
    # stds: [7, 10]
    stds = torch.tensor([[0.4], [0.4]]).to(device)
    # weights: [7]
    weights = torch.tensor([0.5, 0.5]).to(device)
    weights = weights / weights.sum()  # Normalize to sum to 1.

    config = {
        "dim": dim,
        "n_classes": n_classes,
        "n_samples_train": n_samples_train,
        "n_samples_val": n_samples_val,
        "means": means.tolist(),
        "stds": stds.tolist(),
        "weights": weights.tolist(),
        "batch_size": 32,
        "seed": seed
    }

    
    # Create training and validation datasets.
    val_dataset = GaussianMixtureDataset(n_samples_val, means, stds, weights)
    
    # Create DataLoaders.
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)       # Each batch: [32, 10]
    
    # Instantiate the MLP classifier.
    model = ThresholdClassifier(threshold=0.3)
    
    
    # Compute Bayes classifier
    covs = torch.diag_embed(stds ** 2)
    bayes_model = BayesClassifier(means, covs, weights)
    bayes_evaluator = Evaluator(bayes_model, val_loader, device)
    bayes_accuracy, _, _ = bayes_evaluator.evaluate()
    print(f"Bayes Accuracy: {bayes_accuracy:.4f}")

    
    # Finally, evaluate the model.
    evaluator = Evaluator(model, val_loader, device)
    accuracy, _, _ = evaluator.evaluate()
    print(f"Classifier Validation Accuracy: {accuracy:.4f}")

    # Detector
    detector = BayesDetector(model, weights, means, stds, n_classes, device=torch.device('cpu'))
    evaluator = DetectorEvaluator(model, val_loader, device)
    fpr, tpr, thr = evaluator.evaluate(detector)
    print("FPR at TPR=0.95:", fpr)
    print("TPR at TPR=0.95:", tpr)
    print("Threshold at TPR=0.95:", thr)



if __name__ == "__main__":
    main()
