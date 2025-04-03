from sklearn.metrics import auc, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd

def fpr_at_fixed_tpr(fprs, tprs, thresholds, tpr_level: float = 0.95):
    if all(tprs < tpr_level):
        raise ValueError(f"No threshold allows for TPR at least {tpr_level}.")
    idxs = [i for i, x in enumerate(tprs) if x >= tpr_level]
    if len(idxs) > 0:
        idx = min(idxs)
    else:
        idx = 0
    return fprs[idx], tprs[idx], thresholds[idx]


def hard_coverage(scores, thr: float):
    return (scores <= thr).mean()


def selective_net_risk(scores, pred, targets, thr: float):
    covered_idx = scores <= thr
    return np.sum(pred[covered_idx] != targets[covered_idx]) / np.sum(covered_idx)


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


# def evaluate(preds, targets, scores):
#     scores = scores.view(-1).detach().cpu().numpy()
#     targets = targets.view(-1).detach().cpu().numpy()
#     preds = preds.view(-1).detach().cpu().numpy()

#     model_acc = (preds == targets).mean()
#     train_labels = preds != targets
#     fprs, tprs, thrs = roc_curve(train_labels, scores)
#     roc_auc = auc(fprs, tprs)

#     fpr, tpr, thr = fpr_at_fixed_tpr(fprs, tprs, thrs, 0.95)
#     print("Matteo - fpr, tpr, thr", fpr, tpr, thr)
#     risks, coverages, _ = risks_coverages_selective_net(scores, preds, targets)
#     aurc = auc(coverages, risks)
#     return model_acc, roc_auc, fpr, aurc

def get_classification_metrics(preds, targets, scores):
    # Convert tensors to numpy arrays
    # print("score", type(scores), np.size(scores))

    # scores = scores.view(-1).detach().cpu().numpy()
    # targets = targets.view(-1).detach().cpu().numpy()
    # preds = preds.view(-1).detach().cpu().numpy()

    # Dataset Information
    n_samples = len(targets)
    unique_classes, counts = np.unique(targets, return_counts=True)
    n_classes = len(unique_classes)
    
    # Overall Model Accuracy
    model_acc = (preds == targets).mean()

    # Accuracy per Class
    per_class_acc = {}
    for c in unique_classes:
        idx = (targets == c)
        class_acc = (preds[idx] == targets[idx]).mean()
        per_class_acc[c] = class_acc

    # Compute ROC curve and AUC for misclassification detection
    train_labels = preds != targets  # Misclassification indicator
    fprs, tprs, thrs = roc_curve(train_labels, scores)
    roc_auc = auc(fprs, tprs)

    # Compute FPR at a fixed TPR (here, 0.95)
    fpr_val, tpr_val, thr_val = fpr_at_fixed_tpr(fprs, tprs, thrs, 0.95)
    
    # Compute risks and coverages for selective net metrics
    risks, coverages, _ = risks_coverages_selective_net(scores, preds, targets)
    if len(np.unique(coverages)) < 2:
        print("Not enough points to compute AUC. Setting AURC to a default value.")
        aurc = 0.0  # or another appropriate default value
    else:
        aurc = auc(coverages, risks)


    # Print Dataset Information
    # print("Dataset Information:")
    # print(f"  - Number of samples: {n_samples}")
    # print(f"  - Number of classes: {n_classes}")
    # print("  - Class distribution:")
    # for cls, count in zip(unique_classes, counts):
    #     print(f"      Class {cls}: {count} samples")
    
    # # Print Model Accuracy
    # print("\nModel Accuracy:")
    # print(f"  - Overall accuracy: {model_acc:.4f}")
    # print("  - Accuracy per class:")
    # for cls, acc in per_class_acc.items():
    #     print(f"      Class {cls}: {acc:.4f}")
    
    # Print Other Metrics
    # print("\nOther Metrics:")
    # print(f"  - ROC AUC: {roc_auc:.4f}")
    print(f"  - FPR at fixed TPR (0.95): {fpr_val:.4f}")
    print(f"  - TPR corresponding to FPR: {tpr_val:.4f}")
    print(f"  - Threshold at fixed TPR: {thr_val:.4f}")
    # print(f"  - AURC: {aurc:.4f}")
    
    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fprs, tprs, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    return model_acc, roc_auc, fpr_val, tpr_val, aurc



def evaluate_classification(preds, targets, scores, results_folder, name_save_file):

    model_acc, roc_auc, fpr, tpr, aurc = get_classification_metrics(preds, targets, scores)
    results = {
        "accuracy": model_acc,
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc,
        "aurc": aurc,
    }
    # print(json.dumps(results, indent=2))
    
    # Save the results to a CSV file in the experiment folder.
    # The experiment folder should be stored in args.results_folder.
    csv_path = os.path.join(results_folder, name_save_file + "_classif_results.csv")
    results_df = pd.DataFrame([results])
    
    if not os.path.isfile(csv_path):
        results_df.to_csv(csv_path, index=False, header=True)
    else:
        results_df.to_csv(csv_path, mode="a", index=False, header=False)
    
    print(f"Classification results saved to {csv_path}")