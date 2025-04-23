import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
from synthetic_code.utils.model import MLP

def load_data_config(data_folder):
    """
    Loads the dataset configuration from the config.json file in data_folder.
    """
    config_path = os.path.join(data_folder, "config.json")
    with open(config_path, "r") as f:
        data_config = json.load(f)
    return data_config

def load_generating_params(data_folder):
    """
    Loads the generating means and standard deviation from dataset_stats.csv.
    
    Assumes that dataset_stats.csv contains one row per class for a given dataset.
    We look for rows corresponding to the test dataset (if not found, fallback to "train").
    Each row's "mean" and "std" are stored as semicolon–separated numbers.
    
    Returns:
        means (ndarray): Array of shape (n_classes, dim) containing the generating means.
        std (float): The standard deviation (assumed to be constant across classes).
    """
    stats_path = os.path.join(data_folder, "dataset_stats.csv")
    df = pd.read_csv(stats_path)
    
    # Use rows for "test_dataset" if available; otherwise fall back to "train"
    df_subset = df[df["dataset"] == "test_dataset"]
    if df_subset.empty:
        df_subset = df[df["dataset"] == "train"]
    
    df_subset = df_subset.sort_values(by="class")
    means_list = []
    std_list = []
    for _, row in df_subset.iterrows():
        # Strip the square brackets and split on any whitespace.
        mean_str = row["mean"].strip("[]")
        # Split by whitespace and convert each piece to float.
        mean_vals = np.array([float(v) for v in mean_str.split()])
        std_val = float(row["std"])
        means_list.append(mean_vals)
        std_list.append(std_val)
    
    means = np.stack(means_list, axis=0)
    # Assume the standard deviation is identical across classes.
    std = std_list[0]
    return means, std

def bayes_proba(X_input, means_input, std_input):
    """
    For each input x (rows of X_input), returns the probability distribution over classes,
    computed via:
    
        P(y=i | x) = exp(-||x - mu_i||^2/(2 std_input^2)) / sum_j exp(-||x - mu_j||^2/(2 std_input^2))
    
    Parameters:
        X_input (ndarray): Shape (n_samples, dim)
        means_input (ndarray): Shape (n_classes, dim)
        std_input (float): Standard deviation.
        
    Returns:
        probs (ndarray): Shape (n_samples, n_classes)
    """
    # Compute squared Euclidean distances.
    dists = np.sum((X_input[:, None, :] - means_input[None, :, :]) ** 2, axis=2)
    logits = -dists / (2 * std_input**2)
    # For numerical stability, subtract the maximum logit per sample.
    logits -= np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probs

def compute_bayes_error_for_trained_classifier(checkpoint_folder, data_folder, device=None):
    """
    Computes the Bayes misclassification probability for a trained classifier.
    
    The script:
      - Loads the trained classifier from checkpoint_folder (expects best_mlp.pth).
      - Loads the test dataset from data_folder (expects test_dataset.npz).
      - Loads the generating parameters (means and std) from dataset_stats.csv in data_folder.
      - Runs the trained classifier on the test set to obtain predictions f(x).
      - For each sample, computes the true posterior probabilities using the generating parameters.
      - Computes the Bayes misclassification probability for the classifier’s prediction:
      
            δ(x) = 1 - P(y = f(x) | x)
      
      - Saves the results (true label, predicted label, Bayes error) in a CSV file
        "bayes_detector_for_trained_classifier_results.csv" in data_folder.
    """
    # Ensure device is set.
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data configuration.
    data_config = load_data_config(data_folder)
    dim = data_config["dim"]
    n_classes = data_config["n_classes"]
    overlap = data_config["overlap"]
    seed = data_config["seed"]
    
    # Load generating parameters from dataset_stats.csv.
    means, std_from_stats = load_generating_params(data_folder)
    # Optionally, compare std_from_stats with overlap. Here we assume they match.
    std = std_from_stats
    
    # Load test dataset.
    test_path = os.path.join(data_folder, "test_dataset.npz")
    test_data = np.load(test_path)
    X_test = test_data["X"]
    y_true = test_data["y"]
    
    # Load the trained classifier from the checkpoint folder.
    model_path = os.path.join(checkpoint_folder, "best_mlp.pth")
    model = MLP(input_dim=dim, hidden_dims=[64, 32], num_classes=n_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Compute predictions from the trained classifier.
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, pred_labels = torch.max(outputs, 1)
    pred_labels = pred_labels.cpu().numpy()
    
    # Compute the true posterior probabilities.
    probs = bayes_proba(X_test, means, std)
    
    # For each sample, compute the Bayes misclassification probability for the trained classifier's prediction:
    # δ(x) = 1 - P(y = f(x) | x)
    bayes_error = 1 - probs[np.arange(len(X_test)), pred_labels]
    
    # Create a DataFrame with the results.
    results_df = pd.DataFrame({
        "true_label": y_true,
        "pred_label": pred_labels,
        "bayes_error": bayes_error
    })
    
    out_csv = os.path.join(data_folder, "bayes_detector_for_trained_classifier_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"Bayes detector results saved to: {out_csv}")
    return results_df

def main():
    # parser = argparse.ArgumentParser(
    #     description="Compute Bayes misclassification detector for a trained classifier."
    # )
    # parser.add_argument("--checkpoint_folder", type=str, required=True,
    #                     help="Path to the checkpoint folder (e.g., checkpoints/ce/mlp6432_synth/)")
    # parser.add_argument("--data_folder", type=str, required=True,
    #                     help="Path to the data folder (e.g., data/synthetic/dim-2_classes-3)")
    # args = parser.parse_args()

    checkpoint_folder = "checkpoints/ce/mlp6432_synth/"
    data_folder = "data/synthetic/dim-2_classes-3/"

    
    # Compute and save the Bayes misclassification detector results.
    compute_bayes_error_for_trained_classifier(checkpoint_folder, data_folder)

if __name__ == "__main__":
    main()
