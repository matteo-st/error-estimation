import os
import json
import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_analysis_subfolder(results_dir):
    """
    Creates a new subfolder under results/analysis/images/ named analysis_x,
    where x is one plus the highest existing number.
    
    Args:
        results_dir (str): Base directory (e.g., "results")
        
    Returns:
        str: Path to the newly created subfolder.
    """
    analysis_images = os.path.join(results_dir, "analysis", "images")
    os.makedirs(analysis_images, exist_ok=True)
    
    # List existing analysis folders.
    existing = [d for d in os.listdir(analysis_images)
                if os.path.isdir(os.path.join(analysis_images, d)) and d.startswith("analysis_")]
    numbers = []
    for folder in existing:
        try:
            num = int(folder.split("_")[1])
            numbers.append(num)
        except (IndexError, ValueError):
            continue
    new_number = max(numbers) + 1 if numbers else 1
    new_folder = os.path.join(analysis_images, f"analysis_{new_number}")
    os.makedirs(new_folder, exist_ok=True)
    print(f"Created new analysis folder: {new_folder}")
    return new_folder

def load_param(exp_folder, param_key):
    """
    Loads the parameter value (as a string) from the config.json file
    located in exp_folder.
    Returns the parameter value or 'NA' if not found.
    """
    config_path = os.path.join(exp_folder, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        return str(config.get(param_key, "NA"))
    return "NA"

def experiment_matches(exp_folder, fixed_params):
    """
    Checks whether the experiment in exp_folder matches all the fixed parameters.
    
    Args:
        exp_folder (str): Path to the experiment folder.
        fixed_params (dict): Dictionary of fixed parameters.
        
    Returns:
        bool: True if all fixed parameters match; False otherwise.
    """
    config_path = os.path.join(exp_folder, "config.json")
    if not os.path.exists(config_path):
        return False
    with open(config_path, "r") as f:
        config = json.load(f)
    for key, val in fixed_params.items():
        if str(config.get(key, "")) != str(val):
            return False
    return True

def collect_experiment_folders(results_dir, free_param, fixed_params):
    """
    Scans the results/experiments folder and returns a list of experiment folders
    whose config matches the fixed parameters.
    
    Args:
        results_dir (str): Base directory containing experiments (e.g., "results").
        free_param (str): The free parameter key.
        fixed_params (dict): Dictionary of fixed parameters.
        
    Returns:
        list: List of experiment folder paths.
    """
    experiments_base = os.path.join(results_dir, "experiments")
    exp_folders = []
    if not os.path.exists(experiments_base):
        print(f"Experiments folder {experiments_base} not found.")
        return exp_folders

    # Look for folders starting with "experiment_"
    for d in os.listdir(experiments_base):
        path = os.path.join(experiments_base, d)
        if os.path.isdir(path) and d.startswith("experiment_"):
            if experiment_matches(path, fixed_params):
                exp_folders.append(path)
    return exp_folders

def get_fpr(exp_folder, fold):
    """
    Loads the classification results for the specified fold from exp_folder and returns train and test FPR.
    It expects CSV files named 'train_fold{fold}_classif_results.csv' and 'test_fold{fold}_classif_results.csv'
    with a column "fpr".
    """
    train_file = os.path.join(exp_folder, f"train_fold{fold}_classif_results.csv")
    test_file = os.path.join(exp_folder, f"test_fold{fold}_classif_results.csv")
    try:
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
    except Exception as e:
        print(f"Error loading FPR files in {exp_folder}: {e}")
        return None, None

    try:
        train_fpr = float(df_train.iloc[0]["fpr"])
        test_fpr = float(df_test.iloc[0]["fpr"])
    except Exception as e:
        print(f"Error processing FPR values in {exp_folder}: {e}")
        return None, None

    return train_fpr, test_fpr

def get_final_kmeans_loss(exp_folder, fold):
    """
    Loads the k-means loss history CSV from exp_folder and returns the final kmeans_loss value.
    It first looks for a CSV file named 'train_fold{fold}_kmeans_loss_history.csv' with columns
    "outer_iteration" and "kmeans_loss". If that file is not found, it looks for a file named
    'train_fold{fold}_cluster_inertia.csv', which is expected to contain a single inertia value.
    
    Args:
        exp_folder (str): Path to the experiment folder.
        fold (str): The fold number as a string.
        
    Returns:
        The final k-means loss (inertia) value, or None if no valid file is found.
    """
    # First try the kmeans loss history file.
    kmeans_csv_path = os.path.join(exp_folder, f"train_fold{fold}_kmeans_loss_history.csv")
    if os.path.exists(kmeans_csv_path):
        try:
            df = pd.read_csv(kmeans_csv_path)
            if df.empty:
                return None
            # Ensure data is sorted by outer_iteration and take the last kmeans_loss value.
            df_sorted = df.sort_values(by="outer_iteration")
            final_loss = df_sorted.iloc[-1]["kmeans_loss"]
            return final_loss
        except Exception as e:
            print(f"Error reading {kmeans_csv_path}: {e}")
            return None
    else:
        print(f"{kmeans_csv_path} not found in {exp_folder}.")
    
    # If the first file is not found, try the cluster inertia file.
    inertia_csv_path = os.path.join(exp_folder, f"train_fold{fold}_cluster_inertia.csv")
    if os.path.exists(inertia_csv_path):
        try:
            # Assuming the file contains a single value; we read without headers.
            df = pd.read_csv(inertia_csv_path, header=None)
            if df.empty:
                return None
            final_loss = df.iloc[1, 0]
            print("final loss", final_loss)
            return final_loss
        except Exception as e:
            print(f"Error reading {inertia_csv_path}: {e}")
            return None
    else:
        print(f"{inertia_csv_path} not found in {exp_folder}.")
    
    return None

def plot_fpr_vs_kmeans_loss(exp_folders, fold, save_folder, free_param):
    """
    For each experiment folder, extracts train and test FPR from classification results and
    the final kmeans_loss from the k-means loss history.
    Groups experiments by free parameter value, and plots a scatter plot with:
      - x-axis: final kmeans_loss (inertie)
      - y-axis: FPR (with different markers for train and test)
    Each free parameter value is mapped to a unique color.
    
    The resulting figure is saved in save_folder with the fold number in its name.
    Returns a list of experiment names used.
    """
    # Group data by free parameter value.
    data_by_free = {}
    exp_names_by_free = {}
    for folder in exp_folders:
        train_fpr, test_fpr = get_fpr(folder, fold)
        loss = get_final_kmeans_loss(folder, fold)
        free_val = load_param(folder, free_param)
        # Skip experiments where the free parameter is "opt"
        if free_val.lower() == "opt":
            print(f"Skipping {folder} because {free_param} is set to 'opt'.")
            continue
        if train_fpr is None or test_fpr is None or loss is None:
            print(f"Skipping {folder} due to missing data.")
            continue
        if free_val not in data_by_free:
            data_by_free[free_val] = {"loss": [], "train_fpr": [], "test_fpr": []}
            exp_names_by_free[free_val] = []
        data_by_free[free_val]["loss"].append(loss)
        data_by_free[free_val]["train_fpr"].append(train_fpr)
        data_by_free[free_val]["test_fpr"].append(test_fpr)
        exp_names_by_free[free_val].append(os.path.basename(folder))
        print(f"Experiment {os.path.basename(folder)}: {free_param}={free_val}, final loss={loss}, train FPR={train_fpr}, test FPR={test_fpr}")

    if not data_by_free:
        print("No valid data points found for plotting.")
        return []

    # Get a colormap with as many distinct colors as free parameter values.
    unique_free_vals = sorted(data_by_free.keys())
    cmap = plt.cm.get_cmap("tab10", len(unique_free_vals))

    plt.figure(figsize=(8, 6))
    # For each free parameter group, plot train and test FPR with different markers.
    for i, free_val in enumerate(unique_free_vals):
        color = cmap(i)
        losses = data_by_free[free_val]["loss"]
        train_fprs = data_by_free[free_val]["train_fpr"]
        test_fprs = data_by_free[free_val]["test_fpr"]
        # Plot train FPR with circle markers.
        plt.scatter(losses, train_fprs, marker="o", color=color, edgecolors='k', s=80,
                    label=f"Train, {free_param}={free_val}")
        # Plot test FPR with square markers.
        plt.scatter(losses, test_fprs, marker="s", color=color, edgecolors='k', s=80,
                    label=f"Test, {free_param}={free_val}")

    plt.xlabel("Final KMeans Loss (Inertie)")
    plt.ylabel("FPR")
    plt.title(f"FPR vs Final KMeans Loss (Fold {fold})")
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.8, 1])  # Adjust layout to leave space on the right.

    # Place the legend outside the plot area.
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

    out_file = os.path.join(save_folder, f"fpr_vs_kmeans_loss_fold{fold}.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Scatter plot saved to {out_file}")

    # Combine all experiment names used.
    used_exp_names = []
    for names in exp_names_by_free.values():
        used_exp_names.extend(names)
    return used_exp_names

def main():
    parser = argparse.ArgumentParser(description="Plot Train & Test FPR vs Final KMeans Loss for selected experiments.")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Base directory containing experiment folders (default: results)")
    parser.add_argument("--free_param", type=str, default="kmeans_seed",
                        help="The free parameter to compare (e.g., kmeans_seed, grad_lr)")
    parser.add_argument("--fold", type=str, default="0",
                        help="Fold number to use for CSV file names (e.g., 0 for train_fold0)")
    args = parser.parse_args()

    # Define fixed parameters (update these as needed)
    fixed_params = {
        "model_name": "resnet34",
        "dataset": "cifar10",
        "style": "ce",
        "temperature": 260,
        "magnitude": 0,
        "method": "conformal",
        "batch_size_train": 64,
        "batch_size_test": 64,
        "split_ratio": 1.5,
        "seed": 1,
        "lbd": 0.5,
        "clustering_method": "kmeans",
        "init_scheme": "random",
        "f_divergence": "eucli",
        "kmeans_seed": 10,
        "kmens_n_iter": 50,
        "grad_n_iters": 130,
        "grad_lr": 0.05,
        "n_cluster": 170,
        "clustering_space": "probs",
        "alpha": 0.05,
        "n_folds": 4,
        "cross_val": "stratify"
        }
    # Remove the free parameter (if present) from the fixed parameters.
    fixed_params.pop(args.free_param, None)
    print("Filtering experiments with fixed parameters:")
    print(fixed_params)
    print(f"Using free parameter: {args.free_param}")

    # Collect experiment folders matching the fixed parameters.
    exp_folders = collect_experiment_folders(args.results_dir, args.free_param, fixed_params)
    if not exp_folders:
        print("No experiment folders found matching the fixed parameters.")
        return
    print("Found the following experiment folders:")
    for f in exp_folders:
        print(f)

    # Create a new analysis subfolder under results/analysis/images/analysis_x
    analysis_subfolder = create_analysis_subfolder(args.results_dir)

    # Plot the scatter: x = final kmeans_loss, y = FPR (train & test) for the given fold.
    used_exp_names = plot_fpr_vs_kmeans_loss(exp_folders, args.fold, analysis_subfolder, args.free_param)
    if used_exp_names is None:
        used_exp_names = []

    # Save analysis parameters for documentation in the analysis subfolder.
    analysis_params = {
        "free_param": args.free_param,
        "fixed_params": fixed_params,
        "experiment_names": used_exp_names
    }
    analysis_info_file = os.path.join(analysis_subfolder, "analysis_params.json")
    with open(analysis_info_file, "w") as f:
        json.dump(analysis_params, f, indent=2)
    print(f"Analysis parameters saved to {analysis_info_file}")

if __name__ == "__main__":
    main()

