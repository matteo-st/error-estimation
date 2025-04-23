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
    
    # List existing analysis folders
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
        # Compare as strings for robustness.
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

def plot_centroid_convergence_multi(exp_folders, free_param, name_save_file, outer_iteration, cluster, save_folder):
    """
    For each experiment folder, loads the centroid convergence CSV file,
    filters the data by outer_iteration and cluster, and plots loss vs. inner iteration.
    Each curve is labeled with the free parameter value from its config.
    
    The resulting figure is saved in the save_folder.
    """
    plt.figure(figsize=(10, 6))
    for folder in exp_folders:
        free_val = load_param(folder, free_param)
        # if float(free_val) < 0.005:
        #     continue
        centroid_csv = os.path.join(folder, f"{name_save_file}_centroid_convergence.csv")
        if not os.path.exists(centroid_csv):
            print(f"WARNING: {centroid_csv} not found in {folder}. Skipping.")
            continue

        df = pd.read_csv(centroid_csv)
        # Filter rows by the specified outer_iteration and cluster.
        df = df[(df["outer_iteration"] == outer_iteration) & (df["cluster"] == cluster)]
        if df.empty:
            print(f"WARNING: No data for outer_iteration={outer_iteration} and cluster={cluster} in {centroid_csv}.")
            continue
        plt.loglog(df["inner_iteration"], df["loss"], marker="o", label=f"{free_param}={free_val}")

    plt.xlabel("Inner iteration")
    plt.ylabel("Loss")
    plt.title(f"Centroid Convergence (outer_it={outer_iteration}, cluster={cluster})")
    plt.legend()
    plt.tight_layout()

    out_file = os.path.join(save_folder, f"{name_save_file}_centroid_convergence_outer-{outer_iteration}_cluster-{cluster}.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Centroid convergence plot saved to {out_file}")

def plot_kmeans_loss_multi(exp_folders, free_param, name_save_file, save_folder):
    """
    For each experiment folder, loads the k-means loss history CSV file
    and plots kmeans_loss vs. outer_iteration.
    Each curve is labeled with the free parameter value from its config.
    
    The resulting figure is saved in the save_folder.
    """
    plt.figure(figsize=(8, 5))
    for folder in exp_folders:
        free_val = load_param(folder, free_param)
        # Optionally, skip experiments with free_val below a threshold.
        # if float(free_val) < 0.005:
        #     continue
        kmeans_csv = os.path.join(folder, f"{name_save_file}_kmeans_loss_history.csv")
        if not os.path.exists(kmeans_csv):
            print(f"WARNING: {kmeans_csv} not found in {folder}. Skipping.")
            continue

        df = pd.read_csv(kmeans_csv)
        if df.empty:
            print(f"WARNING: No data in {kmeans_csv}. Skipping.")
            continue
        plt.loglog(df["outer_iteration"], df["kmeans_loss"], marker="o", label=f"{free_param}={free_val}")

    plt.xlabel("Outer iteration")
    plt.ylabel("KMeans Loss")
    plt.title("KMeans Loss History (Multi-Experiment)")
    plt.legend()
    plt.tight_layout()

    out_file = os.path.join(save_folder, f"{name_save_file}_kmeans_loss_history.png")
    plt.savefig(out_file)
    plt.close()
    print(f"KMeans loss history plot saved to {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Plot experiment curves for fixed parameters and one free parameter.")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Base directory containing experiment folders (default: results)")
    parser.add_argument("--free_param", type=str, default="kmeans_seed",
                        help="The free parameter to compare (e.g., grad_lr)")
    parser.add_argument("--name_save_file", type=str, default="train_fold0",
                        help="CSV file prefix (e.g., train_fold1)")
    parser.add_argument("--outer_iteration", type=int, default=0,
                        help="The outer iteration value to filter centroid convergence data (default: 0)")
    parser.add_argument("--cluster", type=int, default=0,
                        help="The cluster value to filter centroid convergence data (default: 0)")
    args = parser.parse_args()

    # Define fixed parameters (update these as needed)
    fixed_params = {
  "model_name": "resnet34",
  "dataset": "cifar10",
  "style": "ce",
  "temperature": 100,
  "magnitude": 0,
  "method": "conformal",
  "batch_size_train": 64,
  "batch_size_test": 64,
  "split_ratio": 1.5,
  "seed": 1,
  "lbd": 0.5,
  "clustering_method": "kmeans_grad",
  "init_scheme":"random",
  "f_divergence": "kl",
  "kmens_n_iter": 50,
  "grad_n_iters": 100,
  "grad_lr": 0.1,
  "n_cluster": 150,
  "clustering_space": "probs",
  "alpha": 0.05,
  "n_folds": 4
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

    # Save analysis parameters for documentation in the analysis subfolder.
    analysis_params = {
        "free_param": args.free_param,
        "fixed_params": fixed_params
    }
    analysis_info_file = os.path.join(analysis_subfolder, "analysis_params.json")
    with open(analysis_info_file, "w") as f:
        json.dump(analysis_params, f, indent=2)
    print(f"Analysis parameters saved to {analysis_info_file}")

    # Plot centroid convergence curves (multiple experiments in one figure)
    plot_centroid_convergence_multi(exp_folders, args.free_param, args.name_save_file,
                                    args.outer_iteration, args.cluster, analysis_subfolder)

    # Plot k-means loss history curves (multiple experiments in one figure)
    plot_kmeans_loss_multi(exp_folders, args.free_param, args.name_save_file, analysis_subfolder)

if __name__ == "__main__":
    main()
