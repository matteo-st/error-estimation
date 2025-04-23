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

def collect_experiment_folders(results_dir, fixed_params):
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


def main():
    parser = argparse.ArgumentParser(description="Plot experiment curves for fixed parameters and one free parameter.")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Base directory containing experiment folders (default: results)")

    parser.add_argument("--name_save_file", type=str, default="train_fold1",
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
        "init_scheme": "random",
        "f_divergence": "kl",
        # "kmeans_seed": 10,
        "kmens_n_iter": 50,
        "grad_n_iters": 100,
        "grad_lr": 0.1,
        "n_cluster": 150,
        "clustering_space": "probs",
        "alpha": 0.05,
        "n_folds": 4
        }
    # Remove the free parameter (if present) from the fixed parameters.

    # Collect experiment folders matching the fixed parameters.
    exp_folders = collect_experiment_folders(args.results_dir, fixed_params)
    if not exp_folders:
        print("No experiment folders found matching the fixed parameters.")
        return
    print("Found the following experiment folders:")
    for f in exp_folders:
        print(f)

if __name__ == "__main__":
    main()

