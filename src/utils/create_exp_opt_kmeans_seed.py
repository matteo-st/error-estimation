import os
import json
import argparse
import glob
import pandas as pd
import numpy as np
import shutil

def create_new_experiment_folder(results_dir):
    """
    Creates a new experiment folder under results/experiments/ named experiment_x,
    where x is one plus the highest existing number.
    
    Args:
        results_dir (str): Base directory (e.g., "results")
        
    Returns:
        str: Path to the newly created experiment folder.
    """
    new_base = os.path.join(results_dir, "experiments")
    os.makedirs(new_base, exist_ok=True)
    
    # List existing experiment folders.
    existing = [d for d in os.listdir(new_base)
                if os.path.isdir(os.path.join(new_base, d)) and d.startswith("experiment_")]
    numbers = []
    for folder in existing:
        try:
            num = int(folder.split("_")[1])
            numbers.append(num)
        except (IndexError, ValueError):
            continue
    new_number = max(numbers) + 1 if numbers else 1
    new_folder = os.path.join(new_base, f"experiment_{new_number}")
    os.makedirs(new_folder, exist_ok=True)
    print(f"Created new experiment folder: {new_folder}")
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
        free_param (str): The free parameter key (here, "kmeans_seed").
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
    Loads classification results for a given fold from exp_folder and returns train and test FPR.
    Expects files: train_fold{fold}_classif_results.csv and test_fold{fold}_classif_results.csv.
    """
    train_file = os.path.join(exp_folder, f"train_fold{fold}_classif_results.csv")
    test_file = os.path.join(exp_folder, f"test_fold{fold}_classif_results.csv")
    try:
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
        train_fpr = float(df_train.iloc[0]["fpr"])
        test_fpr = float(df_test.iloc[0]["fpr"])
        return train_fpr, test_fpr
    except Exception as e:
        print(f"Error processing FPR files in {exp_folder} for fold {fold}: {e}")
        return None, None

def get_final_kmeans_loss(exp_folder, fold):
    """
    Loads the k-means loss history CSV from exp_folder for a given fold and returns the final kmeans_loss.
    Expects file: train_fold{fold}_kmeans_loss_history.csv.
    """
    kmeans_csv_path = os.path.join(exp_folder, f"train_fold{fold}_kmeans_loss_history.csv")
    if not os.path.exists(kmeans_csv_path):
        print(f"{kmeans_csv_path} not found in {exp_folder}")
        return None
    try:
        df = pd.read_csv(kmeans_csv_path)
        if df.empty:
            return None
        df_sorted = df.sort_values(by="outer_iteration")
        final_loss = df_sorted.iloc[-1]["kmeans_loss"]
        return final_loss
    except Exception as e:
        print(f"Error reading {kmeans_csv_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="For a set of fixed parameters, select for each fold the experiment (varying only in kmeans_seed) with lowest inertie, create a report CSV, and copy optimal experiment files."
    )
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Base directory containing experiments (default: results)")
    parser.add_argument("--free_param", type=str, default="kmeans_seed",
                        help="The free parameter to vary (default: kmeans_seed)")
    parser.add_argument("--fold_range", type=str, default="0,1,2,3",
                        help="Comma-separated list of fold numbers (default: '0,1,2,3')")
    args = parser.parse_args()

    # Define fixed parameters (update as needed)
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
        "kmens_n_iter": 50,
        # "grad_n_iters": 100,
        "grad_lr": 0.1,
        "n_cluster": 150,
        "clustering_space": "probs",
        "alpha": 0.05,
        "n_folds": 4
    }
    # Remove the free parameter from the fixed parameters if it exists.
    fixed_params.pop(args.free_param, None)

    print("Filtering experiments with fixed parameters:")
    print(fixed_params)
    print(f"Free parameter: {args.free_param}")

    # Collect experiment folders matching the fixed parameters.
    exp_folders = collect_experiment_folders(args.results_dir, args.free_param, fixed_params)
    if not exp_folders:
        print("No experiment folders found matching the fixed parameters.")
        return

    # Ensure experiments differ only by kmeans_seed.
    # If a config doesn't include kmeans_seed, assume it's "1".
    experiments_by_seed = {}
    for folder in exp_folders:
        seed_val = load_param(folder, args.free_param)
        if seed_val == "NA":
            seed_val = "1"
        if seed_val in experiments_by_seed:
            print(f"Warning: Duplicate experiment for {args.free_param}={seed_val} found. Ignoring {folder}.")
            continue
        experiments_by_seed[seed_val] = folder

    # For each fold, collect results from each experiment.
    fold_range = [x.strip() for x in args.fold_range.split(",")]
    results_list = []
    for fold in fold_range:
        for seed_val, folder in experiments_by_seed.items():
            loss = get_final_kmeans_loss(folder, fold)
            train_fpr, test_fpr = get_fpr(folder, fold)
            if loss is None or train_fpr is None or test_fpr is None:
                print(f"Skipping folder {folder} for fold {fold} due to missing data.")
                continue
            results_list.append({
                "fold": fold,
                args.free_param: seed_val,
                "final_inertie": loss,
                "train_fpr": train_fpr,
                "test_fpr": test_fpr,
                "exp_folder": folder
            })
            print(f"Fold {fold}, {args.free_param}={seed_val}: final inertie={loss}, train_fpr={train_fpr}, test_fpr={test_fpr}")

    if not results_list:
        print("No valid results found.")
        return

    # Create a DataFrame from the results, and sort each fold by final_inertie.
    df = pd.DataFrame(results_list)
    df["fold"] = df["fold"].astype(int)
    df.sort_values(by=["fold", "final_inertie"], inplace=True)
    
    # Create the new optimal experiment folder under results/experiments/
    new_experiment_folder = create_new_experiment_folder(args.results_dir)
    
    # Save the CSV report in the new folder.
    analysis_report_csv = os.path.join(new_experiment_folder, "analysis_report.csv")
    df.to_csv(analysis_report_csv, index=False)
    print(f"Analysis report saved to {analysis_report_csv}")

    # For each fold, select the experiment with the lowest final inertie.
    optimal_by_fold = {}
    for fold, group in df.groupby("fold"):
        best_row = group.iloc[0]
        optimal_by_fold[fold] = best_row
        print(f"Optimal for fold {fold}: {args.free_param}={best_row[args.free_param]}, inertie={best_row['final_inertie']}")

    # For each fold, copy all files related to that fold from the optimal experiment folder.
    for fold, row in optimal_by_fold.items():
        src_folder = row["exp_folder"]
        pattern = os.path.join(src_folder, f"*fold{fold}*.csv")
        files = glob.glob(pattern)
        if not files:
            print(f"No files found for fold {fold} in {src_folder}")
            continue
        for fpath in files:
            try:
                shutil.copy2(fpath, new_experiment_folder)
                print(f"Copied {fpath} to {new_experiment_folder}")
            except Exception as e:
                print(f"Error copying {fpath}: {e}")
    
    # Instead of copying multiple config files, copy one config file and update kmeans_seed to "opt".
    # We pick the config from one of the optimal experiments (e.g., the one from the first fold).
    first_fold = sorted(optimal_by_fold.keys())[0]
    src_config = os.path.join(optimal_by_fold[first_fold]["exp_folder"], "config.json")
    if os.path.exists(src_config):
        try:
            with open(src_config, "r") as f:
                config_data = json.load(f)
            # Update the free parameter value to "opt".
            config_data[args.free_param] = "opt"
            dst_config = os.path.join(new_experiment_folder, "config.json")
            with open(dst_config, "w") as f:
                json.dump(config_data, f, indent=2)
            print(f"Copied and updated config file to {dst_config}")
        except Exception as e:
            print(f"Error processing config file {src_config}: {e}")
    else:
        print(f"Config file {src_config} not found. Skipping config copy.")

if __name__ == "__main__":
    main()
