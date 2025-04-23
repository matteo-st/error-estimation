import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
import glob

def create_analysis_folder(results_dir):
    """
    Creates a new analysis folder in the 'analysis' subdirectory of results_dir.
    The new folder is named "comparison_x" where x is one plus the highest existing number.
    
    Args:
        results_dir (str): Base results directory.
    
    Returns:
        str: Full path to the newly created analysis folder.
    """
    analysis_dir = os.path.join(results_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # List existing comparison folders.
    existing = [d for d in os.listdir(analysis_dir)
                if os.path.isdir(os.path.join(analysis_dir, d)) and d.startswith("comparison_")]
    numbers = []
    for folder in existing:
        try:
            number = int(folder.split("_")[1])
            numbers.append(number)
        except (IndexError, ValueError):
            continue
    new_number = max(numbers) + 1 if numbers else 1
    new_folder = os.path.join(analysis_dir, f"comparison_{new_number}")
    os.makedirs(new_folder, exist_ok=True)
    print(f"Analysis folder created: {new_folder}")
    return new_folder

def get_experiment_results(exp_dir, free_param="n_cluster", fixed_params=None):
    """
    Given an experiment folder, loads the configuration and classification CSV files
    (from multiple folds) and returns the free parameter value plus the mean and standard deviation 
    of the FPR values for train and test if the config matches the fixed parameters.
    
    Args:
        exp_dir (str): Full path to an experiment folder.
        free_param (str): Name of the free parameter to extract (e.g. "n_cluster").
        fixed_params (dict): Dictionary of fixed parameters that must match.
        
    Returns:
        tuple: (free_val, train_fpr_mean, train_fpr_std, test_fpr_mean, test_fpr_std) 
               if the experiment config matches fixed_params; otherwise, None.
    """
    config_file = os.path.join(exp_dir, "config.json")
    # Use glob to match all fold files.
    train_files = glob.glob(os.path.join(exp_dir, "train_fold*_classif_results.csv"))
    test_files  = glob.glob(os.path.join(exp_dir, "test_fold*_classif_results.csv"))
    
    # if not (os.path.exists(config_file) and train_files and test_files):
    if not (os.path.exists(config_file) and test_files):
        print("bwe", exp_dir)
        return None

    with open(config_file, "r") as f:
        config = json.load(f)
    
    if fixed_params:
        for key, val in fixed_params.items():
            if str(config.get(key, "")) != str(val):
                print(str(config.get(key, "")), val, key)
                return None

    free_val = config.get(free_param)
    
    # Collect FPR values from all fold files.
    train_fprs = []
    for file in train_files:
        try:
            df = pd.read_csv(file)
            train_fpr = float(df.iloc[0]["fpr"])
            train_fprs.append(train_fpr)
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    test_fprs = []
    for file in test_files:
        try:
            df = pd.read_csv(file)
            test_fpr = float(df.iloc[0]["fpr"])
            test_fprs.append(test_fpr)
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # if not train_fprs or not test_fprs:
    if not test_fprs:
        return None

    train_mean = np.mean(train_fprs)
    train_std  = np.std(train_fprs)
    test_mean  = np.mean(test_fprs)
    test_std   = np.std(test_fprs)
    
    return free_val, train_mean, train_std, test_mean, test_std

def collect_experiment_data(results_dir, free_param="n_cluster", fixed_params=None):
    """
    Scans the experiments subdirectory for experiment folders, filters them by fixed parameters,
    and collects the free parameter and FPR mean/std values.
    
    Args:
        results_dir (str): Base directory containing experiment folders (e.g., "results").
        free_param (str): The free parameter to use (e.g., "n_cluster").
        fixed_params (dict): Fixed parameters to filter by.
        
    Returns:
        DataFrame: A DataFrame with columns [free_param, "train_fpr_mean", "train_fpr_std", 
                 "test_fpr_mean", "test_fpr_std"], sorted by free_param.
    """
    experiments_base = os.path.join(results_dir, "experiments")
    exp_dirs = [os.path.join(experiments_base, d) for d in os.listdir(experiments_base)
                if os.path.isdir(os.path.join(experiments_base, d)) and d.startswith("experiment_")]
    exp_dirs = []
    for d in os.listdir(experiments_base):
        path = os.path.join(experiments_base, d)
        if os.path.isdir(path) and d.startswith("experiment_"):
            try:
                # Extract the numeric part after "experiment_"
                num = int(d.split("_")[1])
            except (IndexError, ValueError):
                print(d)
                continue
            if num > 783:
                exp_dirs.append(path)
    print(exp_dirs)
    records = []
    for exp in exp_dirs:
        res = get_experiment_results(exp, free_param=free_param, fixed_params=fixed_params)
        if res is not None:
            records.append(res)
    if not records:
        print("No valid experiment records found with the specified fixed parameters.")
        return None
    df = pd.DataFrame(records, columns=[free_param, "train_fpr_mean", "train_fpr_std", "test_fpr_mean", "test_fpr_std"])
    try:
        df[free_param] = pd.to_numeric(df[free_param])
    except Exception:
        pass
    df.sort_values(by=free_param, inplace=True)
    return df

def plot_fpr_vs_free_param(df, free_param="n_cluster", save_path=None):
    """
    Plots train and test FPR vs. the free parameter with error bars showing standard deviation.
    
    Args:
        df (DataFrame): DataFrame with columns [free_param, "train_fpr_mean", "train_fpr_std", "test_fpr_mean", "test_fpr_std"].
        free_param (str): The free parameter name.
        save_path (str): If provided, saves the plot to this file; otherwise, displays the plot.
    """
    x = df[free_param].values
    train_mean = df["train_fpr_mean"].values
    train_std = df["train_fpr_std"].values
    test_mean = df["test_fpr_mean"].values
    test_std = df["test_fpr_std"].values

    plt.figure(figsize=(10,6))
    plt.errorbar(x, train_mean, yerr=train_std, fmt='o-', color='blue', capsize=5, label="Train FPR")
    plt.errorbar(x, test_mean, yerr=test_std, fmt='o-', color='red', capsize=5, label="Test FPR")
    plt.xlabel(free_param)
    plt.ylabel("False Positive Rate")
    plt.title(f"FPR vs {free_param}")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compare experiment performance by free parameter")
    parser.add_argument("--results_dir", type=str, default=os.environ.get("RESULTS_DIR", "results"),
                        help="Base directory containing experiment folders")
    parser.add_argument("--free_param", type=str, default="temperature",
                        help="The free parameter to compare (e.g., n_cluster)")
    parser.add_argument("--save", type=str, default="",
                        help="File path to save the plot (if empty, the plot will be shown)")
    args = parser.parse_args()

    # Fixed parameters are defined directly in the file.
    # for temperature in [1., 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8.5]:
    fixed_params = {
            "model_name": "resnet34",
            "dataset": "cifar10",
            "style": "ce",
            # "temperature": 100,
            "magnitude": 0,
            "method": "conformal",
            "batch_size_train": 64,
            "batch_size_test": 64,
            "split_ratio": 1.5,
            "seed": 1,
            "lbd": 0.5,
            "clustering_method": "kmeans",
            # "init_scheme":"random",
            # "f_divergence": "kl",
            # "kmens_n_iter": 50,
            # "grad_n_iters": 100,
            # "grad_lr": 0.05,
            "n_cluster": 170,
            "clustering_space": "probs",
            "alpha": 0.05,
            "n_folds": 4
            }
    fixed_params.pop(args.free_param, None)
    print("Filtering experiments with fixed parameters:")
    print(fixed_params)
    print(f"Using free parameter: {args.free_param}")

    # Collect experiment data.
    print(args.results_dir)
    df = collect_experiment_data(args.results_dir, free_param=args.free_param, fixed_params=fixed_params)
    if df is None or df.empty:
        print("No experiment folders found matching the fixed parameters.")
        return

    print("Collected experiment data:")
    print(df.to_string(index=False))


    # Create a new analysis folder.
    analysis_folder = create_analysis_folder(args.results_dir)
    # Save the fixed and free parameters used.
    analysis_info = {
        "free_param": args.free_param,
        "fixed_params": fixed_params
    }
    json_save_path = os.path.join(analysis_folder, "comparison_params.json")
    with open(json_save_path, "w") as f:
        json.dump(analysis_info, f, indent=2)
    print(f"Comparison parameters saved to: {json_save_path}")

    # Plot and save the results.
    plot_save_path = os.path.join(analysis_folder, "comparison.png")
    plot_fpr_vs_free_param(df, free_param=args.free_param, save_path=plot_save_path)

if __name__ == "__main__":
    main()
