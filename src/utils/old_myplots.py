# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import argparse

# # Parse command-line argument for alpha.
# parser = argparse.ArgumentParser()
# parser.add_argument("--alpha", type=float, required=True, default=0.05, help="Confidence level alpha to filter results")
# args = parser.parse_args()
# alpha_value = args.alpha

# # Path to your results file
# results_file = "results/results.csv"

# # Read the CSV file into a DataFrame
# df = pd.read_csv(results_file)

# # Filter for the conformal method (i.e. your EmbeddingKMeans method)
# # and for the specified alpha value.
# df_conformal = df[(df["method"] == "conformal") & (df["alpha"] == alpha_value)].copy()

# # Convert n_cluster to numeric (if needed) and sort by it.
# df_conformal["n_cluster"] = pd.to_numeric(df_conformal["n_cluster"])
# df_conformal = df_conformal.sort_values("n_cluster")

# # If you have multiple runs per n_cluster, average the FPR.
# df_grouped = df_conformal.groupby("n_cluster", as_index=False)["fpr"].mean()

# # Plot FPR versus n_cluster.
# plt.figure(figsize=(6, 4))
# plt.plot(df_grouped["n_cluster"], df_grouped["fpr"], marker="o", linestyle="-")
# plt.xlabel("Number of Clusters (n_cluster)")
# plt.ylabel("FPR")
# plt.title(f"FPR vs n_cluster - alpha = {alpha_value}")
# plt.grid(True)
# plt.tight_layout()

# # Save the figure to the images directory.
# os.makedirs("images", exist_ok=True)
# plot_filename = os.path.join("images", f"fpr_vs_n_cluster_alpha_{alpha_value}.pdf")
# plt.savefig(plot_filename)
# print(f"Plot saved to {plot_filename}")

# plt.show()
import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil

def parse_fixed_params(param_str):
    """
    Parses a comma-separated list of key=value pairs and returns a dictionary.
    Example input: "model_name=resnet34,dataset=cifar10,style=ce"
    """
    fixed_params = {}
    if param_str:
        pairs = param_str.split(",")
        for pair in pairs:
            if "=" in pair:
                key, value = pair.split("=", 1)
                fixed_params[key.strip()] = value.strip()
    return fixed_params

def create_analysis_folder(results_dir=None):
    """
    Creates a new analysis folder in the 'analysis' subdirectory of results_dir.
    The new folder is named "comparison_x", where x is one plus the highest existing number.
    
    Args:
        results_dir (str): Base results directory. If None, uses the RESULTS_DIR environment variable.
    
    Returns:
        str: Full path to the newly created analysis folder.
    """
    if results_dir is None:
        results_dir = os.environ.get("RESULTS_DIR", "results")
    analysis_dir = os.path.join(results_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # List existing comparison folders.
    existing = [
        d for d in os.listdir(analysis_dir)
        if os.path.isdir(os.path.join(analysis_dir, d)) and d.startswith("comparison_")
    ]
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
    Given an experiment folder, loads the configuration and classification CSV files,
    and returns the free parameter value and FPR values for train and test if the config
    matches the fixed parameters.
    
    Args:
        exp_dir (str): Full path to an experiment folder.
        free_param (str): Name of the free parameter to extract (e.g. "n_cluster").
        fixed_params (dict): Dictionary of fixed parameters that must match.
        
    Returns:
        tuple: (free_param_value, train_fpr, test_fpr) if experiment config matches fixed_params; otherwise, None.
    """
    config_file = os.path.join(exp_dir, "config.json")
    train_classif_file = os.path.join(exp_dir, "train_classif_results.csv")
    test_classif_file = os.path.join(exp_dir, "test_classif_results.csv")
    
    if not (os.path.exists(config_file) and os.path.exists(train_classif_file) and os.path.exists(test_classif_file)):
        return None

    with open(config_file, "r") as f:
        config = json.load(f)
    
    if fixed_params:
        for key, val in fixed_params.items():
            if str(config.get(key, "")) != str(val):
                return None

    free_val = config.get(free_param)
    
    try:
        train_df = pd.read_csv(train_classif_file)
        test_df = pd.read_csv(test_classif_file)
    except Exception as e:
        print(f"Error loading CSV in {exp_dir}: {e}")
        return None

    try:
        train_fpr = float(train_df.iloc[0]["fpr"])
        test_fpr = float(test_df.iloc[0]["fpr"])
    except Exception as e:
        print(f"Error extracting FPR in {exp_dir}: {e}")
        return None

    return free_val, train_fpr, test_fpr

def collect_experiment_data(results_dir, free_param="n_cluster", fixed_params=None):
    """
    Scans the experiments subdirectory for experiment folders, filters them by fixed parameters,
    and collects the free parameter and FPR values.
    
    Args:
        results_dir (str): Base directory containing experiment folders (e.g., "results").
        free_param (str): Name of the free parameter to use (e.g., "n_cluster").
        fixed_params (dict): Dictionary of fixed parameters to filter by.
        
    Returns:
        DataFrame: A DataFrame with columns [free_param, "train_fpr", "test_fpr"] sorted by free_param.
    """
    experiments_base = os.path.join(results_dir, "experiments")
    exp_dirs = [os.path.join(experiments_base, d) for d in os.listdir(experiments_base)
                if os.path.isdir(os.path.join(experiments_base, d)) and d.startswith("experiment_")]
    records = []
    for exp in exp_dirs:
        res = get_experiment_results(exp, free_param=free_param, fixed_params=fixed_params)
        if res is not None:
            records.append(res)
    if not records:
        print("No valid experiment records found with the specified fixed parameters.")
        return None
    df = pd.DataFrame(records, columns=[free_param, "train_fpr", "test_fpr"])
    try:
        df[free_param] = pd.to_numeric(df[free_param])
    except Exception:
        pass
    df.sort_values(by=free_param, inplace=True)
    return df

def plot_fpr_vs_free_param(df, free_param="n_cluster", save_path=None):
    """
    Plots train and test FPR vs. the free parameter.
    
    Args:
        df (DataFrame): DataFrame with columns [free_param, "train_fpr", "test_fpr"].
        free_param (str): Name of the free parameter.
        save_path (str): If provided, saves the plot to this file; otherwise, displays the plot.
    """
    x = df[free_param].values
    train_fpr = df["train_fpr"].values
    test_fpr = df["test_fpr"].values

    plt.figure(figsize=(10,6))
    plt.plot(x, train_fpr, marker="o", linestyle="-", color="blue", label="Train FPR")
    plt.plot(x, test_fpr, marker="o", linestyle="-", color="red", label="Test FPR")
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
    parser.add_argument("--free_param", type=str, default="n_cluster",
                        help="The free parameter to compare (e.g., n_cluster)")
    # parser.add_argument("--save", type=str, default="",
    #                     help="File path to save the plot (if empty, the plot will be shown)")
    args = parser.parse_args()

    # Fixed parameters provided directly in code.
    fixed_params = {
        "model_name": "resnet34",
        "dataset": "cifar10",
        "style": "ce",
        "temperature": 0,
        "magnitude": 0,
        "method": "conformal",
        "batch_size_train": 64,
        "batch_size_test": 64,
        "split_ratio": 1.5,
        "seed": 1,
        "lbd": 0.5,
        "clustering_method": "kmeans",
        "n_cluster": 200,  # This will be removed since it's the free parameter.
        "clustering_space": "probs",
        "alpha": 0.05
    }
    fixed_params.pop(args.free_param, None)

    # Collect experiment data.
    df = collect_experiment_data(args.results_dir, free_param=args.free_param, fixed_params=fixed_params)
    if df is None or df.empty:
        print("No data to plot.")
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

    # Plot the results.
    plot_save_path = os.path.join(analysis_folder, "comparison.png")
    plot_fpr_vs_free_param(df, free_param=args.free_param, save_path=plot_save_path)

if __name__ == "__main__":
    main()
