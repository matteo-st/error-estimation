import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_cluster_data(csv_path):
    """
    Loads cluster results from a CSV file and ensures a 'cluster_id' column.
    """
    df = pd.read_csv(csv_path)
    # If the first column is unnamed, rename it to 'cluster_id'
    if "Unnamed: 0" in df.columns:
        df.rename(columns={"Unnamed: 0": "cluster_id"}, inplace=True)
    else:
        df["cluster_id"] = df.index
    return df

def plot_cluster_means(df, save_path):
    """
    Plots cluster error means with error bars representing standard deviation (sqrt of variance).
    The DataFrame is sorted by the 'cluster_means' column.
    
    Args:
        df (DataFrame): DataFrame containing at least 'cluster_id', 'cluster_means', and 'cluster_vars'.
        save_path (str): Full file path to save the plot.
    """
    # Sort by error means.
    df_sorted = df.sort_values("cluster_means")
    # Compute standard deviation from variance.
    std_vals = np.sqrt(df_sorted["cluster_vars"])
    # Create sequential x positions.
    x = np.arange(len(df_sorted))
    
    plt.figure(figsize=(8, 6))
    plt.bar(x, df_sorted["cluster_means"], yerr=std_vals, color="green",
            alpha=0.7, capsize=5)
    plt.xlabel("Cluster (sorted by mean error)")
    plt.ylabel("Error Mean")
    plt.title("Cluster Error Means with Variance")
    # Set x-ticks to show the original cluster IDs.
    plt.xticks(x, df_sorted["cluster_id"])
    plt.savefig(save_path)
    plt.close()
    print(f"Cluster means plot saved to: {save_path}")

def plot_cluster_counts(df, save_path):
    """
    Plots the number of samples (counts) per cluster.
    
    Args:
        df (DataFrame): DataFrame containing at least 'cluster_id' and 'cluster_counts'.
        save_path (str): Full file path to save the plot.
    """
    # Sort by cluster id (or keep original order).
    df_sorted = df.sort_values("cluster_id")
    x = np.arange(len(df_sorted))
    
    plt.figure(figsize=(8, 6))
    plt.bar(x, df_sorted["cluster_counts"], color="blue", alpha=0.7)
    plt.xlabel("Cluster")
    plt.ylabel("Counts")
    plt.title("Cluster Counts")
    plt.xticks(x, df_sorted["cluster_id"])
    plt.savefig(save_path)
    plt.close()
    print(f"Cluster counts plot saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Display experiment cluster results")
    parser.add_argument("--experiment_folder", type=str, required=True,
                        help="Name of the experiment folder (e.g. experiment_1)")
    args = parser.parse_args()
    
    # Assume environment variable RESULTS_DIR is set; if not, default to 'results'
    results_dir = os.environ.get("RESULTS_DIR", "results")
    # Build the full path to the experiment folder.
    experiment_folder = os.path.join(results_dir, args.experiment_folder)
    
    # Load train and test cluster CSV files.
    train_csv = os.path.join(experiment_folder, "train_cluster_results.csv")
    test_csv = os.path.join(experiment_folder, "test_cluster_results.csv")
    
    try:
        train_df = load_cluster_data(train_csv)
        test_df = load_cluster_data(test_csv)
    except Exception as e:
        print("Error loading cluster CSV files:", e)
        return

    # Plot for training set.
    train_means_path = os.path.join(experiment_folder, "train_cluster_means_with_vars.png")
    train_counts_path = os.path.join(experiment_folder, "train_cluster_counts.png")
    plot_cluster_means(train_df, train_means_path)
    plot_cluster_counts(train_df, train_counts_path)

    # Plot for test set.
    test_means_path = os.path.join(experiment_folder, "test_cluster_means_with_vars.png")
    test_counts_path = os.path.join(experiment_folder, "test_cluster_counts.png")
    plot_cluster_means(test_df, test_means_path)
    plot_cluster_counts(test_df, test_counts_path)

    # Print the classification results from CSV files.
    train_classif_csv = os.path.join(experiment_folder, "train_classif_results.csv")
    test_classif_csv = os.path.join(experiment_folder, "test_classif_results.csv")
    try:
        train_classif_df = pd.read_csv(train_classif_csv)
        test_classif_df = pd.read_csv(test_classif_csv)
    except Exception as e:
        print("Error loading classification results:", e)
        return

    print("=== Train Classification Results ===")
    print(train_classif_df.to_string(index=False))
    print("\n=== Test Classification Results ===")
    print(test_classif_df.to_string(index=False))

if __name__ == "__main__":
    main()
