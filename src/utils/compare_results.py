import os
import json
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

def collect_experiment_comparison_data(results_dir):
    """
    Scans the results/experiments directory, reads each experiment folder,
    and collects all configuration parameters along with the mean and std for 
    each accuracy metric (fpr, auc, aurc) from train and test CSV files.
    
    For fpr, the mean is computed as before. The standard deviation for fpr is computed as:
      - If all folds provided a var_fpr value: new_std = sqrt(sum(var_fpr) / (n_fold^2))
      - Otherwise: new_std = standard deviation of the fpr values.
    For auc and aurc, mean and std are computed in the standard way.
    
    Args:
        results_dir (str): Base directory containing the experiments folder.
        
    Returns:
        DataFrame: DataFrame where each row corresponds to an experiment and 
                   columns include all configuration parameters and the computed metrics.
    """
    experiments_base = os.path.join(results_dir, "experiments")
    experiment_dirs = [
        os.path.join(experiments_base, d)
        for d in os.listdir(experiments_base)
        if os.path.isdir(os.path.join(experiments_base, d)) and d.startswith("experiment_")
    ]
    
    records = []
    
    for exp_dir in tqdm(experiment_dirs, desc="Processing experiment directories"):
        record = {}
        config_path = os.path.join(exp_dir, "config.json")
        if not os.path.exists(config_path):
            print(f"Config file not found in {exp_dir}. Skipping...")
            continue
        
        with open(config_path, "r") as f:
            config = json.load(f)
        record.update(config)
        
        # Define the metrics to collect.
        metrics = ["fpr", "var_fpr", "auc", "aurc"]
        train_metrics = {metric: [] for metric in metrics}
        test_metrics = {metric: [] for metric in metrics}
        
        # Process train fold CSVs.
        train_files = glob.glob(os.path.join(exp_dir, "train_fold*_classif_results.csv"))
        for file in tqdm(train_files, desc=f"Processing train files in {os.path.basename(exp_dir)}", leave=False):
            try:
                df = pd.read_csv(file)
                row = df.iloc[0]
                for metric in metrics:
                    if metric in row:
                        train_metrics[metric].append(float(row[metric]))
            except Exception as e:
                print(f"Error processing train file {file}: {e}")
                
        # Process test fold CSVs.
        test_files = glob.glob(os.path.join(exp_dir, "test_fold*_classif_results.csv"))
        for file in tqdm(test_files, desc=f"Processing test files in {os.path.basename(exp_dir)}", leave=False):
            try:
                df = pd.read_csv(file)
                row = df.iloc[0]
                for metric in metrics:
                    if metric in row:
                        test_metrics[metric].append(float(row[metric]))
            except Exception as e:
                print(f"Error processing test file {file}: {e}")
        
        # Number of folds for train and test
        n_train = len(train_files) if train_files else 1
        n_test = len(test_files) if test_files else 1
        
        # Process fpr metrics for train.
        if train_metrics["fpr"]:
            record["train_fpr_mean"] = np.mean(train_metrics["fpr"])
            # Use new method if all folds provided var_fpr; else fallback.
            if len(train_metrics["var_fpr"]) == len(train_metrics["fpr"]) and len(train_metrics["var_fpr"]) > 0:
                record["train_fpr_new_std"] = np.sqrt(np.sum(train_metrics["var_fpr"]) / (n_train ** 2))
            else:
                record["train_fpr_new_std"] = None
            record["train_fpr_std"] = np.std(train_metrics["fpr"])
        else:
            record["train_fpr_mean"] = None
            record["train_fpr_new_std"] = None
            record["train_fpr_std"] = None

        # Process fpr metrics for test.
        if test_metrics["fpr"]:
            record["test_fpr_mean"] = np.mean(test_metrics["fpr"])
            if len(test_metrics["var_fpr"]) == len(test_metrics["fpr"]) and len(test_metrics["var_fpr"]) > 0:
                record["test_fpr_new_std"] = np.sqrt(np.sum(test_metrics["var_fpr"]) / (n_test ** 2))
            else:
                record["test_fpr_new_std"] = None
            record["test_fpr_std"] = np.std(test_metrics["fpr"])
        else:
            record["test_fpr_mean"] = None
            record["test_fpr_new_std"] = None
            record["test_fpr_std"] = None

        # Process auc and aurc metrics: standard mean and std.
        for metric in ["auc", "aurc"]:
            if train_metrics[metric]:
                record[f"train_{metric}_mean"] = np.mean(train_metrics[metric])
                record[f"train_{metric}_std"] = np.std(train_metrics[metric])
            else:
                record[f"train_{metric}_mean"] = None
                record[f"train_{metric}_std"] = None

            if test_metrics[metric]:
                record[f"test_{metric}_mean"] = np.mean(test_metrics[metric])
                record[f"test_{metric}_std"] = np.std(test_metrics[metric])
            else:
                record[f"test_{metric}_mean"] = None
                record[f"test_{metric}_std"] = None
        
        records.append(record)
    
    if not records:
        print("No experiment records found.")
        return None
    
    df = pd.DataFrame(records)
    if "test_fpr_mean" in df.columns:
        df.sort_values(by="test_fpr_mean", inplace=True, na_position="last")
    return df

def main():
    # Base directory, e.g., "results". You can also set the environment variable RESULTS_DIR.
    results_dir = os.environ.get("RESULTS_DIR", "results")
    
    # Collect data from all experiment folders.
    df = collect_experiment_comparison_data(results_dir)
    if df is None or df.empty:
        print("No data to write to CSV.")
        return
    
    # Create the output directory: results/analysis/tables.
    tables_dir = os.path.join(results_dir, "analysis", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    
    output_csv = os.path.join(tables_dir, "table_comparison.csv")
    df.to_csv(output_csv, index=False)
    print(f"Comparison table saved to: {output_csv}")
    
if __name__ == "__main__":
    main()
