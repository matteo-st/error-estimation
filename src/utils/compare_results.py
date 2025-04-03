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
    
    # Process each experiment folder with tqdm progress bar.
    for exp_dir in tqdm(experiment_dirs, desc="Processing experiment directories"):
        record = {}
        config_path = os.path.join(exp_dir, "config.json")
        if not os.path.exists(config_path):
            print(f"Config file not found in {exp_dir}. Skipping...")
            continue
        
        with open(config_path, "r") as f:
            config = json.load(f)
        record.update(config)
        
        metrics = ["fpr", "auc", "aurc"]
        train_metrics = {metric: [] for metric in metrics}
        test_metrics = {metric: [] for metric in metrics}
        
        # Process train fold CSVs with tqdm.
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
                
        # Process test fold CSVs with tqdm.
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
        
        # Compute mean and std for each metric.
        for metric in metrics:
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
    # Sort the DataFrame by increasing order of test_fpr_mean.
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
