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
        
                
        # Process test fold CSVs.
        df = pd.read_csv(os.path.join(exp_dir, "detector_results.csv"))
        row = df.iloc[0]
        record.update(row.to_dict())
  
        
        records.append(record)
    
    if not records:
        print("No experiment records found.")
        return None
    
    df = pd.DataFrame(records)
    if "fpr" in df.columns:
        df.sort_values(by="fpr", inplace=True, na_position="last")
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
