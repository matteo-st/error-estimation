import json
import os
from typing import Any, Dict
import shutil


import pandas as pd


def str_to_dict(string: str) -> Dict[str, Any]:
    return json.loads(string)


def append_results_to_file(results, filename="results.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if isinstance(results, dict):
        results = {k: [v] for k, v in results.items()}
        results = pd.DataFrame.from_dict(results, orient="columns")
    print(f"Saving results to {filename}")
    # df_pa_table = pa.Table.from_pandas(results)
    if not os.path.isfile(filename):
        results.to_csv(filename, header=True, index=False)
    else:  # it exists, so append without writing the header
        results.to_csv(filename, mode="a", header=False, index=False)


def create_experiment_folder(config_path="config.json", results_dir=None):
    """
    Creates a new experiment folder in the specified results directory.
    The new folder is named "experiment_x" where x is one plus the highest existing number.
    Also copies the configuration file into this folder for reproducibility.
    
    Args:
        config_path (str): Path to the configuration JSON file.
        results_dir (str): Path to the results directory. If None, defaults to the environment 
                           variable RESULTS_DIR or "results/".
    
    Returns:
        str: The path to the newly created experiment folder.
    """
    if results_dir is None:
        results_dir = os.environ.get("RESULTS_DIR", "results/")
    results_dir = os.path.join(results_dir, "experiments")
    os.makedirs(results_dir, exist_ok=True)
    
    # List existing experiment folders
    existing = [
        f for f in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, f)) and f.startswith("experiment_")
    ]
    # Extract experiment numbers.
    numbers = []
    for folder in existing:
        try:
            number = int(folder.split("_")[1])
            numbers.append(number)
        except (IndexError, ValueError):
            continue
    new_number = max(numbers) + 1 if numbers else 1
    experiment_folder = os.path.join(results_dir, f"experiment_{new_number}")
    os.makedirs(experiment_folder, exist_ok=True)
    
    # Archive the configuration file in the experiment folder.
    shutil.copyfile(config_path, os.path.join(experiment_folder, "config.json"))
    print(f"Experiment folder created: {experiment_folder}")
    print(f"Config file copied to: {os.path.join(experiment_folder, 'config.json')}")
    print(10 * "---")
    
    return experiment_folder