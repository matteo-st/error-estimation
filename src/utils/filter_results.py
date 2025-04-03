import os
import glob
import pandas as pd

def filter_table_comparison(table_path, filter_params):
    """
    Reads the table CSV and filters its rows based on the given parameters.
    
    Args:
        table_path (str): Full path to table_comparison.csv.
        filter_params (dict): Dictionary of parameters to filter by.
    
    Returns:
        DataFrame: Filtered DataFrame.
    """
    df = pd.read_csv(table_path)
    # Filter the dataframe: for each parameter, only keep rows that match exactly.
    for key, value in filter_params.items():
        if key in df.columns:
            df = df[df[key] == value]
        else:
            # If the column doesn't exist, no row can match the parameter.
            df = df[df.index < 0]  # yields an empty dataframe.
    return df

def get_next_comparison_filename(tables_dir):
    """
    Determines the next available filename for a filtered comparison CSV.
    
    Args:
        tables_dir (str): Directory where comparison CSV files are stored.
        
    Returns:
        str: Full path for the next comparison file (comparison_x.csv).
    """
    pattern = os.path.join(tables_dir, "comparison_*.csv")
    files = glob.glob(pattern)
    max_number = 0
    for file in files:
        basename = os.path.basename(file)
        try:
            num = int(basename.split("_")[1].split(".")[0])
            if num > max_number:
                max_number = num
        except Exception:
            continue
    return os.path.join(tables_dir, f"comparison_{max_number+1}.csv")

def main():
    # --- Set the filter parameters here ---
    filter_params = {
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
    "clustering_method": "kmeans_grad",
    # "init_scheme":"random",
    "f_divergence": "kl",
    # "kmens_n_iter": 50,
    # "grad_n_iters": 100,
    # "grad_lr": 0.1,
    "n_cluster": 150,
    "clustering_space": "probs",
    "alpha": 0.05,
    "n_folds": 4
    }
    
    # --- Define the paths ---
    # Base directory, either from the environment variable or default to "results"
    base_dir = os.environ.get("RESULTS_DIR", "results")
    tables_dir = os.path.join(base_dir, "analysis", "tables")
    table_csv_path = os.path.join(tables_dir, "table_comparison.csv")
    
    if not os.path.exists(table_csv_path):
        print(f"File {table_csv_path} does not exist. Make sure the table file is created first.")
        return
    
    # --- Filter the table ---
    filtered_df = filter_table_comparison(table_csv_path, filter_params)
    
    if filtered_df.empty:
        print("No experiments match the specified parameters.")
        return
    else:
        # --- Determine the output file name ---
        output_file = get_next_comparison_filename(tables_dir)
        filtered_df.to_csv(output_file, index=False)
        print(f"Filtered results saved to: {output_file}")

if __name__ == "__main__":
    main()
