import os
import argparse
import json
import random
import numpy as np
import torch
import torch.utils.data
from argparse import Namespace
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# These imports assume you already have these functions in your codebase.
# You may need to adjust the paths depending on your project structure.
from src.RelU.methods import get_method
from src.utils.helpers import create_experiment_folder, append_results_to_file
from src.utils.eval import evaluate_classification

# For our toy experiment, we load the dataset from our npz files.
def load_toy_dataset(data_dir, split="test"):
    """
    Loads the toy dataset (saved as an npz file) from data_dir.
    Assumes files are named "train.npz" and "test.npz".
    """
    file_path = os.path.join(data_dir, f"{split}.npz")
    data = np.load(file_path)
    X = data["X"]
    y = data["y"]
    return X, y

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    
    # For misclassification detection on the toy data, we use only the test set.
    data_dir = config.data_dir  # e.g., "synthetic_data/dim-2_classes-3"
    X, y = load_toy_dataset(data_dir, split="test")
    
    # Create a PyTorch Dataset from the numpy arrays.
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    full_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    
    # Load the trained model.
    from synthetic_code.utils.model import MLP
    model = MLP(input_dim=X.shape[1], hidden_dims=config.hidden_dims, num_classes=config.n_classes)
    model_path = os.path.join(data_dir, "best_mlp.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Compute error labels (misclassification indicator) over the full test set.
    batch_size = config.batch_size  # you can set this in your config
    full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    all_error_labels = []
    for inputs, targets in tqdm(full_loader, desc="Computing predictions for stratification"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            # error: True if misclassified, False otherwise.
            error_labels = (preds != targets).cpu().numpy()
        all_error_labels.append(error_labels)
    all_error_labels = np.concatenate(all_error_labels, axis=0)
    
    # Use StratifiedKFold (with 2 splits) to split the test set into new train and test sets.
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=config.seed)
    
    # For each split, we will fit our misclassification detection method on the new train
    # and evaluate it on the new test.
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.arange(len(full_dataset)), all_error_labels)):
        print(f"Processing misclassification detection fold {fold+1} ...")
        
        # Create training and test subsets.
        fold_train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        fold_test_dataset = torch.utils.data.Subset(full_dataset, test_idx)
        
        train_loader = torch.utils.data.DataLoader(fold_train_dataset, batch_size=config.batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(fold_test_dataset, batch_size=config.batch_size, shuffle=False)
        
        # Get the misclassification detection method.
        # The method is assumed to have a fit() and evaluate() interface.
        method = get_method(config.method, model, **vars(config))
        
        if hasattr(method.method, "fit"):
            print(3 * "---", "Fitting misclassification detection method", 3 * "---")
            train_preds, train_targets, train_scores = method.fit(train_loader, None, name_save_file=f"misd_train_fold{fold}")
            evaluate_classification(train_preds, train_targets, train_scores, config.results_folder,
                                    name_save_file=f"misd_train_fold{fold}")
        
        print(3 * "---", "Evaluating misclassification detection method", 3 * "---")
        test_preds, test_targets, test_scores = method.evaluate(test_loader, name_save_file=f"misd_test_fold{fold}")
        evaluate_classification(test_preds, test_targets, test_scores, config.results_folder,
                                name_save_file=f"misd_test_fold{fold}")
    
    print("End of misclassification detection experiment.")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Path to the configuration file for the toy misclassification detection experiment.
    config_path = os.path.join(current_dir, "config.json")
    # Create a new experiment folder (for saving results) based on the config.
    experiment_folder = create_experiment_folder(config_path)
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    config["results_folder"] = experiment_folder
    # Ensure the data directory is defined (e.g., "synthetic_data/dim-2_classes-3")
    # and any other required parameters (e.g., n_classes, hidden_dims, batch_size, seed, method, etc.)
    config = Namespace(**config)
    main(config)
