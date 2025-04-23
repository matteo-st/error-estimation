import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Import the MLP architecture from our utils.
from synthetic_code.utils.model import MLP

def load_dataset(data_dir, split):
    """
    Loads the synthetic dataset from data_dir.
    
    Expected file naming:
      - For split "train":           train.npz
      - For split "test":            test.npz
      - For split "train_detector":  train_detector_dataset.npz
      - For split "concentration":   concentration_dataset.npz
      - For split "test_dataset":    test_dataset.npz
    """
    mapping = {
        "train": "train.npz",
        "test": "test.npz",
        "train_detector": "train_detector_dataset.npz",
        "concentration": "concentration_dataset.npz",
        "test_dataset": "test_dataset.npz"
    }
    if split not in mapping:
        raise ValueError(f"Unknown split '{split}'. Allowed splits: {list(mapping.keys())}.")
    file_path = os.path.join(data_dir, mapping[split])
    data = np.load(file_path)
    X = data["X"]
    y = data["y"]
    return X, y

def evaluate_model(model, dataset, device, batch_size):
    """
    Evaluates the given model on the dataset and returns the accuracy.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    accuracy = correct / total
    return accuracy

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset.
    print(f"Loading dataset split '{args.split}' from {args.data_dir} ...")
    X, y = load_dataset(args.data_dir, args.split)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Create the model.
    # For the classifier, we use the provided number of classes.
    # For the error detector, the output dimension is fixed to 2.
    if args.model_type == "classifier":
        output_dim = args.n_classes
    elif args.model_type == "detector":
        output_dim = 2
    else:
        raise ValueError("Unknown model type. Use 'classifier' or 'detector'.")
    
    model = MLP(input_dim=args.dim, hidden_dims=args.hidden_dims, num_classes=output_dim)
    model.to(device)
    
    # Load model checkpoint.
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded {args.model_type} model from {args.model_path}")
    
    # Evaluate.
    accuracy = evaluate_model(model, dataset, device, args.batch_size)
    print(f"Evaluation on split '{args.split}': Accuracy = {accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a synthetic model (classifier or detector) on a dataset.")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["classifier", "detector"],
                        help="Type of model to evaluate: 'classifier' or 'detector'.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory of the synthetic dataset (e.g., synthetic_data/dim-2_class-3)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test", "train_detector", "concentration", "test_dataset"],
                        help="Dataset split to evaluate on.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model checkpoint file (e.g., best_mlp.pth or best_error_detector.pth)")
    parser.add_argument("--dim", type=int, default=2,
                        help="Input data dimension.")
    parser.add_argument("--n_classes", type=int, default=3,
                        help="Number of classes for the classifier (ignored for detector).")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[64, 32],
                        help="Hidden layer sizes for the MLP.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation.")
    
    args = parser.parse_args()
    main(args)
