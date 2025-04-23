import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

# Reuse the MLP architecture defined in synthetic_code/utils/model.py
from synthetic_code.utils.model import MLP

def load_dataset(data_dir, split="train"):
    """
    Loads the synthetic dataset from data_dir.
    
    Allowed splits:
      - "train"           : primary classifier training set (if needed)
      - "train_detector"  : dataset for training the error detector
      - "concentration"   : dataset for concentration fitting
      - "test_dataset"    : dataset for evaluating the error detector
      
    Expects files named accordingly (e.g., "train_detector_dataset.npz", "test_dataset.npz", etc.)
    """
    mapping = {
        "train": "train.npz",
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

def create_error_detection_dataset(data_dir, model, device, split="train_detector", batch_size=32):
    """
    Loads the dataset indicated by `split` (for training the error detector) from data_dir
    and uses the provided primary classification model to compute binary error labels
    (1 if misclassified, 0 if correctly classified).
    
    Returns:
        X (np.ndarray): Input features.
        error_labels (np.ndarray): Binary error indicators.
    """
    X, y = load_dataset(data_dir, split=split)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    all_error_labels = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            error = (preds != targets).cpu().numpy().astype(np.int64)
            all_error_labels.append(error)
    all_error_labels = np.concatenate(all_error_labels, axis=0)
    return X, all_error_labels

def create_experiment_folder(config, base_dir="synthetic_results"):
    """
    Creates a new experiment folder to store logs and checkpoints.
    The folder structure is:
    
        synthetic_results/dim-{dim}_class-{n_classes}/mlp_detector/experiment_{x}
    
    The configuration is saved as config.json in that folder.
    """
    folder_path = os.path.join(
        base_dir,
        f"dim-{config['dim']}_class-{config['n_classes']}",
        "mlp_detector"
    )
    os.makedirs(folder_path, exist_ok=True)
    
    experiments = [
        d for d in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, d)) and d.startswith("experiment_")
    ]
    numbers = []
    for folder in experiments:
        try:
            num = int(folder.split("_")[1])
            numbers.append(num)
        except Exception:
            continue
    new_number = max(numbers) + 1 if numbers else 1
    new_folder = os.path.join(folder_path, f"experiment_{new_number}")
    os.makedirs(new_folder, exist_ok=True)
    
    # Save the configuration.
    config_path = os.path.join(new_folder, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"Experiment folder created: {new_folder}")
    return new_folder

def evaluate_error_detector(error_detector, data_dir, device, batch_size, split="test_dataset"):
    """
    Evaluates the trained error detector on the specified dataset (e.g., test_dataset).
    Returns the accuracy.
    """
    X, true_errors = load_dataset(data_dir, split=split)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, torch.tensor(true_errors, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    error_detector.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = error_detector(inputs)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    acc = correct / total
    return acc

def train_error_detector(data_dir, main_model_path, config):
    """
    Trains an MLP to detect when the primary classifier makes an error.
    The error detector learns from the original features and the binary error indicator.
    Training is performed on the dataset "train_detector_dataset.npz" and evaluation is done on "test_dataset.npz".
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the primary classifier.
    primary_model = MLP(input_dim=config["dim"], hidden_dims=config["hidden_dims"], num_classes=config["n_classes"])
    primary_model.load_state_dict(torch.load(main_model_path, map_location=device))
    primary_model.to(device)
    
    # Build the error detection training dataset from "train_detector_dataset.npz".
    X, error_labels = create_error_detection_dataset(data_dir, primary_model, device, split="train_detector", batch_size=config["batch_size"])
    print(f"Error detection training dataset: {X.shape[0]} samples")
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    error_tensor = torch.tensor(error_labels, dtype=torch.long)
    full_dataset = TensorDataset(X_tensor, error_tensor)
    
    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(config["seed"])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # Define the error detector model (2 output classes).
    error_detector = MLP(input_dim=config["dim"], hidden_dims=config["hidden_dims"], num_classes=2)
    error_detector.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(error_detector.parameters(), lr=config["lr_error"])
    num_epochs = config["num_epochs_error"]
    
    best_val_acc = 0.0
    best_state = None
    for epoch in range(num_epochs):
        error_detector.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = error_detector(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
        epoch_loss = running_loss / n_train
        
        # Evaluate on validation set.
        error_detector.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = error_detector(batch_X)
                _, preds = torch.max(outputs, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        val_acc = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = error_detector.state_dict()
    
    # Create an experiment folder and save configuration.
    experiment_folder = create_experiment_folder(config)
    config["results_folder"] = experiment_folder
    
    # Save the best error detector in the experiment folder.
    save_path = os.path.join(experiment_folder, "best_error_detector.pth")
    torch.save(best_state, save_path)
    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Error detector saved to {save_path}")
    
    # Evaluate the error detector on the test dataset ("test_dataset.npz").
    test_acc = evaluate_error_detector(error_detector, data_dir, device, batch_size=config["batch_size"], split="test_dataset")
    print(f"Error detector test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    # Define the configuration for the error detector.
    config = {
        "data_dir": "synthetic_data/dim-2_classes-3",  # Folder with your synthetic dataset.
        "dim": 2,                # Data dimension.
        "n_classes": 3,          # Number of classes in the original classification.
        "hidden_dims": [64, 32], # Architecture parameters for the MLPs.
        "batch_size": 32,
        "seed": 42,
        "lr_error": 0.001,       # Learning rate for the error detector.
        "num_epochs_error": 50   # Number of epochs for training the error detector.
    }
    
    # Path to the primary classifier model checkpoint.
    main_model_path = os.path.join(config["data_dir"], "best_mlp.pth")
    
    print("Starting training of error detector...")
    train_error_detector(config["data_dir"], main_model_path, config)
