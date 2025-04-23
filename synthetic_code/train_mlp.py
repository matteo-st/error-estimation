import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from synthetic_code.utils.model import MLP
import datetime

def load_dataset(data_dir):
    """
    Loads train and test sets from npz files.
    
    Expects that:
      - Training set is stored in "train.npz"
      - Test set is stored in "test_dataset.npz"
    """
    train_path = os.path.join(data_dir, "train.npz")
    test_path = os.path.join(data_dir, "test_dataset.npz")
    train_npz = np.load(train_path)
    test_npz = np.load(test_path)
    X_train = train_npz["X"]
    y_train = train_npz["y"]
    X_test = test_npz["X"]
    y_test = test_npz["y"]
    return (X_train, y_train), (X_test, y_test)

def train_model(data_dir, config, checkpoint_dir):
    # Load dataset
    (X_train, y_train), (X_test, y_test) = load_dataset(data_dir)
    
    # Convert to PyTorch tensors.
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_dim = X_train.shape[1]
    num_classes = config["n_classes"]
    
    model = MLP(input_dim=input_dim, hidden_dims=config["hidden_dims"], num_classes=num_classes)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    num_epochs = config["num_epochs"]
    best_acc = 0.0
    results = []  # To store epoch-wise loss and accuracy.
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Evaluate on test set.
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, preds = torch.max(outputs, 1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        acc = correct / total
        results.append({"epoch": epoch+1, "loss": epoch_loss, "accuracy": acc})
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Test Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            # Save checkpoint if current accuracy is best so far.
            checkpoint_path = os.path.join(checkpoint_dir, "best_mlp.pth")
            torch.save(model.state_dict(), checkpoint_path)
    
    print(f"Training completed. Best Test Accuracy: {best_acc:.4f}")
    # Save training results to a CSV file.
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(checkpoint_dir, "training_results.csv"), index=False)

if __name__ == "__main__":
    # Set configuration data for the dataset.
    config_data = {
        "dim": 2,
        "n_classes": 3
    }
    # Data directory where the synthetic dataset is stored.
    data_dir = f"synthetic_data/dim-{config_data['dim']}_classes-{config_data['n_classes']}"
    
    # Define training configuration parameters.
    hidden_dims = [64, 32]
    hidden_dims_str = "".join(str(x) for x in hidden_dims)  # e.g. "6432"
    seed = 42
    config = {
        "seed": seed,
        "batch_size": 32,
        "lr": 0.001,
        "num_epochs": 50,
        "n_classes": config_data["n_classes"],
        "hidden_dims": hidden_dims
    }
    
    # Create the checkpoint folder: checkpoints/mlp{hidden_dims_str}_synth
    checkpoint_dir = os.path.join("checkpoints/ce", f"mlp{hidden_dims_str}_synth")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Add the current date and time to the configuration.
    config["trained_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save the training configuration for reference in the checkpoint folder.
    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # Set the seed.
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    train_model(data_dir, config, checkpoint_dir)
