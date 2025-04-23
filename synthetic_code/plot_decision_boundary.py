import os
import json
import numpy as np
import torch
from synthetic_code.utils.model import MLP
from synthetic_code.utils.visualization import plot_decision_boundary

def load_dataset(data_dir, dataset_name):
    data = np.load(os.path.join(data_dir, dataset_name))
    X = data["X"]
    y = data["y"]
    return X, y

if __name__ == "__main__":
    # Change the folder as needed.
    seed = 1
    dim = 2
    num_classes = 3
    dataset_name = "test_dataset.npz"
    model_name = "mlp6432_synth"
    data_dir = f"data/synthetic/dim-{dim}_classes-{num_classes}"  
    model_dir = os.path.join("checkpoints/ce", model_name) 
    np.random.seed(seed)

    # Load dataset configuration.
    with open(os.path.join(data_dir, "config.json"), "r") as f:
        config = json.load(f)
    if dim != 2:
        raise ValueError("Decision boundary visualization is only available for 2D data.")
    
    # Load the training data.
    X, y = load_dataset(data_dir, dataset_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create and load the trained model.
    model = MLP(input_dim=dim, hidden_dims=[64, 32], num_classes=num_classes)
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_mlp.pth"), map_location=device))
    model.to(device)
    
    # Plot the decision boundary.
    # You can adjust grid_points and sample_fraction to balance resolution and clarity.
    plot_decision_boundary(model, X, y, device=device,
                           grid_points=400, sample_fraction=0.05,
                           title=f"Decision Boundary of {model_name} on {dataset_name}",
                           save_path=os.path.join(model_dir, "decision_boundary.png"))
