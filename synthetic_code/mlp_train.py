import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
import numpy as np
from tqdm import tqdm
import os
import json
import datetime
import pandas as pd
from synthetic_code.utils.dataset import GaussianMixtureDataset
from synthetic_code.utils.model import BayesClassifier, MLPClassifier



# -------------------------------
# 2. MLP Classifier for Mixture-of-Gaussians
# -------------------------------


# -------------------------------
# 3. Evaluator: Compute Accuracy
# -------------------------------
class Evaluator:
    def __init__(self, model, dataloader, device):
        """
        Evaluator for measuring model accuracy.
        
        Args:
            model (nn.Module): The trained classifier.
            dataloader (DataLoader): DataLoader for evaluation data.
            device (torch.device): Device to perform evaluation on.
        """
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x, labels in self.dataloader:
                # x: [batch_size, dim], labels: [batch_size]
                x = x.to(self.device)
                labels = labels.to(self.device)
                logits, _ = self.model(x)  # logits: [batch_size, num_classes]
                preds = torch.argmax(logits, dim=1)  # [batch_size]
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        accuracy = correct / total
        return accuracy, torch.cat(all_preds), torch.cat(all_labels)

# -------------------------------
# 4. Trainer: Training Loop and Model Optimization
# -------------------------------
class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, device, checkpoints_dir, epochs=20, lr=1e-3):
        """
        Trainer for the MLP classifier.
        
        Args:
            model (nn.Module): The classifier to be trained.
            train_dataloader (DataLoader): Training data loader.
            val_dataloader (DataLoader): Validation data loader.
            device (torch.device): Device for computation.
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()  # Expects logits: [batch_size, num_classes] & labels: [batch_size]
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.checkpoint_dir = checkpoints_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self):

        best_accuracy = 0.0
        results = [] 
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for x, labels in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                # x: [batch_size, dim], labels: [batch_size]
                x = x.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                logits, _ = self.model(x)  # logits: [batch_size, num_classes]
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * x.size(0)
            epoch_loss = running_loss / len(self.train_dataloader.dataset)
            
            # Evaluate on validation set after each epoch.
            evaluator = Evaluator(self.model, self.val_dataloader, self.device)
            val_accuracy, _, _ = evaluator.evaluate()
            results.append({"epoch": epoch+1, "loss": epoch_loss, "val_accuracy": val_accuracy})
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {epoch_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")

            if best_accuracy < val_accuracy:
                best_accuracy = val_accuracy
                print(f"Model saved at epoch {epoch+1} with accuracy {val_accuracy:.4f}")           
                # Save checkpoint if current accuracy is best so far.
                checkpoint_path = os.path.join(self.checkpoint_dir, "best_mlp.pth")
                torch.save(self.model.state_dict(), checkpoint_path)

        print(f"Training completed. Best Test Accuracy: {best_accuracy:.4f}")
            # Save training results to a CSV file.
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.checkpoint_dir, "training_results.csv"), index=False)


# -------------------------------
# 5. Main Pipeline: Data, Model, Training and Evaluation
# -------------------------------
def main():
    # Set random seeds for reproducibility.
    seed = 107
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Mixture-of-Gaussians parameters.
    dim = 10         # Feature dimension: each sample is [10]
    n_classes = 7    # Mixture has 7 components/classes.
    n_samples_train = 20000
    n_samples_val = 20000
    lr = 1e-3
    epochs = 30
    batch_size_train = 32
    batch_size_val = 32
    
    # Generate mixture parameters.
    # means: [7, 10]
    means = torch.rand(n_classes, dim)
    # stds: [7, 10]
    stds = torch.rand(n_classes, dim)
    # weights: [7]
    weights = torch.rand(n_classes)
    weights = weights / weights.sum()  # Normalize to sum to 1.

    config = {
        "dim": dim,
        "n_classes": n_classes,
        "n_samples_train": n_samples_train,
        "n_samples_val": n_samples_val,
        "lr": lr,
        "epochs": epochs,
        "means": means.tolist(),
        "stds": stds.tolist(),
        "weights": weights.tolist(),
        "hidden_dims": [128, 128],
        "num_hidden_layers": 2,
        "dropout_p": 0.2,
        "batch_size": 32,
        "seed": seed
    }

    checkpoint_dir = os.path.join("checkpoints/ce", f"mlp_synth_dim-{dim}_classes-{n_classes}")
    # Create the checkpoint folder.

    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Add the current date and time to the configuration.
    config["trained_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save the training configuration for reference in the checkpoint folder.
    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # Create training and validation datasets.
    train_dataset = GaussianMixtureDataset(n_samples_train, means, stds, weights)
    val_dataset = GaussianMixtureDataset(n_samples_val, means, stds, weights)
    
    # Create DataLoaders.
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)  # Each batch: [32, 10]
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)       # Each batch: [32, 10]
    
    # Instantiate the MLP classifier.
    model = MLPClassifier(input_dim=dim, hidden_size=128, num_hidden_layers=2, dropout_p=0.2, num_classes=n_classes)
    
    # Define device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Compute Bayes classifier
    covs = torch.diag_embed(stds ** 2)
    bayes_model = BayesClassifier(means, covs, weights)
    bayes_evaluator = Evaluator(bayes_model, val_loader, device)
    bayes_accuracy, _, _ = bayes_evaluator.evaluate()
    print(f"Bayes Accuracy: {bayes_accuracy:.4f}")

    # Create and run the trainer.
    trainer = Trainer(model, train_loader, val_loader, device, checkpoint_dir, epochs=epochs, lr=lr)
    trainer.train()
    
    # Finally, evaluate the model.
    evaluator = Evaluator(model, val_loader, device)
    accuracy, preds, labels = evaluator.evaluate()
    print(f"Final Validation Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
