import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
import numpy as np
from tqdm import tqdm
import os
import json
import datetime
import random
import pandas as pd
from synthetic_code.utils.datasets import GaussianMixtureDataset
from synthetic_code.utils.models import BayesClassifier, MLPClassifier, resnet



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
                logits = self.model(x)  # logits: [batch_size, num_classes]
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
                # logits, _ = self.model(x)  # logits: [batch_size, num_classes]
                logits = self.model(x)  # logits: [batch_size, num_classes]
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

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Mixture-of-Gaussians parameters.
    dim = 3072        # Feature dimension: each sample is [10]
    n_classes = 20    # Mixture has 7 components/classes.
    mu_scale = 0.1
    cov_scale = 1
    seed_parameters = 42  # Seed for generating means and covariances.
    n_samples_train = 50000
    n_samples_val = 20000
    lr = 1e-3
    epochs = 10
    batch_size_train = 252
    batch_size_val = 252
    model_name = "resnet34"

    
    # Generate mixture parameters.
    # means: [7, 10]
    # means = torch.rand(n_classes, dim)
    gen = torch.Generator().manual_seed(seed_parameters)
    means = torch.rand(n_classes, dim, generator = gen) * mu_scale   # Scale the means to control their spread.
    # stds: [7, 10]
    # 1) Sample random A for each component
    A = torch.randn(n_classes, dim, dim, generator = gen) * cov_scale 
    # 2) Form Σ = A @ Aᵀ
    print("symmetic covariance matrices")
    covs =  A @ A.transpose(-2, -1)
    covs = 0.5 * (covs + covs.transpose(-2, -1))
    # 3) (Optional) add small diagonal for numerical stability
    eps = 1e-3
    covs += eps * torch.eye(dim).unsqueeze(0) 
    print("end of symmetric covariance matrices")
    # print("covs shape:", covs.shape)  # Should be [n_classes, dim, dim]
    # eigs = torch.linalg.eigvalsh(covs)
    # print("eigs shape", eigs.shape)
    # for i in range(n_classes):
    #     print("Class", i, "eigenvalues:", eigs[i].min())
    

    # stds = torch.rand(n_classes, dim)
    # weights: [7]
    weights = torch.rand(n_classes, generator = gen)
    weights = weights / weights.sum()  # Normalize to sum to 1.

    config = {
        "data" : {
            "dim": dim,
            "n_classes": n_classes,
            "mu_scale" : mu_scale,
            "cov_scale": cov_scale,
            "seed_parameters": seed_parameters,
            "n_samples_train": n_samples_train,
            "n_samples_val": n_samples_val,
            "seed_train": 107,
            "seed_val": -107,
            "means": means.tolist(),
            "covs": covs.tolist(), # std before
            "weights": weights.tolist(),
        },
        "training": {
            "lr": lr,
            "epochs": epochs,
            "dropout_p": 0.2,
            "batch_size_train": batch_size_train,
            "batch_size_val": batch_size_val,
        },
        "model": {
            "name": model_name,
            # "hidden_dims": [128, 128],
            "num_hidden_layers": 2
            },
        
        "seed": seed
    }

    checkpoint_dir = os.path.join("checkpoints/ce", f"{model_name}_synth_dim-{dim}_classes-{n_classes}")
    # # Create the checkpoint folder.
    print(f"Creating checkpoint directory: {checkpoint_dir}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Add the current date and time to the configuration.
    config["trained_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save the training configuration for reference in the checkpoint folder.
    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # Create training and validation datasets.
    print("Creating datasets...")
    train_dataset = GaussianMixtureDataset(n_samples_train, means, covs, weights, 
                                        #    seed=config["data"]["seed_train"]
                                           )
    val_dataset = GaussianMixtureDataset(n_samples_val, means, covs, weights, 
                                        #  seed=config["data"]["seed_val"]
                                         )
    
    # Create DataLoaders.
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size_train"], shuffle=False,
                              num_workers=4, pin_memory=True
                              )  # Each batch: [32, 10]
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size_val"], shuffle=False,
                            num_workers=4, pin_memory=True
                              )  # Each batch: [32, 10])       # Each batch: [32, 10]
    
    # Instantiate the MLP classifier.
    print("Creating model...")
    if config["model"]["name"] == "resnet34":
        model = resnet.ResNet34(n_classes)
        model.to(device)
        # model.eval()
        # print('mode', model(torch.randn(1, 3, 32, 32).to(device)))  # Dummy forward pass to initialize model.
        # exit()
    elif config["model"]["name"] == "mlp":
        model = MLPClassifier(input_dim=dim, hidden_size=128, num_hidden_layers=config["model"]["num_hidden_layers"], 
                              dropout_p=config["training"]["dropout_p"], num_classes=n_classes)
    else:
        raise ValueError(f"Unknown model name: {config['model']['name']}")
    
    # # Define device.
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Compute Bayes classifier
    # covs = torch.diag_embed(stds ** 2)
    print("Computing Bayes classifier...")
    bayes_model = BayesClassifier(means, covs, weights)
    bayes_evaluator = Evaluator(bayes_model, val_loader, device)
    bayes_accuracy, _, _ = bayes_evaluator.evaluate()
    print(f"Bayes Val Accuracy: {bayes_accuracy:.4f}")

    bayes_evaluator = Evaluator(bayes_model, train_loader, device)
    bayes_accuracy, _, _ = bayes_evaluator.evaluate()
    print(f"Bayes Train Accuracy: {bayes_accuracy:.4f}")

    # Create and run the trainer.
    trainer = Trainer(model, train_loader, val_loader, device, checkpoint_dir, epochs=epochs, lr=lr)
    trainer.train()
    
    # Finally, evaluate the model.
    evaluator = Evaluator(model, val_loader, device)
    accuracy, preds, labels = evaluator.evaluate()
    print(f"Final Validation Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
