
# -------------------------------
from typing import Any, Dict
import os
import torch
from torchvision import transforms
import json 
from . import resnet
from .models import ThresholdClassifier, BayesClassifier, MLPClassifier

DATA_DIR = os.environ.get("DATA_DIR", "./data")
CHECKPOINTS_DIR_BASE = os.environ.get("CHECKPOINTS_DIR", "checkpoints/")

__all__ = ["ThresholdClassifier", "BayesClassifier", "MLPClassifier"]


def _get_default_cifar10_transforms():
    statistics = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize(*statistics),
        ]
    )
    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(*statistics),
        ]
    )
    return train_transforms, test_transforms

    
def ResNet34Cifar10(features_nodes=None):
    model = resnet.ResNet34(10)
    train_transforms, test_transforms = _get_default_cifar10_transforms()
    input_dim = (3, 32, 32)
    if features_nodes is None:
        features_nodes = {"view": "features", "linear": "linear"}
    return {
        "model": model,
        "features_nodes": features_nodes,
        "input_dim": input_dim,
        "test_transforms": test_transforms,
        "train_transforms": train_transforms,
    }



models_registry = {
    "resnet34_cifar10": ResNet34Cifar10,
}


def get_model_essentials(model, dataset, features_nodes=None) -> Dict[str, Any]:

    name = "_".join([model, dataset])
    
    if name not in models_registry:
        raise ValueError("Unknown model name: {}".format(name))
    return models_registry[name](features_nodes=features_nodes)





def get_model(model_name: str, 
              dataset_name: str, 
              n_classes: int,
              input_dim,
              model_seed, 
              checkpoint_dir,
              desired_indices=None) -> torch.nn.Module:

    if model_name == "mlp_synth_dim-10_classes-7":
       
        checkpoint_dir = os.path.join(checkpoint_dir, model_name)
        config_model_path = os.path.join(checkpoint_dir, "config.json")

        if not os.path.exists(config_model_path):
            raise FileNotFoundError(f"Configuration file not found at {config_model_path}")
        # Load the configuration file
        with open(config_model_path, "r") as f:
            config_model = json.load(f)
        
        # Instantiate the MLP classifier
        model = MLPClassifier(
            input_dim=config_model["dim"], 
            hidden_size=config_model["hidden_dims"][0], 
            num_hidden_layers=config_model["num_hidden_layers"], 
            dropout_p=config_model["dropout_p"], 
            num_classes=config_model["n_classes"]
            )
        
        # Load the model weights
        checkpoint_path = os.path.join(checkpoint_dir, "best_mlp.pth")

    
    elif (model_name == "resnet34") and (dataset_name == "gaussian_mixture"):

        checkpoint_dir = os.path.join(checkpoint_dir, 
                                      model_name + f"_synth_dim-{input_dim}_classes-{n_classes}")
        # config_model_path = os.path.join(checkpoint_dir, "config.json")

        # if not os.path.exists(config_model_path):
        #     raise FileNotFoundError(f"Configuration file not found at {config_model_path}")
        # # Load the configuration file
        # with open(config_model_path, "r") as f:
        #     config_model = json.load(f)
        
        # Instantiate the MLP classifier
        model = resnet.ResNet34(n_classes)
        
        # Load the model weights
        checkpoint_path = os.path.join(checkpoint_dir, "best_mlp.pth")


    elif (model_name == "resnet34") and (dataset_name == "cifar10"):
        model_essentials = get_model_essentials(model_name, dataset_name)
        model = model_essentials["model"]
        checkpoint_path = os.path.join(checkpoint_dir, "_".join([model_name, dataset_name]), str(model_seed), "best.pth")
        

    if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    # print("checkpoint keys", checkpoint.keys())
    # print("model keys before loading", model.state_dict().keys())
    model.load_state_dict(checkpoint)
    # missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    # print("Missing:", missing_keys)
    # print("Unexpected:", unexpected_keys)

    if desired_indices is not None:
        model = SubsetLogitWrapper(model, desired_indices)
    
    return model


class SubsetLogitWrapper(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, desired_indices: list[int]):
        """
        Wraps any classifier that outputs logits over N classes,
        and on forward() returns only the logits of `desired_indices`.
        """
        super().__init__()
        self.model = base_model
        self.indices = torch.tensor(desired_indices, dtype=torch.long)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get full logits from base model: shape [B, N]
        logits = self.model(x)
        # Select only desired class logits: shape [B, K]
        return logits[:, self.indices]