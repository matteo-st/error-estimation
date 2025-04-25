
# -------------------------------
from typing import Any, Dict

from torchvision import transforms

from . import resnet



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
    model_name = "_".join([model, dataset])
    if model_name not in models_registry:
        raise ValueError("Unknown model name: {}".format(model_name))
    return models_registry[model_name](features_nodes=features_nodes)
