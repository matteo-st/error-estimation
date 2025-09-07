# utils/detection/factory.py

# Make sure all your decorated classes register themselves:
import code.utils.detection.methods  
import os

from code.utils.detection.registry import DETECTOR_REGISTRY

def get_detector(config: dict,
                 model,
                 device=None,
                 experiment_folder=None,
                 checkpoints_dir=None):
    """
    Instantiate (but do not fit) the detector you asked for.

    Args:
      config: must include
        - "method_name": str
        - config[method_name]: dict of that method’s params
      model:   your PyTorch model
      device:  torch.device (optional)
      experiment_folder: path for any outputs (optional)

    Returns:
      An instance of the requested detector, NOT yet fitted.
    """
    name   = config["method_name"]
    if name not in DETECTOR_REGISTRY:
        raise ValueError(f"Unknown method {name!r}. "
                         f"Available: {list(DETECTOR_REGISTRY)}")
    
    method_cfg = config[name]
    data_cfg = config["data"]
    method_cfg["n_classes"]    = data_cfg["n_classes"]
    method_cfg["class_subset"] = data_cfg.get("class_subset")


    dim = data_cfg["dim"]
    nc  = data_cfg["n_classes"]
    method_cfg["params_path"] = os.path.join(
            checkpoints_dir,
            "ce",
            f"resnet34_synth_dim-{dim}_classes-{nc}",
            "data_parameters.npz"
        )



    DetectorCls = DETECTOR_REGISTRY[name]
    # Call your detector’s constructor; do NOT call .fit() here

    detector = DetectorCls(
        model=model,
        device=device,
        experiment_folder=experiment_folder,
        **flatten_dict(method_cfg)
    )

    return detector


def flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
    """
    Recursively flatten a nested dict.
    Keys from nested dicts get joined by `sep`.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items