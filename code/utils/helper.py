from itertools import product
from code.utils import set_nested
from copy import deepcopy
from time import strftime, localtime
import os
import pandas as pd

def append_results_to_file(config, train_results, val_results, result_file):

    config = _prepare_config_for_results(config)
    config = pd.json_normalize(config, sep="_")
    results = pd.concat([config, train_results, val_results], axis=1)
    # print(results)
    print(f"Saving results to {result_file}")
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    # df_pa_table = pa.Table.from_pandas(results)
    if not os.path.isfile(result_file):
        results.to_csv(result_file, header=True, index=False)
    else:  # it exists, so append without writing the header
        results.to_csv(result_file, mode="a", header=False, index=False)


def make_config_list(base_config: dict, parameter_space: dict | None) -> list[dict]:
    """
    Expand a dict of parameter lists into a list of full configs.
    If parameter_space is empty/None, return a single config (base_config).
    """
    if not parameter_space:                     # covers {}, None
        return [deepcopy(base_config)]

    keys, values = zip(*parameter_space.items()) 
    grid = [dict(zip(keys, combo)) for combo in product(*values)]
    list_configs = []
    for params in grid:
        config = deepcopy(base_config)
        for path, val in params.items():
            set_nested(config, path, val) 
        list_configs.append(config)
    return list_configs



def _prepare_config_for_results(config, experiment_nb=None):

    def noneify(d):
        """
        Return a new dict with the same keys (and nested dict‐structure),
        but with every non‐dict value replaced by None.
        """
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out[k] = noneify(v)
            else:
                out[k] = None
        return out

    timestamp = strftime("%Y-%m-%d_%H-%M-%S", localtime())
    config["experiment"] = {}
    config["experiment"]["datetime"] = timestamp
    if experiment_nb is not None:
        config["experiment"]["folder"] = f"experiment_{experiment_nb}"
    else:
        config["experiment"]["folder"] = "bwe"

    list_methods = ["gini", "metric_learning", "clustering", "bayes", "max_proba","knn", "logistic", "random_forest"]
    method_name = config.get("method_name")

    if method_name not in list_methods:
        raise ValueError(f"Unknown method '{method_name}'")

    for m in list_methods:
        if m == method_name:
            continue

        subconf = config.get(m)
        if isinstance(subconf, dict):
            # reset all its keys to None
            config[m] = noneify(subconf)
        else:
            # nothing to reset (either missing or not a dict)
            # Optionally, you could initialize it:
            # config[m] = {}
            pass
    if config["clustering"]["reduction"]["name"] is None:
        config["clustering"]["reduction"]= dict.fromkeys(config["clustering"]["reduction"].keys(), None)

    return config
