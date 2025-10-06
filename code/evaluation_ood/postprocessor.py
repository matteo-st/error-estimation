import os
import urllib.request

from code.postprocessors import (
    BasePostprocessor,
    ODINPostprocessor,
    HistoPostprocessor,
  
)
from code.utils.config import Config, merge_configs

postprocessors = {
    "msp": BasePostprocessor,
    "odin": ODINPostprocessor,
    "histo": HistoPostprocessor,
}

link_prefix = (
    "https://raw.githubusercontent.com/Jingkang50/OpenOOD/main/configs/postprocessors/"
)


def get_postprocessor(config_root: str, postprocessor_name: str, id_data_name: str):

    if postprocessor_name == "nac":
        postprocessor_config_path = os.path.join(
            config_root,
            f"postprocessors/nac/resnet/{postprocessor_name}_{id_data_name}.yml",
        )
        # postprocessor_config_path = os.path.join(config_root, f'{postprocessor_name}_{id_data_name}.yml')

    else:
        postprocessor_config_path = os.path.join(
            config_root, "postprocessors", f"{postprocessor_name}.yml"
        )

    if not os.path.exists(postprocessor_config_path):
        os.makedirs(os.path.dirname(postprocessor_config_path), exist_ok=True)
        urllib.request.urlretrieve(
            link_prefix + f"{postprocessor_name}.yml", postprocessor_config_path
        )

    config = Config(postprocessor_config_path)
    config = merge_configs(config, Config(**{"dataset": {"name": id_data_name}}))
   
    postprocessor = postprocessors[postprocessor_name](config)
    postprocessor.APS_mode = config.postprocessor.APS_mode
    postprocessor.hyperparam_search_done = False
    return postprocessor
