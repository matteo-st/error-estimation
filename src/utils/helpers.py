import json
import os
from typing import Any, Dict
import shutil


import pandas as pd


def str_to_dict(string: str) -> Dict[str, Any]:
    return json.loads(string)


def append_results_to_file(results, filename="results.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if isinstance(results, dict):
        results = {k: [v] for k, v in results.items()}
        results = pd.DataFrame.from_dict(results, orient="columns")
    print(f"Saving results to {filename}")
    # df_pa_table = pa.Table.from_pandas(results)
    if not os.path.isfile(filename):
        results.to_csv(filename, header=True, index=False)
    else:  # it exists, so append without writing the header
        results.to_csv(filename, mode="a", header=False, index=False)


import datetime

def create_experiment_folder(config_path="config.json", results_dir=None):
    """
    Crée un nouveau dossier d'expérience dans le répertoire des résultats spécifié.
    Le nouveau dossier est nommé "experiment_x" où x est un incrément du plus grand numéro existant.
    Le fichier de configuration est modifié pour y ajouter la date et l'heure de l'expérience
    et est ensuite copié dans ce dossier pour assurer la reproductibilité.
    
    Args:
        config_path (str): Chemin vers le fichier de configuration JSON.
        results_dir (str): Chemin vers le répertoire des résultats. Si None, utilise la variable d'environnement 
                           RESULTS_DIR ou "results/" par défaut.
    
    Returns:
        str: Le chemin vers le dossier d'expérience nouvellement créé.
    """
    # Détermination du répertoire des résultats
    if results_dir is None:
        results_dir = os.environ.get("RESULTS_DIR", "results/")
    results_dir = os.path.join(results_dir, "experiments")
    os.makedirs(results_dir, exist_ok=True)
    
    # Liste des dossiers d'expériences existants
    existing = [
        f for f in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, f)) and f.startswith("experiment_")
    ]
    # Extraction des numéros d'expérience
    numbers = []
    for folder in existing:
        try:
            number = int(folder.split("_")[1])
            numbers.append(number)
        except (IndexError, ValueError):
            continue
    new_number = max(numbers) + 1 if numbers else 1

    # Création du nouveau dossier d'expérience
    experiment_folder = os.path.join(results_dir, f"experiment_{new_number}")
    os.makedirs(experiment_folder, exist_ok=True)
    
    # Chargement du fichier de configuration existant
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    # Ajout de la date et de l'heure de l'expérience au fichier de configuration
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    config_data["experiment_datetime"] = timestamp
    config_data["var_cross_val_computation"] =  "v2"
    
    # Sauvegarde du fichier de configuration modifié dans le dossier de l'expérience
    new_config_path = os.path.join(experiment_folder, "config.json")
    with open(new_config_path, "w") as f:
        json.dump(config_data, f, indent=4)
    
    print(f"Experiment folder created: {experiment_folder}")
    print(f"Updated config file saved to: {new_config_path}")
    print(10 * "---")
    
    return experiment_folder