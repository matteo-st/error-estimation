import os
import json
import subprocess
import argparse
import numpy as np
import itertools

def main():
    parser = argparse.ArgumentParser(
        description="Run experiments with different n_cluster values"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/Conf/config.json",
        help="Path to the config.json file"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=10,
        help="Start value for n_cluster"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=200,
        help="End value (exclusive) for n_cluster"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="Step size for n_cluster"
    )
    args = parser.parse_args()
    n_cluster = None

    # [1., 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8.5]
    # for n_cluster in range(args.start, args.end, args.step):
    temperature_range = np.linspace(0.2, 2, 40)
    # TEMPERATURES = [0.2, 0.4, 0.6, 0.8, 1.1, 1.2, 1.3, 1.4, 1.5, 2]
    TEMPERATURES = range(100, 1000, 200)
  
    # MAGNITUDES = [
    #         0,
    #         0.0005,
    #         0.001,
    #         0.0015,
    #         0.002,
    #         0.0025,
    #         0.003,
    #         0.0035,
    #         0.0040,
    #         0.0036,
    #         0.0038,
    #         0.004,
    #         0.0042,
    #         0.0044,
    #     ]
    CLUSTERS = range(100, 200, 20)
    # CLUSTERS = [190]
    GRAD_LR = [0.05, 0.01, 0.005, 0.0001]
    F_DIVERGENCES = ["js", "chi2", "reverse_kl", "hellinger", "tv"]

    # for grad_lr, f_divergence, temperature, n_cluster  in itertools.product(GRAD_LR, F_DIVERGENCES, TEMPERATURES, CLUSTERS):
    for kmeans_seed in range(1, 11): 
        # Load the original configuration.
        with open(args.config, "r") as f:
            config = json.load(f)
        
        # Update the configuration parameters.
        config["kmeans_seed"] = kmeans_seed
        # config["temperature"] = temperature
        # config["magnitude"] = magnitude
        # config["n_cluster"] = n_cluster
        # config["grad_lr"] = grad_lr
        # config["f_divergence"] = f_divergence

        # Write the updated configuration back to the file.
        with open(args.config, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"Running experiment with kmeans_seed = {kmeans_seed}")
        
        # Run the experiment.
        subprocess.run(["python", "-m", "src.Conf.main"], check=True)

if __name__ == "__main__":
    main()
