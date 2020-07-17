import numpy as np
import os
import random
import pandas as pd
import configargparse as argparse
import itertools

from prediction_utils.util import yaml_write
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser(
    config_file_parser_class=argparse.YAMLConfigFileParser,
)

parser.add_argument(
    "--data_path",
    type=str,
    default="/share/pi/nigam/projects/spfohl/cohorts/admissions/scratch/",
    help="The root data path",
)

parser.add_argument(
    "--experiment_name", type=str, default="scratch", help="The name of the experiment",
)

parser.add_argument(
    "--grid_size",
    type=int,
    default=50,
    help="The number of elements in the random grid",
)

parser.add_argument("--seed", type=int, default=234)
parser.add_argument("--tasks", type=str, nargs="+", required=True)

if __name__ == "__main__":

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Generate a grid of hyperparameters
    param_grid_base = {
        "lr": [1e-3, 1e-4, 1e-5],
        "batch_size": [128, 256, 512],
        "drop_prob": [0.0, 0.25, 0.5, 0.75],
        "num_epochs": [150],
        "num_hidden": [1, 2, 3],
        "hidden_dim": [128, 256],
        "gamma": [1.0],
        "early_stopping": [True],
        "early_stopping_patience": [10],
    }
    the_grid = list(ParameterGrid(param_grid_base))
    np.random.shuffle(the_grid)
    the_grid = the_grid[: args.grid_size]
    print(args.grid_size)
    for task in args.tasks:
        # the_grid = [{**args.__dict__, **x} for x in the_grid]
        grid_df = pd.DataFrame(the_grid)
        config_path = os.path.join(
            args.data_path, "experiments", args.experiment_name, "config", task
        )
        os.makedirs(config_path, exist_ok=True)
        grid_df.to_csv(os.path.join(config_path, "config.csv"), index_label="id")

        for i, config_dict in enumerate(the_grid):
            config_dict["label_col"] = task
            yaml_write(config_dict, os.path.join(config_path, "{}.yaml".format(i)))
