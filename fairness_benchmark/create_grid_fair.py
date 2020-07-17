import numpy as np
import os
import random
import pandas as pd
import configargparse as argparse
import itertools
import glob
from .create_grid_baseline import parser as parent_parser
from prediction_utils.util import yaml_write, yaml_read
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser(
    parents=[parent_parser],
    conflict_handler="resolve",
    config_file_parser_class=argparse.YAMLConfigFileParser,
)

parser.add_argument(
    "--base_config_path", type=str, required=True, help="A base config to use",
)

if __name__ == "__main__":
    random.seed(234)
    np.random.seed(5432)

    args = parser.parse_args()

    lambda_group_regularization = [float(x) for x in np.logspace(-3, 1, num=10)]
    param_grid_lambda = {
        "lambda_group_regularization": lambda_group_regularization,
    }
    the_grid_lambda = list(ParameterGrid(param_grid_lambda))

    param_grid_method = [
        {
            "regularization_metric": ["mean_prediction"],
            "mean_prediction_mode": ["conditional", "unconditional", "conditional_pos"],
        },
        {
            "regularization_metric": ["mmd"],
            "mmd_mode": ["conditional", "unconditional", "conditional_pos"],
        },
    ]

    the_grid_method = list(ParameterGrid(param_grid_method))
    the_grid = [
        {**x, **y} for x, y in itertools.product(the_grid_lambda, the_grid_method)
    ]
    print(len(the_grid))

    for task in args.tasks:
        print(args.base_config_path)
        base_config_path_list = glob.glob(
            os.path.join(args.base_config_path, task, "*.yaml")
        )
        print(base_config_path_list)
        assert len(base_config_path_list) == 1
        base_config = yaml_read(base_config_path_list[0])
        the_grid = [{**base_config, **x} for x in the_grid]
        grid_df = pd.DataFrame(the_grid)

        config_path = os.path.join(
            args.data_path, "experiments", args.experiment_name, "config", task
        )
        os.makedirs(config_path, exist_ok=True)
        grid_df.to_csv(os.path.join(config_path, "config.csv"), index_label="id")

        for i, config_dict in enumerate(the_grid):
            yaml_write(config_dict, os.path.join(config_path, "{}.yaml".format(i)))
