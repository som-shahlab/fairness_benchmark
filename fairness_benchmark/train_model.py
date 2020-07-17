import pandas as pd
import os
import joblib
import configargparse as argparse
import copy
from prediction_utils.pytorch_utils.models import FixedWidthModel
from prediction_utils.pytorch_utils.datasets import (
    ArrayLoaderGenerator,
    ParquetLoaderGenerator,
)
from prediction_utils.util import yaml_write
from prediction_utils.pytorch_utils.group_fairness import group_regularized_model
from prediction_utils.pytorch_utils.metrics import (
    StandardEvaluator,
    FairOVAEvaluator,
    CalibrationEvaluator,
)

parser = argparse.ArgumentParser(config_file_parser_class=argparse.YAMLConfigFileParser)

parser.add_argument("--config_path", required=False, is_config_file=True)

# Path configuration
parser.add_argument(
    "--data_path",
    type=str,
    default="/labs/shahlab/projects/spfohl/cohorts/admissions/starr",
    help="The root path where data is stored",
)

parser.add_argument(
    "--features_path",
    type=str,
    default="/labs/shahlab/projects/spfohl/cohorts/admissions/starr/merged_features/features/features.gz",
    help="The root path where data is stored",
)

parser.add_argument(
    "--cohort_path",
    type=str,
    default="/labs/shahlab/projects/spfohl/cohorts/admissions/starr/cohort/cohort.csv",
    help="File name for the file containing label information",
)

parser.add_argument(
    "--vocab_path",
    type=str,
    default="/labs/shahlab/projects/spfohl/cohorts/admissions/starr/merged_features/covariates_csv/covariateRef.csv",
    help="File name for the file containing label information",
)

parser.add_argument(
    "--features_row_id_map_path",
    type=str,
    default="/labs/shahlab/projects/spfohl/cohorts/admissions/starr/merged_features/features/features_row_id_map.csv",
)

parser.add_argument(
    "--hdf5_path",
    type=str,
    default="/labs/shahlab/projects/spfohl/cohorts/admissions/starr/merged_features/covariates_h5/covariates.h5",
)

parser.add_argument(
    "--parquet_path",
    type=str,
    default="/labs/shahlab/projects/spfohl/cohorts/admissions/starr/merged_features/covariates_parquet",
)


# Model Hyperparameters
parser.add_argument(
    "--num_epochs", type=int, default=10, help="The number of epochs of training"
)
parser.add_argument(
    "--iters_per_epoch",
    type=int,
    default=100,
    help="The number of batches to run per epoch",
)

parser.add_argument("--batch_size", type=int, default=256, help="The batch size")

parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate")

parser.add_argument("--gamma", type=float, default=0.95, help="Learning rate decay")

parser.add_argument(
    "--num_hidden", type=int, default=3, help="The number of hidden layers"
)

parser.add_argument(
    "--hidden_dim", type=int, default=128, help="The dimension of the hidden layers"
)

parser.add_argument(
    "--normalize", dest="normalize", action="store_true", help="Use layer normalization"
)

parser.add_argument(
    "--drop_prob", type=float, default=0.75, help="The dropout probability"
)

parser.add_argument(
    "--early_stopping",
    dest="early_stopping",
    action="store_true",
    help="Whether to use early stopping",
)

parser.add_argument("--early_stopping_patience", type=int, default=5)

parser.add_argument(
    "--selection_metric",
    type=str,
    default="loss",
    help="The metric to use for model selection",
)

parser.add_argument("--fold_id", type=str, default="1", help="The fold id")

parser.add_argument(
    "--experiment_name", type=str, default="scratch", help="The name of the experiment"
)

parser.add_argument("--label_col", type=str, default="LOS_7", help="The label to use")

parser.add_argument(
    "--data_mode", type=str, default="array", help="Which mode of source data to use"
)
parser.add_argument("--sparse_mode", type=str, default="csr", help="the sparse mode")
parser.add_argument(
    "--num_workers",
    type=int,
    default=5,
    help="The number of workers to use for data loading during training in parquet mode",
)

parser.add_argument(
    "--save_outputs",
    dest="save_outputs",
    action="store_true",
    help="Whether to save the outputs of evaluation",
)

parser.add_argument(
    "--run_evaluation",
    dest="run_evaluation",
    action="store_true",
    help="Whether to evaluate the model",
)

parser.add_argument(
    "--no_run_evaluation",
    dest="run_evaluation",
    action="store_false",
    help="Whether to evaluate the model",
)

parser.add_argument(
    "--run_evaluation_group",
    dest="run_evaluation",
    action="store_true",
    help="Whether to evaluate the model for each group",
)

parser.add_argument(
    "--no_run_evaluation_group",
    dest="run_evaluation_group",
    action="store_false",
    help="Whether to evaluate the model for each group",
)

parser.add_argument(
    "--run_evaluation_group_standard",
    dest="run_evaluation_group_standard",
    action="store_true",
    help="Whether to evaluate the model",
)
parser.add_argument(
    "--no_run_evaluation_group_standard",
    dest="run_evaluation_group_standard",
    action="store_false",
    help="Whether to evaluate the model",
)

parser.add_argument(
    "--run_evaluation_group_fair_ova",
    dest="run_evaluation_group_fair_ova",
    action="store_true",
    help="Whether to evaluate the model",
)
parser.add_argument(
    "--no_run_evaluation_group_fair_ova",
    dest="run_evaluation_group_fair_ova",
    action="store_false",
    help="Whether to evaluate the model",
)

parser.add_argument(
    "--run_evaluation_group_calibration",
    dest="run_evaluation_group_calibration",
    action="store_true",
    help="Whether to evaluate the model",
)
parser.add_argument(
    "--no_run_evaluation_group_calibration",
    dest="run_evaluation_group_calibration",
    action="store_false",
    help="Whether to evaluate the model",
)

parser.add_argument(
    "--eval_attributes", type=str, nargs="+", required=False, default=None
)

parser.add_argument("--sample_keys", type=str, nargs="*", required=False, default=None)

parser.add_argument(
    "--replicate_id", type=str, default="", help="Optional replicate id"
)

## Arguments for fair models
parser.add_argument(
    "--sensitive_attribute",
    type=str,
    default=None,
    help="The attribute to be fair with respect to",
)

parser.add_argument(
    "--regularization_metric",
    type=str,
    default="loss",
    help="The metric to use for fairness regularization",
)

parser.add_argument(
    "--lambda_group_regularization",
    type=float,
    default=1e-1,
    help="The extent to which to penalize group differences in the regularization_metric",
)

parser.add_argument(
    "--temperature",
    type=float,
    default=1.0,
    help="The sigmoid temperature for indicator function approximations in surrogate losses",
)

parser.add_argument(
    "--mmd_mode",
    type=str,
    default="unconditional",
    help="""
    Options: `conditional`, `unconditional`. Conditional corresponds to equalized odds, unconditional corresponds to demographic parity.
    """,
)

parser.add_argument(
    "--mean_prediction_mode",
    type=str,
    default="unconditional",
    help="""
    Options: `conditional`, `unconditional`. Conditional corresponds to equalized odds, unconditional corresponds to demographic parity.
    """,
)

parser.add_argument(
    "--group_regularization_mode",
    type=str,
    default="overall",
    help="The type of group regularization used. Valid options are `overall` and `group`",
)


parser.set_defaults(
    normalize=False,
    early_stopping=False,
    run_evaluation=True,
    save_outputs=True,
    run_evaluation_group=True,
    run_evaluation_group_standard=True,
    run_evaluation_group_fair_ova=False,
)


def filter_cohort(cohort):
    cohort = cohort.query('gender_concept_name != "No matching concept"')
    return cohort


def get_loader_generator_class(data_mode="parquet"):
    if data_mode == "parquet":
        return ParquetLoaderGenerator
    elif data_mode == "array":
        return ArrayLoaderGenerator


def read_file(filename, columns=None, **kwargs):
    print(filename)
    load_extension = os.path.splitext(filename)[-1]
    if load_extension == ".parquet":
        return pd.read_parquet(filename, columns=columns, **kwargs)
    elif load_extension == ".csv":
        return pd.read_csv(filename, usecols=columns, **kwargs)


if __name__ == "__main__":
    args = parser.parse_args()
    config_dict = copy.deepcopy(args.__dict__)

    if args.fold_id == "":
        train_keys = ["train"]
        eval_keys = ["test"]
    else:
        train_keys = ["train", "val"]
        eval_keys = ["val", "test"]

    vocab = read_file(args.vocab_path, engine="pyarrow")
    config_dict["input_dim"] = vocab.col_id.max() + 1

    cohort = read_file(args.cohort_path)
    cohort = filter_cohort(cohort)

    if args.data_mode == "array":
        features = joblib.load(args.features_path)
        if args.features_row_id_map_path != "":
            row_id_map = read_file(args.features_row_id_map_path, engine="pyarrow")
            cohort = cohort.merge(row_id_map)
            config_dict["row_id_col"] = "features_row_id"
    else:
        features = None

    if config_dict.get("config_path") is None:
        result_path_suffix = ""
    else:
        result_path_suffix = os.path.basename(config_dict["config_path"])

    result_path = os.path.join(
        args.data_path,
        "experiments",
        args.experiment_name,
        "performance",
        args.label_col,
        str(args.sensitive_attribute) if args.sensitive_attribute is not None else "",
        result_path_suffix,
        str(config_dict["fold_id"]),
        str(args.replicate_id),
    )
    print("Result path: {}".format(result_path))
    os.makedirs(result_path, exist_ok=True)

    if args.sensitive_attribute is None:
        loader_generator = get_loader_generator_class(data_mode=args.data_mode)(
            features=features, cohort=cohort, **config_dict
        )
        model = FixedWidthModel(**config_dict)
    else:
        loader_generator = get_loader_generator_class(data_mode=args.data_mode)(
            features=features,
            cohort=cohort,
            include_group_in_dataset=True,
            **config_dict
        )
        model_class = group_regularized_model(config_dict["regularization_metric"])
        model = model_class(**config_dict)
    print(model.config_dict)

    # Write the resulting config
    yaml_write(config_dict, os.path.join(result_path, "config.yaml"))

    loaders = loader_generator.init_loaders(sample_keys=args.sample_keys)

    result_df = model.train(loaders, phases=train_keys)["performance"]
    del loaders

    # Dump training results to disk
    result_df.to_parquet(
        os.path.join(result_path, "result_df_training.parquet"),
        index=False,
        engine="pyarrow",
    )

    if args.run_evaluation:
        print("Evaluating model")
        loaders_predict = loader_generator.init_loaders_predict()
        predict_dict = model.predict(loaders_predict, phases=eval_keys)
        del loaders_predict
        output_df_eval, result_df_eval = (
            predict_dict["outputs"],
            predict_dict["performance"],
        )
        print(result_df_eval)

        # Dump evaluation result to disk
        result_df_eval.to_parquet(
            os.path.join(result_path, "result_df_training_eval.parquet"),
            index=False,
            engine="pyarrow",
        )
        if args.save_outputs:
            output_df_eval.to_parquet(
                os.path.join(result_path, "output_df.parquet"),
                index=False,
                engine="pyarrow",
            )
        if args.run_evaluation_group:
            if args.eval_attributes is None:
                raise ValueError(
                    "If using run_evaluation_group, must specify eval_attributes"
                )
            group_vars = ["phase", "task", "sensitive_attribute", "attribute"]
            output_df_eval = output_df_eval.assign(task=args.label_col)
            output_df_eval = output_df_eval.merge(
                row_id_map, left_on="row_id", right_on="features_row_id"
            ).merge(cohort)
            output_df_long = output_df_eval.melt(
                id_vars=set(output_df_eval.columns) - set(args.eval_attributes),
                value_vars=args.eval_attributes,
                var_name="attribute",
                value_name="group",
            )
            if args.run_evaluation_group_standard:
                evaluator = StandardEvaluator()
                result_df_group_standard_eval = evaluator.get_result_df(
                    output_df_long, group_vars=group_vars,
                )
                print(result_df_group_standard_eval)
                result_df_group_standard_eval.to_parquet(
                    os.path.join(result_path, "result_df_group_standard_eval.parquet"),
                    engine="pyarrow",
                    index=False,
                )
            if args.run_evaluation_group_fair_ova:
                evaluator = FairOVAEvaluator()
                result_df_group_fair_ova = evaluator.get_result_df(
                    output_df_long, group_vars=group_vars
                )
                print(result_df_group_fair_ova)
                result_df_group_fair_ova.to_parquet(
                    os.path.join(result_path, "result_df_group_fair_ova.parquet"),
                    engine="pyarrow",
                    index=False,
                )

            if args.run_evaluation_group_calibration:
                evaluator = CalibrationEvaluator()
                calibration_df = evaluator.get_calibration_df_combined(
                    output_df_long, group_vars=group_vars
                )
                print(calibration_df)
                calibration_result = evaluator.get_calibration_result(
                    calibration_df, group_vars=group_vars + ["group"]
                )
                print(calibration_result)
                calibration_df.to_parquet(
                    os.path.join(result_path, "calibration_df.parquet"),
                    engine="pyarrow",
                    index=False,
                )
                calibration_result.to_parquet(
                    os.path.join(result_path, "calibration_result.parquet"),
                    engine="pyarrow",
                    index=False,
                )
