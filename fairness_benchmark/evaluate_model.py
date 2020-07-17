import pandas as pd
import os
import configargparse as argparse
import copy
from prediction_utils.pytorch_utils.metrics import (
    StandardEvaluator,
    FairOVAEvaluator,
    CalibrationEvaluator,
)

from .train_model import parser as parent_parser
from .train_model import read_file, filter_cohort

parser = argparse.ArgumentParser(
    parents=[parent_parser],
    conflict_handler="resolve",
    config_file_parser_class=argparse.YAMLConfigFileParser,
)

parser.add_argument(
    "--output_df_filename", type=str, default="output_df.parquet",
)

if __name__ == "__main__":
    args = parser.parse_args()
    config_dict = copy.deepcopy(args.__dict__)

    cohort = read_file(args.cohort_path)
    cohort = filter_cohort(cohort)

    if args.features_row_id_map_path != "":
        row_id_map = read_file(args.features_row_id_map_path, engine="pyarrow")

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

    output_df_eval = pd.read_parquet(os.path.join(result_path, args.output_df_filename))

    if args.eval_attributes is None:
        raise ValueError(
            "Must specify eval_attributes"
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
