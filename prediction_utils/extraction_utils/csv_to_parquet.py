import os
import glob
from pyarrow import csv

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--base_path",
    type=str,
    default="/share/pi/nigam/projects/spfohl/cohorts/admissions/optum_bq_1pcnt",
)

parser.add_argument(
    "--features_by_analysis_path", type=str, default="features_by_analysis"
)

parser.add_argument(
    "--parquet_path", type=str, default="features_by_analysis_parquet_indexed"
)

parser.add_argument("--row_id_field", type=str, default="prediction_id")

if __name__ == "__main__":
    args = parser.parse_args()
    # the path where the csv files are stored
    features_by_analysis_base_path = os.path.join(
        args.base_path, args.features_by_analysis_path
    )
    # the path where the parquets will be stored
    parquet_base_path = os.path.join(args.base_path, args.parquet_path)
    # a list of csv files
    csv_files = glob.glob(
        os.path.join(features_by_analysis_base_path, "**", "*.csv"), recursive=True
    )
    assert len(csv_files) > 0

    for i, csv_file in enumerate(csv_files):
        # Mirror the directory structure used for the csv files
        _, csv_path_suffix = csv_file.split(
            "/{}/".format(args.features_by_analysis_path)
        )
        parquet_path_suffix = "{}.parquet".format(os.path.splitext(csv_path_suffix)[0])
        parquet_path = os.path.join(parquet_base_path, parquet_path_suffix)
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        table = csv.read_csv(csv_file).to_pandas()
        table = table.sort_values(args.row_id_field)
        table.to_parquet(parquet_path)
