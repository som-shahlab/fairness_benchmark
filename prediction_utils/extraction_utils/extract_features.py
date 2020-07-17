import configargparse as argparse
import os
from prediction_utils.extraction_utils.featurizer import BigQueryOMOPFeaturizer

parser = argparse.ArgumentParser(description="An extraction script")
parser.add_argument(
    "--data_path",
    type=str,
    default="/share/pi/nigam/projects/spfohl/cohorts/admissions/starr_20200404",
)

parser.add_argument(
    "--features_by_analysis_path", type=str, default="features_by_analysis"
)

parser.add_argument(
    "--gcloud_storage_path",
    type=str,
    default="gs://feature_extraction_exports/cohorts/scratch/",
)

parser.add_argument("--gcloud_project", type=str, default="som-nero-phi-nigam-starr")
parser.add_argument("--dataset_project", type=str, default="")
parser.add_argument("--rs_dataset_project", type=str, default="")

parser.add_argument("--dataset", type=str, default="starr_omop_cdm5_deid_20200404")
parser.add_argument("--rs_dataset", type=str, default="plp_cohort_tables")
parser.add_argument("--features_dataset", type=str, default="temp_dataset")
parser.add_argument("--features_prefix", type=str, default="features")

parser.add_argument(
    "--cohort_name", type=str, default="admission_rollup_20200404_with_labels_sampled"
)

parser.add_argument("--index_date_field", type=str, default="admit_date")
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--row_id_field", type=str, default="prediction_id")

parser.add_argument(
    "--google_application_credentials",
    type=str,
    default=os.path.expanduser("~/.config/gcloud/application_default_credentials.json"),
)

parser.add_argument("--dask_temp_dir", type=str, default=None)

parser.add_argument("--time_bins", type=int, default=None, nargs="*")

parser.add_argument("--time_bins_hourly", type=int, default=None, nargs="*")

parser.add_argument("--analysis_ids", type=str, default=None, nargs="*")

parser.add_argument("--exclude_analysis_ids", type=str, default=None, nargs="*")

parser.add_argument("--merged_name", type=str, default="merged_features")

parser.add_argument("--binary", dest="binary", action="store_true")

parser.add_argument(
    "--featurize",
    dest="featurize",
    action="store_true",
    help="Whether to run the featurization",
)
parser.add_argument(
    "--no_featurize",
    dest="featurize",
    action="store_false",
    help="Whether to run the featurization",
)

parser.add_argument(
    "--cloud_storage",
    dest="cloud_storage",
    action="store_true",
    help="Whether to write the results to cloud storage",
)
parser.add_argument(
    "--no_cloud_storage",
    dest="cloud_storage",
    action="store_false",
    help="Whether to write the results to cloud storage",
)

parser.add_argument(
    "--merge_features",
    dest="merge_features",
    action="store_true",
    help="Whether to merge the features",
)

parser.add_argument(
    "--no_merge_features",
    dest="merge_features",
    action="store_false",
    help="Whether to merge the features",
)

parser.add_argument(
    "--create_parquet",
    dest="create_parquet",
    action="store_true",
    help="Whether to create parquet on merge",
)
parser.add_argument(
    "--no_create_parquet",
    dest="create_parquet",
    action="store_false",
    help="Whether to create parquet on merge",
)

parser.add_argument(
    "--create_sparse",
    dest="create_sparse",
    action="store_true",
    help="Whether to create sparse array on merge",
)
parser.add_argument(
    "--no_create_sparse",
    dest="create_sparse",
    action="store_false",
    help="Whether to create sparse array on merge",
)

parser.add_argument(
    "--overwrite",
    dest="overwrite",
    action="store_true",
    help="Whether to overwrite results",
)

parser.set_defaults(
    merge_features=False,
    featurize=True,
    create_parquet=True,
    create_sparse=True,
    create_h5=False,
    binary=False,
    cloud_storage=False,
    overwrite=False,
)

if __name__ == "__main__":

    args = parser.parse_args()

    featurizer = BigQueryOMOPFeaturizer(
        include_all_history=True,
        **args.__dict__
    )

    if args.featurize:
        if args.cloud_storage:
            featurizer.featurize_to_destination(
                analysis_ids=args.analysis_ids,
                exclude_analysis_ids=args.exclude_analysis_ids,
                merge_features=args.merge_features,
            )
        else:
            featurizer.featurize(
                analysis_ids=args.analysis_ids,
                exclude_analysis_ids=args.exclude_analysis_ids,
            )

    if args.merge_features:

        featurizer.merge_features(
            merged_name=args.merged_name,
            create_sparse=args.create_sparse,
            create_parquet=args.create_parquet,
            binary=args.binary,
            load_extension="parquet",
            dask_temp_dir=args.dask_temp_dir,
        )
