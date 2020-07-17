import configargparse as argparse
import os
from prediction_utils.cohorts.admissions.cohort import BQAdmissionRollupCohort
from prediction_utils.cohorts.admissions.cohort import BQAdmissionOutcomeCohort
from prediction_utils.cohorts.admissions.cohort import BQFilterInpatientCohort
from prediction_utils.util import patient_split

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset", type=str, default="starr_omop_cdm5_deid_1pcent_lite_latest"
)
parser.add_argument("--rs_dataset", type=str, default="temp_dataset")
parser.add_argument("--limit", type=int, default=0)
parser.add_argument("--gcloud_project", type=str, default="som-nero-phi-nigam-starr")
parser.add_argument("--dataset_project", type=str, default="som-rit-phi-starr-prod")
parser.add_argument(
    "--rs_dataset_project", type=str, default="som-nero-phi-nigam-starr"
)
parser.add_argument("--cohort_name", type=str, default="admission_rollup_temp")
parser.add_argument(
    "--cohort_name_labeled", type=str, default="admission_rollup_labeled_temp"
)
parser.add_argument(
    "--cohort_name_filtered", type=str, default="admission_rollup_filtered_temp"
)
parser.add_argument(
    "--has_birth_datetime", dest="has_birth_datetime", action="store_true"
)
parser.add_argument(
    "--no_has_birth_datetime", dest="has_birth_datetime", action="store_false"
)
parser.add_argument(
    "--data_path",
    type=str,
    default="/share/pi/nigam/projects/spfohl/cohorts/admissions/scratch",
)
parser.add_argument(
    "--google_application_credentials",
    type=str,
    default=os.path.expanduser("~/.config/gcloud/application_default_credentials.json"),
)
parser.set_defaults(has_birth_datetime=True)

if __name__ == "__main__":
    args = parser.parse_args()
    cohort = BQAdmissionRollupCohort(**args.__dict__)
    print(cohort.get_create_query())
    cohort.create_cohort_table()

    cohort_labeled = BQAdmissionOutcomeCohort(**args.__dict__)
    print(cohort_labeled.get_create_query())
    cohort_labeled.create_cohort_table()

    cohort_filtered = BQFilterInpatientCohort(**args.__dict__)
    cohort_filtered.create_cohort_table()
    cohort_df = cohort_filtered.db.read_sql_query(
        """
            SELECT *
            FROM {rs_dataset_project}.{rs_dataset}.{cohort_name_filtered}
        """.format(
            **args.__dict__
        ),
        use_bqstorage_api=True,
    )
    cohort_df = patient_split(
        cohort_df, patient_col="person_id", test_frac=0.1, nfold=10, seed=657
    )
    cohort_path = os.path.join(args.data_path, "cohort")
    os.makedirs(cohort_path, exist_ok=True)
    cohort_df.to_parquet(
        os.path.join(cohort_path, "cohort.parquet"), engine="pyarrow", index=False,
    )
