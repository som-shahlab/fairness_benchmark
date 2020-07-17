#!/bin/bash
#SBATCH --job-name=extract_features_mimic
#SBATCH --output=/home/spfohl/slurm_out/extract_features_mimic.out
#SBATCH --error=/home/spfohl/slurm_out/extract_features_mimic.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=24000
#SBATCH -c 5
#SBATCH -p normal,gpu,nigam

source activate prediction_utils

DATASET="mimic_omop"
RS_DATASET="temp_dataset"
COHORT_NAME="mimic_icu_cohort"
DATA_PATH="/share/pi/nigam/projects/spfohl/cohorts/admissions/mimic_omop/"
GCLOUD_STORAGE_PATH="gs://feature_extraction_exports/cohorts/admissions/mimic_omop/"

FEATURES_DATASET="temp_dataset"
GCLOUD_PROJECT="som-nero-phi-nigam-starr"
DATASET_PROJECT="som-nero-phi-nigam-starr"
RS_DATASET_PROJECT="som-nero-phi-nigam-starr"
FEATURES_PREFIX="features_"$USER
INDEX_DATE_FIELD='index_datetime'
ROW_ID_FIELD='prediction_id'
MERGED_NAME='merged_features_binary'

python -m prediction_utils.extraction_utils.extract_features \
    --data_path=$DATA_PATH \
    --features_by_analysis_path="features_by_analysis" \
    --dataset=$DATASET \
    --rs_dataset=$RS_DATASET \
    --cohort_name=$COHORT_NAME \
    --gcloud_project=$GCLOUD_PROJECT \
    --dataset_project=$DATASET_PROJECT \
    --rs_dataset_project=$RS_DATASET_PROJECT \
    --features_dataset=$FEATURES_DATASET \
    --features_prefix=$FEATURES_PREFIX \
    --index_date_field=$INDEX_DATE_FIELD \
    --row_id_field=$ROW_ID_FIELD \
    --merged_name=$MERGED_NAME \
    --time_bins_hourly="-24" "-12" "-4" "0" \
    --exclude_analysis_ids "note_nlp" "note_nlp_dt" "condition_occurrence" "condition_occurrence_dt" \
    --binary \
    --featurize \
    --no_cloud_storage \
    --merge_features \
    --create_sparse \
    --no_create_parquet \
    --overwrite