#!/bin/bash
#SBATCH --job-name=extract_features_storage_starr
#SBATCH --output=/home/spfohl/slurm_out/extract_features_storage_starr.out
#SBATCH --error=/home/spfohl/slurm_out/extract_features_storage_starr.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=25000
#SBATCH -c 20
#SBATCH -p normal,gpu

source activate prediction_utils

DATASET="starr_omop_cdm5_deid_latest"
RS_DATASET="plp_cohort_tables"
COHORT_NAME="admission_rollup_filtered_starr_20200425"
DATA_PATH="/share/pi/nigam/projects/spfohl/cohorts/admissions/starr_20200425"

FEATURES_DATASET="temp_dataset"
GCLOUD_PROJECT="som-nero-phi-nigam-starr"
DATASET_PROJECT="som-rit-phi-starr-prod"
RS_DATASET_PROJECT="som-nero-phi-nigam-starr"
FEATURES_PREFIX="features_"$USER
INDEX_DATE_FIELD='admit_date'
ROW_ID_FIELD='prediction_id'
MERGED_NAME='merged_features_binary'

FEATURES_BY_ANALYSIS_PATH="features_by_analysis"
FEATURES_BY_ANALYSIS_PARQUET_PATH='features_by_analysis_parquet'

GCLOUD_STORAGE_PATH="gs://feature_extraction_exports/cohorts/admissions/starr_20200425"
SOURCE_PATH=$GCLOUD_STORAGE_PATH'/'$FEATURES_BY_ANALYSIS_PATH
TARGET_PATH=$DATA_PATH'/'$FEATURES_BY_ANALYSIS_PATH

python -m prediction_utils.extraction_utils.extract_features \
    --data_path=$DATA_PATH \
    --features_by_analysis_path=$FEATURES_BY_ANALYSIS_PATH \
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
    --time_bins="-365" "-180" "-90" "-30" "0" \
    --time_bins_hourly="-168" "-72", "-24", "-12", "-4", "0" \
    --binary \
    --featurize \
    --no_cloud_storage \
    --no_merge_features \
    --create_sparse \
    --no_create_parquet \
    --cloud_storage \
    --gcloud_storage_path=$GCLOUD_STORAGE_PATH \
    --overwrite \
&& mkdir -p $TARGET_PATH \
&& module load google \
&& module load anaconda/2 \
&& gsutil -m rsync -r $SOURCE_PATH $TARGET_PATH \
&& module unload anaconda/2 \
&& python -m prediction_utils.extraction_utils.csv_to_parquet \
    --base_path=$DATA_PATH \
    --features_by_analysis_path=$FEATURES_BY_ANALYSIS_PATH \
    --parquet_path=$FEATURES_BY_ANALYSIS_PARQUET_PATH \
&& python -m prediction_utils.extraction_utils.extract_features \
    --data_path=$DATA_PATH \
    --no_featurize \
    --merge_features \
    --binary \
    --features_by_analysis_path=$FEATURES_BY_ANALYSIS_PARQUET_PATH \
    --merged_name=$MERGED_NAME \
    --create_sparse \
    --create_parquet