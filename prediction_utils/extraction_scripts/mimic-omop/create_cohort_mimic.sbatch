#!/bin/bash
#SBATCH --job-name=create_cohort_mimic
#SBATCH --output=/home/spfohl/slurm_out/create_cohort_mimic.out
#SBATCH --error=/home/spfohl/slurm_out/create_cohort_mimic.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=20000
#SBATCH -c 5
#SBATCH -p normal,gpu

. activate /share/pi/nigam/envs/anaconda/envs/prediction_utils

DATASET="mimic_omop"
RS_DATASET="temp_dataset"

GCLOUD_PROJECT="som-nero-phi-nigam-starr"
DATASET_PROJECT="som-nero-phi-nigam-starr"
RS_DATASET_PROJECT="som-nero-phi-nigam-starr"

COHORT_NAME="mimic_icu_cohort"
DATA_PATH="/share/pi/nigam/projects/spfohl/cohorts/admissions/mimic_omop/"

/share/pi/nigam/envs/anaconda/envs/prediction_utils/bin/python -m prediction_utils.cohorts.mimic_omop.create_cohort \
    --dataset=$DATASET \
    --rs_dataset=$RS_DATASET \
    --cohort_name=$COHORT_NAME \
    --gcloud_project=$GCLOUD_PROJECT \
    --dataset_project=$DATASET_PROJECT \
    --rs_dataset_project=$RS_DATASET_PROJECT \
    --data_path=$DATA_PATH \
    --has_birth_datetime