#!/bin/bash 
#SBATCH --job-name=evaluate_fair_model_optum
#SBATCH --output=/home/spfohl/slurm_out/evaluate_fair_model_optum_%a.out
#SBATCH --error=/home/spfohl/slurm_out/evaluate_fair_model_optum_%a.err
#SBATCH --time=6-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH -c 5
#SBATCH --array=0-59
#SBATCH -p normal,gpu

source activate /share/pi/nigam/envs/anaconda/envs/prediction_utils

EXPERIMENT_NAME='fair_tuning_fold_1'
BASE_PATH='/share/pi/nigam/projects/spfohl/cohorts/admissions/optum'
TASKS='LOS_7 readmission_30'
ATTRIBUTES='gender_concept_name age_group'

train_model_func() {
    python -m prediction_utils.experiments.fairness_benchmark.evaluate_model \
        --data_path=$BASE_PATH \
        --cohort_path=$BASE_PATH'/cohort/cohort.parquet' \
        --features_row_id_map_path=$BASE_PATH'/merged_features_binary/features_sparse/features_row_id_map.parquet' \
        --config_path=$BASE_PATH'/experiments/'$EXPERIMENT_NAME'/config/'$1'/'$SLURM_ARRAY_TASK_ID'.yaml' \
        --experiment_name=$EXPERIMENT_NAME \
        --num_workers=5 \
        --data_mode="array" \
        --label_col=$1 \
        --sensitive_attribute=$2 \
        --fold_id=1 \
        --no_run_evaluation_group_standard \
        --no_run_evaluation_group_fair_ova \
        --run_evaluation_group_calibration \
        --eval_attributes "age_group" "gender_concept_name"
}

for task in $TASKS
do
    for attribute in $ATTRIBUTES
    do
        train_model_func $task $attribute
    done
done
