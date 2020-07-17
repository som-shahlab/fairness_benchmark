#!/bin/bash
#SBATCH --job-name=evaluate_baseline_model_mimic
#SBATCH --output=/home/spfohl/slurm_out/evaluate_baseline_model_mimic_%a.out
#SBATCH --error=/home/spfohl/slurm_out/evaluate_baseline_model_mimic_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH -c 5
#SBATCH --array=0-49
#SBATCH -p normal,gpu

EXPERIMENT_NAME='baseline_tuning_fold_1_10'
BASE_PATH='/share/pi/nigam/projects/spfohl/cohorts/admissions/mimic_omop'
TASKS="los_icu_3days los_icu_7days mortality_hospital mortality_icu"
source activate /share/pi/nigam/envs/anaconda/envs/prediction_utils

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
        --fold_id=$2 \
        --no_run_evaluation_group_standard \
        --no_run_evaluation_group_fair_ova \
        --run_evaluation_group_calibration \
        --eval_attributes "age_group" "gender_concept_name" "race_eth"
}

for task in $TASKS
do
    for fold_id in {1..10}
    do
	   train_model_func $task $fold_id
    done
done
