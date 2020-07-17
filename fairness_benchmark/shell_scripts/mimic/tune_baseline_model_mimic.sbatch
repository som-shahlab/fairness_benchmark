#!/bin/bash 
#SBATCH --job-name=tune_baseline_model_mimic
#SBATCH --output=/home/spfohl/slurm_out/tune_baseline_model_mimic_%a.out
#SBATCH --error=/home/spfohl/slurm_out/tune_baseline_model_mimic_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH -c 5
#SBATCH --array=0-49
#SBATCH -p gpu
#SBATCH --gres=gpu:1

EXPERIMENT_NAME='baseline_tuning_fold_1_10'
BASE_PATH='/share/pi/nigam/projects/spfohl/cohorts/admissions/mimic_omop/'
TASKS="los_icu_3days los_icu_7days mortality_hospital mortality_icu"

train_model_func() {
    /share/pi/nigam/envs/anaconda/envs/prediction_utils/bin/python -m prediction_utils.experiments.fairness_benchmark.train_model \
        --data_path=$BASE_PATH \
        --features_path=$BASE_PATH'/merged_features_binary/features_sparse/features.gz' \
        --cohort_path=$BASE_PATH'/cohort/cohort.parquet' \
        --vocab_path=$BASE_PATH'/merged_features_binary/vocab/vocab.parquet' \
        --features_row_id_map_path=$BASE_PATH'/merged_features_binary/features_sparse/features_row_id_map.parquet' \
        --config_path=$BASE_PATH'/experiments/'$EXPERIMENT_NAME'/config/'$1'/'$SLURM_ARRAY_TASK_ID'.yaml' \
        --experiment_name=$EXPERIMENT_NAME \
        --num_workers=5 \
        --data_mode="array" \
        --label_col=$1 \
        --fold_id=$2 \
        --run_evaluation \
        --run_evaluation_group \
        --run_evaluation_group_standard \
        --run_evaluation_group_fair_ova \
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
