#!/bin/bash 
#SBATCH --job-name=tune_baseline_model_optum
#SBATCH --output=/home/spfohl/slurm_out/tune_baseline_model_optum_%a.out
#SBATCH --error=/home/spfohl/slurm_out/tune_baseline_model_optum_%a.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=180GB
#SBATCH -c 3
#SBATCH --array=0-49%4
#SBATCH -p nigam
#SBATCH --gres=gpu:1

EXPERIMENT_NAME='baseline_tuning_fold_1'
BASE_PATH='/local-scratch-nvme/nigam/projects/spfohl/cohorts/admissions/optum'
TASKS='LOS_7 readmission_30'
source activate /share/pi/nigam/envs/anaconda/envs/prediction_utils

train_model_func() {
    python -m prediction_utils.experiments.fairness_benchmark.train_model \
        --data_path=$BASE_PATH \
        --features_path=$BASE_PATH'/merged_features_binary/features_sparse/features.gz' \
        --cohort_path=$BASE_PATH'/cohort/cohort.parquet' \
        --vocab_path=$BASE_PATH'/merged_features_binary/vocab/vocab.parquet' \
        --features_row_id_map_path=$BASE_PATH'/merged_features_binary/features_sparse/features_row_id_map.parquet' \
        --config_path=$BASE_PATH'/experiments/'$EXPERIMENT_NAME'/config/'$1'/'$SLURM_ARRAY_TASK_ID'.yaml' \
        --experiment_name=$EXPERIMENT_NAME \
        --num_workers=3 \
        --data_mode="array" \
        --label_col=$1 \
        --fold_id=1 \
        --run_evaluation \
        --run_evaluation_group \
        --run_evaluation_group_standard \
        --run_evaluation_group_fair_ova \
        --no_run_evaluation_group_calibration \
        --eval_attributes "age_group" "gender_concept_name"
}

for task in $TASKS
do
	train_model_func $task
done
