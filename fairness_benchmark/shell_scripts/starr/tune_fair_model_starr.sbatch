#!/bin/bash 
#SBATCH --job-name=tune_fair_model_starr
#SBATCH --output=/home/spfohl/slurm_out/tune_fair_model_starr_%a.out
#SBATCH --error=/home/spfohl/slurm_out/tune_fair_model_starr_%a.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=50000
#SBATCH -c 5
#SBATCH --array=0-59
#SBATCH -p gpu
#SBATCH --gres=gpu:1

EXPERIMENT_NAME='fair_tuning_fold_1_10'
BASE_PATH='/share/pi/nigam/projects/spfohl/cohorts/admissions/starr_20200523'
TASKS="LOS_7 hospital_mortality readmission_30" 
ATTRIBUTES='gender_concept_name age_group race_eth'

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
        --sensitive_attribute=$2 \
        --fold_id=$3 \
        --run_evaluation \
        --run_evaluation_group \
        --run_evaluation_group_standard \
        --run_evaluation_group_fair_ova \
        --run_evaluation_group_calibration \
        --eval_attributes "age_group" "gender_concept_name" "race_eth"
}

for task in $TASKS
do
    for attribute in $ATTRIBUTES
    do
        for fold_id in {1..10}
        do
            train_model_func $task $attribute $fold_id
        done
    done
done
