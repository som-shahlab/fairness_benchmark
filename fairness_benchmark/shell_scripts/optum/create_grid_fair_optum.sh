#!/bin/bash
DATA_PATH="/share/pi/nigam/projects/spfohl/cohorts/admissions/optum"
EXPERIMENT_NAME='fair_tuning_fold_1'
BASE_CONFIG_PATH=$DATA_PATH'/experiments/baseline_tuning_fold_1/config/selected_models'

python -m  prediction_utils.experiments.fairness_benchmark.create_grid_fair \
    --data_path=$DATA_PATH \
    --experiment_name=$EXPERIMENT_NAME \
    --tasks "LOS_7" "readmission_30" \
    --base_config_path=$BASE_CONFIG_PATH
