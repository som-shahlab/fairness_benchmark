#!/bin/bash
DATA_PATH="/share/pi/nigam/projects/spfohl/cohorts/admissions/mimic_omop/"
EXPERIMENT_NAME='fair_tuning_fold_1_10'
BASE_CONFIG_PATH=$DATA_PATH'/experiments/baseline_tuning_fold_1_10/config/selected_models'

python -m  prediction_utils.experiments.fairness_benchmark.create_grid_fair \
    --data_path=$DATA_PATH \
    --experiment_name=$EXPERIMENT_NAME \
    --tasks "los_icu_3days" "los_icu_7days" "mortality_hospital" "mortality_icu" \
    --base_config_path=$BASE_CONFIG_PATH