#!/bin/bash
DATA_PATH="/share/pi/nigam/projects/spfohl/cohorts/admissions/mimic_omop/"
EXPERIMENT_NAME='baseline_tuning_fold_1_10'
python -m  prediction_utils.experiments.fairness_benchmark.create_grid_baseline \
    --data_path=$DATA_PATH \
    --experiment_name=$EXPERIMENT_NAME \
    --grid_size=50 \
    --tasks "los_icu_3days" "los_icu_7days" "mortality_hospital" "mortality_icu"