#!/bin/bash
DATA_PATH="/share/pi/nigam/projects/spfohl/cohorts/admissions/starr_20200523"
EXPERIMENT_NAME='baseline_tuning_fold_1_10'
python -m  prediction_utils.experiments.fairness_benchmark.create_grid_baseline \
    --data_path=$DATA_PATH \
    --experiment_name=$EXPERIMENT_NAME \
    --grid_size=50 \
    --tasks "LOS_7" "hospital_mortality" "readmission_30"
