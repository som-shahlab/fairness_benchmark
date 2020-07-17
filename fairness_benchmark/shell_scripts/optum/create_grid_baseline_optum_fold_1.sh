#!/bin/bash
DATA_PATH="/local-scratch-nvme/nigam/projects/spfohl/cohorts/admissions/optum/"
EXPERIMENT_NAME='baseline_tuning_fold_1'
/share/pi/nigam/envs/anaconda/envs/prediction_utils/bin/python -m  prediction_utils.experiments.fairness_benchmark.create_grid_baseline \
    --data_path=$DATA_PATH \
    --experiment_name=$EXPERIMENT_NAME \
    --grid_size=50 \
    --tasks "LOS_7" "readmission_30"
