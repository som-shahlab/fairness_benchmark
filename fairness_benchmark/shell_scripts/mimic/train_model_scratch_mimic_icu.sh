#!/bin/bash

EXPERIMENT_NAME='scratch'
BASE_PATH='/share/pi/nigam/projects/spfohl/cohorts/mimic_omop/icu_admission_cohort'

python -m prediction_utils.experiments.fairness_benchmark.train_model \
    --data_path=$BASE_PATH \
    --features_path=$BASE_PATH'/merged_features_binary/features_sparse/features.gz' \
    --cohort_path=$BASE_PATH'/cohort/cohort.parquet' \
    --vocab_path=$BASE_PATH'/merged_features_binary/vocab/vocab.parquet' \
    --features_row_id_map_path=$BASE_PATH'/merged_features_binary/features_sparse/features_row_id_map.parquet' \
    --experiment_name=$EXPERIMENT_NAME \
    --fold_id=1 \
    --label_col='los_icu_7days' \
    --num_workers=5 \
    --data_mode="array" \
    --drop_prob=0.75 \
    --gamma=1.0 \
    --num_epochs=50 \
    --batch_size=32 \
    --num_hidden=1 \
    --hidden_dim=64 \
    --early_stopping \
    --early_stopping_patience=10 \
    --run_evaluation \
    --run_evaluation_group \
    --run_evaluation_group_standard \
    --run_evaluation_group_fair_ova \
    --run_evaluation_group_calibration \
    --eval_attributes "age_group" "gender_concept_name" "race_eth"