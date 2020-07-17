#!/bin/bash

EXPERIMENT_NAME='scratch_fair'
BASE_PATH='/share/pi/nigam/projects/spfohl/cohorts/admissions/optum_1pcnt'

python -m prediction_utils.experiments.fairness_benchmark.train_model \
    --data_path=$BASE_PATH \
    --features_path=$BASE_PATH'/merged_features_binary/features_sparse/features.gz' \
    --cohort_path=$BASE_PATH'/cohort/cohort.parquet' \
    --vocab_path=$BASE_PATH'/merged_features_binary/vocab/vocab.parquet' \
    --features_row_id_map_path=$BASE_PATH'/merged_features_binary/features_sparse/features_row_id_map.parquet' \
    --experiment_name=$EXPERIMENT_NAME \
    --fold_id=1 \
    --label_col='readmission_30' \
    --num_workers=5 \
    --data_mode="array" \
    --drop_prob=0.5 \
    --gamma=1.0 \
    --num_epochs=2 \
    --batch_size=128 \
    --num_hidden=2 \
    --hidden_dim=128 \
    --early_stopping \
    --early_stopping_patience=10 \
    --run_evaluation \
    --run_evaluation_group \
    --run_evaluation_group_standard \
    --run_evaluation_group_fair_ova \
    --run_evaluation_group_calibration \
    --sensitive_attribute="age_group" \
    --regularization_metric="mmd" \
    --mmd_mode="conditional" \
    --lambda_group_regularization=0.01 \
    --eval_attributes "age_group" "gender_concept_name"
