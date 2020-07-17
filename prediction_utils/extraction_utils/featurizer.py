import numpy as np
import pandas as pd
import os
import dask
import dask.dataframe as dd
from dask.distributed import Client
import scipy.sparse as sp
import joblib

import glob
from prediction_utils.util import overwrite_dir
from prediction_utils.extraction_utils.database import BQDatabase


class FeatureQuery:
    def __init__(self, *args, **kwargs):
        self.config_dict = self.get_config_dict()
        self.base_query = self.get_base_query()

    def get_config_dict(self):
        return {**self.get_base_config(), **self.get_query_config()}

    def get_base_config(self):
        return {"requires_time_bin": False, "requires_time_bin_hourly": False}

    def get_query_config(self):
        raise NotImplementedError


class CountQuery(FeatureQuery):
    """
    A query that counts occurrences of concept_ids
    """

    def get_base_config(self):
        return {"requires_time_bin": True, "requires_time_bin_hourly": False}

    def get_base_query(self):
        """
        A generic query that can be used to get the counts of unique concepts.
        Allows for binning by time
        """
        return """
            WITH source_table as (
                SELECT 
                    {row_id_field},
                    t1.person_id,
                    CAST(t2.{index_date_field} AS DATE) as index_date, 
                    {concept_id} AS concept_id, 
                    CAST({concept_date} AS DATE) AS concept_date,
                    '{analysis_id}' AS analysis_id,
                    CONCAT('bin_', {bin_left}, '_', {bin_right}) AS time_bin
                FROM {dataset_project}.{dataset}.{source_table} t1
                INNER JOIN {rs_dataset_project}.{rs_dataset}.{cohort_name} AS t2 ON
                    t1.person_id = t2.person_id
                INNER JOIN {dataset_project}.{dataset}.concept AS t3 ON
                    t1.{concept_id} = t3.concept_id
                WHERE 
                    CAST({concept_date} AS DATE) BETWEEN 
                        DATE_ADD(CAST(t2.{index_date_field} AS DATE), INTERVAL {bin_left} DAY) AND
                        DATE_ADD(CAST(t2.{index_date_field} AS DATE), INTERVAL {bin_right} DAY)
                    AND standard_concept = 'S'
                {limit_str}
            )
            SELECT 
                {row_id_field}, person_id, index_date, concept_id, analysis_id, time_bin, 
                CONCAT(concept_id, '_', time_bin, '_', analysis_id) AS feature_id,
                COUNT(*) AS concept_count
            FROM source_table
            GROUP BY {row_id_field}, person_id, index_date, concept_id, analysis_id, time_bin
        """


class ConditionOccurrenceCountQuery(CountQuery):
    def get_query_config(self):
        return {
            "concept_id": "condition_concept_id",
            "concept_date": "condition_start_date",
            "source_table": "condition_occurrence",
            "analysis_id": "condition_occurrence",
        }


class DrugExposureCountQuery(CountQuery):
    def get_query_config(self):
        return {
            "concept_id": "drug_concept_id",
            "concept_date": "drug_exposure_start_date",
            "source_table": "drug_exposure",
            "analysis_id": "drug_exposure",
        }


class DeviceExposureCountQuery(CountQuery):
    def get_query_config(self):
        return {
            "concept_id": "device_concept_id",
            "concept_date": "device_exposure_start_date",
            "source_table": "device_exposure",
            "analysis_id": "device_exposure",
        }


class MeasurementCountQuery(CountQuery):
    def get_query_config(self):
        return {
            "concept_id": "measurement_concept_id",
            "concept_date": "measurement_date",
            "source_table": "measurement",
            "analysis_id": "measurement",
        }


class ProcedureOccurrenceCountQuery(CountQuery):
    def get_query_config(self):
        return {
            "concept_id": "procedure_concept_id",
            "concept_date": "procedure_date",
            "source_table": "procedure_occurrence",
            "analysis_id": "procedure_occurrence",
        }


class NoteTypeCountQuery(CountQuery):
    def get_query_config(self):
        return {
            "concept_id": "note_type_concept_id",
            "concept_date": "note_date",
            "source_table": "note",
            "analysis_id": "note_type",
        }


class ObservationCountQuery(CountQuery):
    def get_query_config(self):
        return {
            "concept_id": "observation_concept_id",
            "concept_date": "observation_date",
            "source_table": "observation",
            "analysis_id": "observation",
        }


## Datetime queries that bin on an hourly basis
class CountDTQuery(FeatureQuery):
    """
    A query that counts occurrences of concepts stored in datetime
    Will only pull data elements that are not recorded at midnight.
    Bins are assumed to be hourly
    """

    def get_base_config(self):
        return {"requires_time_bin": False, "requires_time_bin_hourly": True}

    def get_base_query(self):
        return """
            WITH source_table as (
                SELECT 
                    {row_id_field},
                    t1.person_id,
                    CAST(t2.{index_date_field} AS DATETIME) as index_date, 
                    {concept_id} AS concept_id, 
                    {concept_datetime} AS concept_datetime,
                    '{analysis_id}' AS analysis_id,
                    CONCAT('bin_', {bin_left}, '_', {bin_right}) AS time_bin
                FROM {dataset_project}.{dataset}.{source_table} t1
                INNER JOIN {rs_dataset_project}.{rs_dataset}.{cohort_name} AS t2 ON
                    t1.person_id = t2.person_id
                INNER JOIN {dataset_project}.{dataset}.concept AS t3 ON
                    t1.{concept_id} = t3.concept_id
                WHERE 
                    {concept_datetime} BETWEEN 
                        DATETIME_ADD(CAST(t2.{index_date_field} AS DATETIME), INTERVAL {bin_left} HOUR) AND
                        DATETIME_ADD(CAST(t2.{index_date_field} AS DATETIME), INTERVAL {bin_right} HOUR)
                    AND standard_concept = 'S'
                    AND CAST({concept_date} AS DATETIME) != {concept_datetime}
                {limit_str}
            )
            SELECT 
                {row_id_field}, person_id, index_date, concept_id, analysis_id, time_bin, 
                CONCAT(concept_id, '_', time_bin, '_', analysis_id) AS feature_id,
                COUNT(*) AS concept_count
            FROM source_table
            GROUP BY {row_id_field}, person_id, index_date, concept_id, analysis_id, time_bin
        """


class NoteNLPCountQuery(CountQuery):
    def get_base_query(self):
        """
        A generic query that can be used to get the counts of unique concepts.
        Allows for binning by time
        """
        return """
            WITH source_table as (
                SELECT 
                    {row_id_field},
                    t2.person_id,
                    CAST(t2.{index_date_field} AS DATE) as index_date, 
                    CONCAT(CAST(note_nlp_concept_id AS STRING), '_', term_exists) AS concept_id,
                    '{analysis_id}' AS analysis_id,
                    CONCAT('bin_', {bin_left}, '_', {bin_right}) AS time_bin
                FROM {dataset_project}.{dataset}.note t1
                INNER JOIN {rs_dataset_project}.{rs_dataset}.{cohort_name} AS t2 ON
                    t1.person_id = t2.person_id
                INNER JOIN {dataset_project}.{dataset}.note_nlp AS t3 ON
                    t1.note_id = t3.note_id
                INNER JOIN {dataset_project}.{dataset}.concept AS t4 ON
                    t3.note_nlp_concept_id = t4.concept_id       
                WHERE 
                    CAST(note_date AS DATE) BETWEEN 
                        DATE_ADD(CAST(t2.{index_date_field} AS DATE), INTERVAL {bin_left} DAY) AND
                        DATE_ADD(CAST(t2.{index_date_field} AS DATE), INTERVAL {bin_right} DAY)
                    AND standard_concept = 'S'
                {limit_str}
            )
            SELECT 
                {row_id_field}, person_id, index_date, concept_id, analysis_id, time_bin, 
                CONCAT(concept_id, '_', time_bin, '_', analysis_id) AS feature_id,
                COUNT(*) AS concept_count
            FROM source_table
            GROUP BY {row_id_field}, person_id, index_date, concept_id, analysis_id, time_bin
        """

    def get_query_config(self):
        return {"analysis_id": "note_nlp"}


class MeasurementRangeCountQuery(CountQuery):
    def get_base_query(self):
        """
        A generic query that can be used to get the counts of unique concepts.
        Allows for binning by time
        """
        return """
            WITH source_table as (
                SELECT 
                    {row_id_field},
                    t1.person_id,
                    CAST(t2.{index_date_field} AS DATE) as index_date, 
                    CAST(measurement_date AS DATE) AS concept_date,
                    '{analysis_id}' AS analysis_id,
                    CONCAT('bin_', {bin_left}, '_', {bin_right}) AS time_bin,
                    CONCAT(
                        CAST(measurement_concept_id AS STRING),
                        '_',
                        CASE
                            WHEN value_as_number > range_high THEN 'abnormal_high'
                            WHEN value_as_number < range_low THEN 'abnormal_low'
                            ELSE 'normal'
                        END
                    ) AS concept_id
                FROM {dataset_project}.{dataset}.measurement t1
                INNER JOIN {rs_dataset_project}.{rs_dataset}.{cohort_name} AS t2 ON
                    t1.person_id = t2.person_id
                INNER JOIN {dataset_project}.{dataset}.concept AS t3 ON
                    t1.measurement_concept_id = t3.concept_id
                WHERE 
                    CAST(measurement_date AS DATE) BETWEEN 
                        DATE_ADD(CAST(t2.{index_date_field} AS DATE), INTERVAL {bin_left} DAY) AND
                        DATE_ADD(CAST(t2.{index_date_field} AS DATE), INTERVAL {bin_right} DAY)
                    AND standard_concept = 'S'
                    AND value_as_number is NOT NULL
                    AND range_low is NOT NULL
                    AND range_high is NOT NULL
                {limit_str}
            )
            SELECT 
                {row_id_field}, person_id, index_date, concept_id, analysis_id, time_bin, 
                CONCAT(concept_id, '_', time_bin, '_', analysis_id) AS feature_id,
                COUNT(*) AS concept_count
            FROM source_table
            GROUP BY {row_id_field}, person_id, index_date, concept_id, analysis_id, time_bin
        """

    def get_query_config(self):
        return {
            "analysis_id": "measurement_range",
        }


class MeasurementBinCountQuery(CountQuery):
    def get_base_query(self):
        return """
            WITH source_table AS (
                SELECT measurement_concept_id, value_as_number,
                        {row_id_field},
                        t1.person_id,
                        CAST(t2.{index_date_field} AS DATE) as index_date, 
                        CAST(measurement_date AS DATE) AS concept_date,
                        CONCAT('bin_', {bin_left}, '_', {bin_right}) AS time_bin,
                        '{analysis_id}' AS analysis_id,
                FROM {dataset_project}.{dataset}.measurement t1
                INNER JOIN {rs_dataset_project}.{rs_dataset}.{cohort_name} AS t2 ON
                    t1.person_id = t2.person_id
                INNER JOIN {dataset_project}.{dataset}.concept AS t3 ON
                    t1.measurement_concept_id = t3.concept_id
                WHERE 
                    CAST(measurement_date AS DATE) BETWEEN 
                        DATE_ADD(CAST(t2.{index_date_field} AS DATE), INTERVAL {bin_left} DAY) AND
                        DATE_ADD(CAST(t2.{index_date_field} AS DATE), INTERVAL {bin_right} DAY)
                    AND standard_concept = 'S'
                    AND value_as_number is NOT NULL
            ),
            quantiles_raw AS (
                SELECT APPROX_QUANTILES(value_as_number, {num_bins_measurement}) as quantiles, measurement_concept_id
                FROM source_table
                GROUP BY measurement_concept_id
            ),
            quantile_start_table AS (
                SELECT measurement_concept_id, quantile_start, ROW_NUMBER() OVER(PARTITION BY measurement_concept_id ORDER BY quantile_start) as quantile_id
                FROM quantiles_raw
                CROSS JOIN UNNEST(quantiles_raw.quantiles) AS quantile_start
            ),
            quantile_end_table AS (
                SELECT * EXCEPT (quantile_start, quantile_id), quantile_id - 1 as quantile_id, quantile_start as quantile_end,
                FROM quantile_start_table
            ),
            merged_quantiles AS (
                SELECT *, 
                    CONCAT(
                        measurement_concept_id, 
                        '_bin_', 
                        quantile_id, 
                        '_', 
                        quantile_start, 
                        '_', 
                        quantile_end
                    ) as concept_id
                FROM quantile_start_table
                INNER JOIN quantile_end_table USING (measurement_concept_id, quantile_id)
            ),
            source_with_quantiles AS (
                SELECT *
                FROM source_table
                INNER JOIN merged_quantiles USING (measurement_concept_id)
                WHERE value_as_number BETWEEN quantile_start AND quantile_end
            )
            SELECT {row_id_field}, person_id, index_date, concept_id, analysis_id, time_bin, 
                    CONCAT(concept_id, '_', time_bin, '_', analysis_id) AS feature_id,
                    COUNT(*) AS concept_count
            FROM source_with_quantiles
            GROUP BY {row_id_field}, person_id, index_date, concept_id, analysis_id, time_bin
        """

    def get_query_config(self):
        return {"analysis_id": "measurement_bin", "num_bins_measurement": 5}


## Datetime queries with hourly time bins


class ConditionOccurrenceCountDTQuery(CountDTQuery):
    def get_query_config(self):
        return {
            "concept_id": "condition_concept_id",
            "concept_date": "condition_start_date",
            "concept_datetime": "condition_start_datetime",
            "source_table": "condition_occurrence",
            "analysis_id": "condition_occurrence_dt",
        }


class DrugExposureCountDTQuery(CountDTQuery):
    def get_query_config(self):
        return {
            "concept_id": "drug_concept_id",
            "concept_date": "drug_exposure_start_date",
            "concept_datetime": "drug_exposure_start_datetime",
            "source_table": "drug_exposure",
            "analysis_id": "drug_exposure_dt",
        }


class DeviceExposureCountDTQuery(CountDTQuery):
    def get_query_config(self):
        return {
            "concept_id": "device_concept_id",
            "concept_date": "device_exposure_start_date",
            "concept_datetime": "device_exposure_start_datetime",
            "source_table": "device_exposure",
            "analysis_id": "device_exposure_dt",
        }


class MeasurementCountDTQuery(CountDTQuery):
    def get_query_config(self):
        return {
            "concept_id": "measurement_concept_id",
            "concept_date": "measurement_date",
            "concept_datetime": "measurement_datetime",
            "source_table": "measurement",
            "analysis_id": "measurement_dt",
        }


class ProcedureOccurrenceCountDTQuery(CountDTQuery):
    def get_query_config(self):
        return {
            "concept_id": "procedure_concept_id",
            "concept_date": "procedure_date",
            "concept_datetime": "procedure_datetime",
            "source_table": "procedure_occurrence",
            "analysis_id": "procedure_occurrence_dt",
        }


class NoteTypeCountDTQuery(CountDTQuery):
    def get_query_config(self):
        return {
            "concept_id": "note_type_concept_id",
            "concept_date": "note_date",
            "concept_datetime": "note_datetime",
            "source_table": "note",
            "analysis_id": "note_type_dt",
        }


class ObservationCountDTQuery(CountDTQuery):
    def get_query_config(self):
        return {
            "concept_id": "observation_concept_id",
            "concept_date": "observation_date",
            "concept_datetime": "observation_datetime",
            "source_table": "observation",
            "analysis_id": "observation_dt",
        }


class NoteNLPCountDTQuery(CountDTQuery):
    def get_base_query(self):
        """
        A generic query that can be used to get the counts of unique concepts.
        Allows for binning by time
        """
        return """
            WITH source_table as (
                SELECT 
                    {row_id_field},
                    t2.person_id,
                    CAST(t2.{index_date_field} AS DATETIME) as index_date, 
                    CONCAT(CAST(note_nlp_concept_id AS STRING), '_', term_exists) AS concept_id,
                    '{analysis_id}' AS analysis_id,
                    CONCAT('bin_', {bin_left}, '_', {bin_right}) AS time_bin
                FROM {dataset_project}.{dataset}.note t1
                INNER JOIN {rs_dataset_project}.{rs_dataset}.{cohort_name} AS t2 ON
                    t1.person_id = t2.person_id
                INNER JOIN {dataset_project}.{dataset}.note_nlp AS t3 ON
                    t1.note_id = t3.note_id
                INNER JOIN {dataset_project}.{dataset}.concept AS t4 ON
                    t3.note_nlp_concept_id = t4.concept_id       
                WHERE 
                    note_datetime BETWEEN 
                        DATETIME_ADD(CAST(t2.{index_date_field} AS DATETIME), INTERVAL {bin_left} HOUR) AND
                        DATETIME_ADD(CAST(t2.{index_date_field} AS DATETIME), INTERVAL {bin_right} HOUR)
                    AND standard_concept = 'S'
                    AND CAST(note_date AS DATETIME) != note_datetime
                {limit_str}
            )
            SELECT 
                {row_id_field}, person_id, index_date, concept_id, analysis_id, time_bin, 
                CONCAT(concept_id, '_', time_bin, '_', analysis_id) AS feature_id,
                COUNT(*) AS concept_count
            FROM source_table
            GROUP BY {row_id_field}, person_id, index_date, concept_id, analysis_id, time_bin
        """

    def get_query_config(self):
        return {"analysis_id": "note_nlp_dt"}


class MeasurementRangeCountDTQuery(CountDTQuery):
    def get_base_query(self):
        return """
            WITH source_table as (
                SELECT 
                    {row_id_field},
                    t1.person_id,
                    CAST(t2.{index_date_field} AS DATETIME) as index_date, 
                    '{analysis_id}' AS analysis_id,
                    CONCAT('bin_', {bin_left}, '_', {bin_right}) AS time_bin,
                    CONCAT(
                        CAST(measurement_concept_id AS STRING),
                        '_',
                        CASE
                            WHEN value_as_number > range_high THEN 'abnormal_high'
                            WHEN value_as_number < range_low THEN 'abnormal_low'
                            ELSE 'normal'
                        END
                    ) AS concept_id
                FROM {dataset_project}.{dataset}.measurement t1
                INNER JOIN {rs_dataset_project}.{rs_dataset}.{cohort_name} AS t2 ON
                    t1.person_id = t2.person_id
                INNER JOIN {dataset_project}.{dataset}.concept AS t3 ON
                    t1.measurement_concept_id = t3.concept_id
                WHERE 
                    measurement_datetime BETWEEN 
                        DATETIME_ADD(CAST(t2.{index_date_field} AS DATETIME), INTERVAL {bin_left} HOUR) AND
                        DATETIME_ADD(CAST(t2.{index_date_field} AS DATETIME), INTERVAL {bin_right} HOUR)
                    AND standard_concept = 'S'
                    AND value_as_number is NOT NULL
                    AND range_low is NOT NULL
                    AND range_high is NOT NULL
                    AND CAST(measurement_date AS DATETIME) != measurement_datetime
                {limit_str}
            )
            SELECT 
                {row_id_field}, person_id, index_date, concept_id, analysis_id, time_bin, 
                CONCAT(concept_id, '_', time_bin, '_', analysis_id) AS feature_id,
                COUNT(*) AS concept_count
            FROM source_table
            GROUP BY {row_id_field}, person_id, index_date, concept_id, analysis_id, time_bin
        """

    def get_query_config(self):
        return {
            "analysis_id": "measurement_range_dt",
        }


class MeasurementBinCountDTQuery(CountDTQuery):
    def get_base_query(self):
        return """
            WITH source_table AS (
                SELECT measurement_concept_id, value_as_number,
                        {row_id_field},
                        t1.person_id,
                        CAST(t2.{index_date_field} AS DATETIME) as index_date, 
                        -- CAST(measurement_date AS DATE) AS concept_date,
                        CONCAT('bin_', {bin_left}, '_', {bin_right}) AS time_bin,
                        '{analysis_id}' AS analysis_id,
                FROM {dataset_project}.{dataset}.measurement t1
                INNER JOIN {rs_dataset_project}.{rs_dataset}.{cohort_name} AS t2 ON
                    t1.person_id = t2.person_id
                INNER JOIN {dataset_project}.{dataset}.concept AS t3 ON
                    t1.measurement_concept_id = t3.concept_id
                WHERE 
                    measurement_datetime BETWEEN 
                        DATETIME_ADD(CAST(t2.{index_date_field} AS DATETIME), INTERVAL {bin_left} HOUR) AND
                        DATETIME_ADD(CAST(t2.{index_date_field} AS DATETIME), INTERVAL {bin_right} HOUR)
                    AND standard_concept = 'S'
                    AND value_as_number is NOT NULL
                    AND CAST(measurement_date AS DATETIME) != measurement_datetime
            ),
            quantiles_raw AS (
                SELECT APPROX_QUANTILES(value_as_number, {num_bins_measurement}) as quantiles, measurement_concept_id
                FROM source_table
                GROUP BY measurement_concept_id
            ),
            quantile_start_table AS (
                SELECT measurement_concept_id, quantile_start, ROW_NUMBER() OVER(PARTITION BY measurement_concept_id ORDER BY quantile_start) as quantile_id
                FROM quantiles_raw
                CROSS JOIN UNNEST(quantiles_raw.quantiles) AS quantile_start
            ),
            quantile_end_table AS (
                SELECT * EXCEPT (quantile_start, quantile_id), quantile_id - 1 as quantile_id, quantile_start as quantile_end,
                FROM quantile_start_table
            ),
            merged_quantiles AS (
                SELECT *, 
                    CONCAT(
                        measurement_concept_id, 
                        '_bin_', 
                        quantile_id, 
                        '_', 
                        quantile_start, 
                        '_', 
                        quantile_end
                    ) as concept_id
                FROM quantile_start_table
                INNER JOIN quantile_end_table USING (measurement_concept_id, quantile_id)
            ),
            source_with_quantiles AS (
                SELECT *
                FROM source_table
                INNER JOIN merged_quantiles USING (measurement_concept_id)
                WHERE value_as_number BETWEEN quantile_start AND quantile_end
            )
            SELECT {row_id_field}, person_id, index_date, concept_id, analysis_id, time_bin, 
                    CONCAT(concept_id, '_', time_bin, '_', analysis_id) AS feature_id,
                    COUNT(*) AS concept_count
            FROM source_with_quantiles
            GROUP BY {row_id_field}, person_id, index_date, concept_id, analysis_id, time_bin
        """

    def get_query_config(self):
        return {"analysis_id": "measurement_bin_dt", "num_bins_measurement": 5}


class DemographicsQuery(FeatureQuery):
    def get_base_config(self):
        return {"requires_time_bin": False, "requires_time_bin_hourly": False}

    def get_base_query(self):
        return """
            WITH source_table as (
                SELECT 
                    {row_id_field}, 
                    t1.person_id, 
                    t1.{concept_id} AS concept_id, 
                    t2.{index_date_field} AS index_date,
                    '{analysis_id}' as analysis_id
                FROM {dataset_project}.{dataset}.person t1
                INNER JOIN {rs_dataset_project}.{rs_dataset}.{cohort_name} AS t2 ON
                    t1.person_id = t2.person_id
                {limit_str}
            )
                SELECT {row_id_field}, person_id, index_date, 
                concept_id, analysis_id, 'static' as time_bin, 
                CONCAT(concept_id, '_', analysis_id) AS feature_id,
                1 AS concept_count
            FROM source_table
        """


class GenderQuery(DemographicsQuery):
    def get_query_config(self):
        return {"concept_id": "gender_concept_id", "analysis_id": "gender"}


class RaceQuery(DemographicsQuery):
    def get_query_config(self):
        return {"concept_id": "race_concept_id", "analysis_id": "race"}


class EthnicityQuery(DemographicsQuery):
    def get_query_config(self):
        return {
            "concept_id": "ethnicity_concept_id",
            "analysis_id": "ethnicity",
        }


class AgeGroupQuery(FeatureQuery):
    def get_base_config(self):
        return {"requires_time_bin": False, "requires_time_bin_hourly": False}

    def get_base_query(self):
        return """
            WITH age_group_query AS (
                SELECT {row_id_field}, t1.person_id, '{analysis_id}' as analysis_id, {index_date_field} AS index_date,
                    COALESCE(
                        DATE_DIFF(CAST({index_date_field} AS DATE), CAST(t1.birth_datetime AS DATE), YEAR),
                        EXTRACT(YEAR FROM CAST({index_date_field} AS DATE)) - year_of_birth
                    ) as age_in_years
                    FROM {dataset_project}.{dataset}.person t1
                    INNER JOIN {rs_dataset_project}.{rs_dataset}.{cohort_name} AS t2 ON
                        t1.person_id = t2.person_id
                    {limit_str}
            ),
            source_table AS (
                SELECT *,
                CONCAT('age_group_', CAST(FLOOR(SAFE_DIVIDE(age_in_years, CAST({age_bin_size} AS INT64))) AS STRING)) AS concept_id
                FROM age_group_query
            )
            SELECT {row_id_field}, person_id, index_date, 
                concept_id, analysis_id, 'static' as time_bin, 
                CONCAT(concept_id, '_', analysis_id) AS feature_id,
                1 AS concept_count
            FROM source_table
        """

    def get_query_config(self):
        return {"analysis_id": "age_group", "age_bin_size": 5}


class BigQueryOMOPFeaturizer:
    """
    Executes feature extraction against an OMOP CDM database stored in Google BigQuery
    """

    def __init__(self, **kwargs):
        """
        Args:
            data_path: the root directory path where the resulting data will be stored
            gcloud_storage_path: a google cloud storage bucket path where results can be stored
            features_by_analysis_path: the name of a subdirectory to be created within data_path
            gcloud_project: the name of the default GCP project
            dataset_project: the name of the project where the source data is stored
            rs_dataset_project: the name of the project where the cohort table is stored
            dataset: the name of the GCP dataset where the source data is stored
            rs_dataset: the name of the GCP dataset where the cohort table is stored
            features_dataset: the name of the GCP dataset where features tables may be written
            features_prefix: a string prefix for features tables
            cohort_name: the name of the table in rs_dataset that defines the cohort
            row_id_field: the name of the field in the cohort table that identifies unique predictions
            index_date_field: the name of the field in the cohort table that defines the index date
            limit: a limit on queries applied to the cohort table. None grabs all data
            google_application_credentials: path to google application credentials
            overwrite: whether extracted features should overwrite existing extraction
            merged_name: A directory name for the merged features
            binary: whether to assign a binary filter (value > 0) to merged features
            time_bins: a list of time bins
        """
        self.config_dict = self.get_config_dict(**kwargs)
        self.query_dict = self.get_default_queries()
        self.valid_queries = list(self.query_dict.keys())
        self.time_bin_dict = self.get_time_bin_dict(
            bins=self.config_dict["time_bins"],
            include_all_history=self.config_dict["include_all_history"],
            inclusive_right=False,
        )

        self.time_bin_hourly_dict = self.get_time_bin_dict(
            bins=self.config_dict["time_bins_hourly"],
            include_all_history=self.config_dict["include_all_history"],
            inclusive_right=True,
        )

        self.db = BQDatabase(
            gcloud_project=self.config_dict["gcloud_project"],
            google_application_credentials=self.config_dict[
                "google_application_credentials"
            ],
        )

    def get_defaults(self):
        """
        Defines default config_dict parameters
        """
        return {
            "data_path": "/share/pi/nigam/projects/spfohl/cohorts/scratch",
            "gcloud_storage_path": "gs://feature_extraction_exports/cohorts/scratch/",
            "features_by_analysis_path": "features_by_analysis",
            "dataset": "starr_omop_cdm5_deid_20200404",
            "rs_dataset": "temp_dataset",
            "features_dataset": "temp_dataset",
            "features_prefix": "features",
            "cohort_name": "temp_cohort",
            "row_id_field": "prediction_id",
            "index_date_field": "admit_date",
            "time_bins": [-365, -180, -90, -30, 0],
            "time_bins_hourly": [-7 * 24, -24 * 3, -24, -12, -4, 0],
            "include_all_history": True,
            "limit": None,
            "gcloud_project": "som-nero-phi-nigam-starr",
            "dataset_project": None,
            "rs_dataset_project": None,
            "google_application_credentials": os.path.expanduser(
                "~/.config/gcloud/application_default_credentials.json"
            ),
            "overwrite": False,
            "merged_name": "merged_features_binary",
            "binary": True,
        }

    def override_defaults(self, **kwargs):
        return {**self.get_defaults(), **kwargs}

    def get_config_dict(self, **kwargs):
        """
        Gets the config_dict defaults and formats some additional elements
        """
        config_dict = self.override_defaults(**kwargs)

        # Handle special parameters
        config_dict["limit_str"] = (
            "LIMIT {}".format(config_dict["limit"])
            if (
                (config_dict["limit"] is not None)
                and (config_dict["limit"] != "")
                and (config_dict["limit"] != 0)
            )
            else ""
        )
        config_dict["dataset_project"] = (
            config_dict["dataset_project"]
            if (
                (config_dict["dataset_project"] is not None)
                and (config_dict["dataset_project"] != "")
            )
            else config_dict["gcloud_project"]
        )
        config_dict["rs_dataset_project"] = (
            config_dict["rs_dataset_project"]
            if (
                (config_dict["rs_dataset_project"] is not None)
                and (config_dict["rs_dataset_project"] != "")
            )
            else config_dict["gcloud_project"]
        )
        return config_dict

    def get_default_queries(self):
        query_classes = [
            ConditionOccurrenceCountQuery(),
            DrugExposureCountQuery(),
            DeviceExposureCountQuery(),
            MeasurementCountQuery(),
            ProcedureOccurrenceCountQuery(),
            NoteTypeCountQuery(),
            ObservationCountQuery(),
            NoteNLPCountQuery(),
            MeasurementRangeCountQuery(),
            MeasurementBinCountQuery(),
            ConditionOccurrenceCountDTQuery(),
            DrugExposureCountDTQuery(),
            DeviceExposureCountDTQuery(),
            MeasurementCountDTQuery(),
            ProcedureOccurrenceCountDTQuery(),
            NoteTypeCountDTQuery(),
            ObservationCountDTQuery(),
            NoteNLPCountDTQuery(),
            MeasurementRangeCountDTQuery(),
            MeasurementBinCountDTQuery(),
            GenderQuery(),
            RaceQuery(),
            EthnicityQuery(),
            AgeGroupQuery(),
        ]
        return {
            query_class.config_dict["analysis_id"]: query_class
            for query_class in query_classes
        }

    def get_time_bin_dict(
        self,
        bins=None,
        include_all_history=True,
        inclusive_right=False,
        all_history_bound=-100 * 365,
    ):
        """
        Construct a dictionary of time bins from bins.
        include_all_history: includes a general time bin of the last 100 years of history
        inclusive_right: Whether bins are inclusive on the right.
            For expected behavior, set to False if binning dates, and True if binning datetimes
        Example: 
            bins = [-365, -180, -90, -30, 0], include_all_history=True
            returns [{'bin_left':-36500, 'bin_right': -1}
                     {'bin_left': -365, 'bin_right': -181},
                     {'bin_left': -180, 'bin_right': -91},
                     {'bin_left': -90, 'bin_right': -31},
                     {'bin_left': -30, 'bin_right': -1}]
        """
        if (bins is None) and (not include_all_history):
            raise ValueError("if bins is None, include_all_history must be true")

        bin_right_correction = 0 if inclusive_right else -1
        if include_all_history:
            result = [
                {"bin_left": all_history_bound, "bin_right": bin_right_correction}
            ]
        else:
            result = []

        if bins is not None:
            for i in range(len(bins) - 1):
                result.append(
                    {
                        "bin_left": bins[i],
                        "bin_right": bins[i + 1] + bin_right_correction,
                    }
                )
        return result

    def featurize(self, analysis_ids=None, exclude_analysis_ids=None):
        """
        Runs the feature extraction pipeline for the set of analysis_ids.
        """
        if analysis_ids is None:
            analysis_ids = self.query_dict.keys()

        if exclude_analysis_ids is not None:
            analysis_ids = [
                analysis_id
                for analysis_id in analysis_ids
                if analysis_id not in exclude_analysis_ids
            ]
        for analysis_id in analysis_ids:
            if analysis_id not in self.query_dict.keys():
                raise ValueError("Provided analysis_id not defined")

        query_dict = {key: self.query_dict[key] for key in analysis_ids}

        features_path = os.path.join(
            self.config_dict["data_path"], self.config_dict["features_by_analysis_path"]
        )
        for analysis, query in query_dict.items():
            # Time binned queries
            if query.config_dict["requires_time_bin"]:
                for time_bin in self.time_bin_dict:
                    formatted_query = query.base_query.format_map(
                        {**self.config_dict, **query.config_dict, **time_bin}
                    )
                    output_path = os.path.join(
                        features_path,
                        analysis,
                        "bin_{bin_left}_{bin_right}".format_map(time_bin),
                    )
                    self.db.stream_query(
                        query=formatted_query,
                        output_path=output_path,
                        overwrite=self.config_dict["overwrite"],
                    )
            elif query.config_dict["requires_time_bin_hourly"]:
                for time_bin in self.time_bin_hourly_dict:
                    formatted_query = query.base_query.format_map(
                        {**self.config_dict, **query.config_dict, **time_bin}
                    )
                    output_path = os.path.join(
                        features_path,
                        analysis,
                        "bin_hourly_{bin_left}_{bin_right}".format_map(time_bin),
                    )
                    self.db.stream_query(
                        query=formatted_query,
                        output_path=output_path,
                        overwrite=self.config_dict["overwrite"],
                    )
            # Not time binned_queries
            else:
                formatted_query = query.base_query.format_map(
                    {**self.config_dict, **query.config_dict}
                )
                output_path = os.path.join(features_path, analysis, "static")
                self.db.stream_query(
                    query=formatted_query,
                    output_path=output_path,
                    overwrite=self.config_dict["overwrite"],
                )

    def featurize_to_destination(
        self, analysis_ids=None, exclude_analysis_ids=None, merge_features=False
    ):
        """
        Runs the feature extraction pipeline for the set of analysis_ids.
        """
        if analysis_ids is None:
            analysis_ids = self.query_dict.keys()

        if exclude_analysis_ids is not None:
            analysis_ids = [
                analysis_id
                for analysis_id in analysis_ids
                if analysis_id not in exclude_analysis_ids
            ]
        for analysis_id in analysis_ids:
            if analysis_id not in self.query_dict.keys():
                raise ValueError("Provided analysis_id not defined")

        query_dict = {key: self.query_dict[key] for key in analysis_ids}

        destination_tables = []
        for analysis, query in query_dict.items():
            # Time binned queries
            if query.config_dict["requires_time_bin"]:
                for time_bin in self.time_bin_dict:
                    formatted_query = query.base_query.format_map(
                        {**self.config_dict, **query.config_dict, **time_bin}
                    )
                    destination_table = "{rs_dataset_project}.{features_dataset}.{features_prefix}_{analysis}_bin_{bin_left}_{bin_right}".format_map(
                        {
                            **self.config_dict,
                            **query.config_dict,
                            **{key: abs(value) for key, value in time_bin.items()},
                            **{"analysis": analysis},
                        }
                    )
                    destination_tables.append(destination_table)
                    self.db.execute_sql_to_destination_table(
                        formatted_query, destination=destination_table
                    )
                    self.db.client.extract_table(
                        destination_table,
                        "{gcloud_storage_path}/{features_by_analysis_path}/{analysis}/bin_{bin_left}_{bin_right}/features*.csv".format(
                            **self.config_dict, **time_bin, analysis=analysis
                        ),
                    )
            elif query.config_dict["requires_time_bin_hourly"]:
                for time_bin in self.time_bin_hourly_dict:
                    formatted_query = query.base_query.format_map(
                        {**self.config_dict, **query.config_dict, **time_bin}
                    )
                    destination_table = "{rs_dataset_project}.{features_dataset}.{features_prefix}_{analysis}_bin_hourly_{bin_left}_{bin_right}".format_map(
                        {
                            **self.config_dict,
                            **query.config_dict,
                            **{key: abs(value) for key, value in time_bin.items()},
                            **{"analysis": analysis},
                        }
                    )
                    destination_tables.append(destination_table)
                    self.db.execute_sql_to_destination_table(
                        formatted_query, destination=destination_table
                    )
                    self.db.client.extract_table(
                        destination_table,
                        "{gcloud_storage_path}/{features_by_analysis_path}/{analysis}/bin_hourly_{bin_left}_{bin_right}/features*.csv".format(
                            **self.config_dict, **time_bin, analysis=analysis
                        ),
                    )

            # Not time binned_queries
            else:
                formatted_query = query.base_query.format_map(
                    {**self.config_dict, **query.config_dict}
                )
                destination_table = "{rs_dataset_project}.{features_dataset}.{features_prefix}_{analysis}".format_map(
                    {**self.config_dict, **query.config_dict, **{"analysis": analysis},}
                )
                destination_tables.append(destination_table)
                self.db.execute_sql_to_destination_table(
                    formatted_query, destination=destination_table
                )
                self.db.client.extract_table(
                    destination_table,
                    "{gcloud_storage_path}/{features_by_analysis_path}/{analysis}/static/features*.csv".format(
                        **self.config_dict, analysis=analysis
                    ),
                )
        if merge_features:
            self.merge_features_in_bq(
                tables=destination_tables, binary=self.config_dict["binary"]
            )

    def merge_features_in_bq(
        self, tables, binary=False,
    ):
        """
        Unions several bigquery feature tables into one large result
        """
        base_query = """
            SELECT * FROM (
            {inner_query}
            )
            ORDER BY {row_id_field}
        """

        if binary:
            table_query = """
                SELECT {row_id_field}, person_id, feature_id, CAST(concept_count > 0 AS INT64) as concept_count
                FROM {table}
            """
        else:
            table_query = """
                SELECT {row_id_field}, person_id, feature_id, concept_count
                FROM {table}
            """

        for i, table in enumerate(tables):
            if i == 0:
                inner_query = table_query.format(**self.config_dict, table=table)
            else:
                inner_query = """
                    {inner_query}
                    UNION ALL
                    {table_query}
                """.format(
                    inner_query=inner_query,
                    table_query=table_query.format(**self.config_dict, table=table),
                )
        final_query = base_query.format(inner_query=inner_query, **self.config_dict)

        final_destination_table = "{rs_dataset_project}.{features_dataset}.features_merged".format(
            **self.config_dict
        )
        self.db.execute_sql_to_destination_table(
            final_query, destination=final_destination_table,
        )

        self.db.client.extract_table(
            final_destination_table,
            "{gcloud_storage_path}/{merged_name}/features*.csv".format(
                **self.config_dict
            ),
        )

    def merge_features(
        self,
        merged_name="merged_features",
        create_sparse=False,
        create_parquet=False,
        binary=False,
        load_extension="parquet",
        dask_temp_dir=None,
        existing_vocab_path=None,
        row_id_field="prediction_id",
        **kwargs
    ):
        """
        Merges the features extracted from several analyses on disk
        Args:
            merged_name: the name of the directory to create
            create_sparse: whether to generate a merged scipy.csr_matrix
            create_parquet: whether to generate a merged parquet dataset indexed row_id_field
            binary: whether to save the results as binary (1 if count > 1 else 0) or as the count
            load_extension: the extension of the files to load
            dask_temp_dir: the name of a temporary directory for dask to use
            existing_vocab_path: the path to an existing vocabulary
        """
        if dask_temp_dir is not None:
            dask.config.set({"temporary_directory": dask_temp_dir})
        dask_client = Client(processes=False)
        dask.config.set(scheduler="threads")

        features_path = os.path.join(
            self.config_dict["data_path"], self.config_dict["features_by_analysis_path"]
        )
        merged_path = os.path.join(self.config_dict["data_path"], merged_name)

        overwrite_dir(merged_path, overwrite=True)

        if existing_vocab_path is None:
            vocab_path = os.path.join(merged_path, "vocab")
            overwrite_dir(vocab_path, overwrite=True)
            # Create a vocabulary
            vocab = self.get_vocab(features_path, load_extension=load_extension)
            vocab.to_parquet(os.path.join(vocab_path, "vocab.parquet"))
        else:
            pd.read_parquet(existing_vocab_path, engine="pyarrow")

        ## Read all the data with dask dataframe
        paths = glob.glob(
            os.path.join(
                features_path,
                "**",
                "*.{load_extension}".format(load_extension=load_extension),
            ),
            recursive=True,
        )

        load_columns = [
            row_id_field,
            "person_id",
            "feature_id",
            "concept_count",
        ]
        table_df = dd.concat(
            [self.read_file(path, columns=load_columns) for path in paths],
            interleave_partitions=True,
        ).merge(vocab)

        # If writing sparse data
        if create_sparse:
            sparse_path = os.path.join(merged_path, "features_sparse")
            overwrite_dir(sparse_path, overwrite=True)
            features_row_id_map = (
                table_df[[self.config_dict["row_id_field"]]]
                .drop_duplicates()
                .reset_index(drop=True)
                .reset_index()
                .rename(columns={"index": "features_row_id"})
                .compute()
            )
            features_row_id_map.to_parquet(
                os.path.join(sparse_path, "features_row_id_map.parquet"),
                engine="pyarrow",
            )
            table_df = table_df.merge(features_row_id_map)

        table_df = table_df.set_index(self.config_dict["row_id_field"])
        # print('set_index successful')
        if create_sparse:
            table_df_pd = table_df.compute()
            if binary:
                features = sp.csr_matrix(
                    (
                        np.ones_like(table_df_pd.concept_count, dtype=np.int64),
                        (table_df_pd["features_row_id"], table_df_pd["col_id"]),
                    )
                )
            else:
                features = sp.csr_matrix(
                    (
                        table_df_pd.concept_count,
                        (table_df_pd["features_row_id"], table_df_pd["col_id"]),
                    )
                )
            joblib.dump(features, os.path.join(sparse_path, "features.gz"))

        # If writing parquets
        if create_parquet:
            parquet_path = os.path.join(merged_path, "features_parquet")
            overwrite_dir(parquet_path, overwrite=True)
            if binary:
                table_df = table_df.assign(
                    concept_count=lambda x: (x.concept_count >= 1).astype(np.int64)
                )
            print("Conversion to binary successful")
            table_df.to_parquet(parquet_path, engine="pyarrow")

    @staticmethod
    def read_file(filename, columns=None, **kwargs):
        """
        Reads files in dask, on the basis of the extension
        """
        load_extension = os.path.splitext(filename)[-1]
        if load_extension == ".parquet":
            return dd.read_parquet(filename, columns=columns, **kwargs)
        elif load_extension == ".csv":
            return dd.read_csv(filename, usecols=columns, **kwargs)

    def get_vocab(self, the_path, load_extension="parquet"):
        """
        Constructs a dictionary of unique features from a set of parquets
        """
        paths = glob.glob(
            os.path.join(
                the_path,
                "**",
                "*.{load_extension}".format(load_extension=load_extension),
            ),
            recursive=True,
        )
        vocab = (
            dd.concat(
                [
                    self.read_file(filename, columns=["feature_id"]).drop_duplicates()
                    for filename in paths
                ]
            )
            .drop_duplicates()
            .compute()
            .reset_index(drop=True)
            .rename_axis("col_id")
            .reset_index()
        )
        return vocab
