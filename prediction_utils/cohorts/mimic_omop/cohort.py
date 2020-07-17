from prediction_utils.cohorts.cohort import BQCohort


class MIMICICUCohort(BQCohort):

    def get_defaults(self):
        return {
            **super().get_defaults(),
            **{
                "relative_hours_index": 24,
                "min_los_icu_hours": 12,
                "max_los_icu_hours": 240,
                "min_age_in_years": 15.0,
                "min_hours_until_icu_disch": 6.0,
            },
        }
    def get_base_query(self, format_query=True):
        return """
            WITH initial_cohort AS (
                SELECT t1.*, 
                    ROW_NUMBER() OVER (PARTITION BY t1.person_id ORDER BY t1.visit_start_datetime) as row_id,
                    DATETIME_ADD(t2.visit_start_datetime, INTERVAL {relative_hours_index} HOUR) as index_datetime,
                    DATETIME_DIFF(t1.visit_end_datetime, t1.visit_start_datetime, HOUR) as los_icu_hours,
                    DATETIME_DIFF(t2.visit_end_datetime, t2.visit_start_datetime, HOUR) as los_hospital_hours,
                    CAST(t1.discharge_to_concept_id = 4216643 AS INT64) AS mortality_icu,
                    CAST(t2.discharge_to_concept_id = 4216643 AS INT64) AS mortality_hospital,
                    t1.visit_end_datetime as icu_disch_time
                    FROM {dataset_project}.{dataset}.visit_detail t1
                    LEFT JOIN {dataset_project}.{dataset}.visit_occurrence as t2 USING (visit_occurrence_id)
                    WHERE visit_detail_concept_id = 32037
            ),
            transformed_cohort AS (
                SELECT t1.*,
                    CAST(DATETIME_DIFF(index_datetime, birth_datetime, DAY) AS FLOAT64) / 365.25 as age_in_years,
                    CAST(los_icu_hours > 3*24 AS INT64) as los_icu_3days,
                    CAST(los_icu_hours > 7*24 AS INT64) as los_icu_7days,
                    CAST(DATETIME_DIFF(icu_disch_time, index_datetime, HOUR) AS FLOAT64) AS hours_until_icu_disch
                FROM initial_cohort t1
                INNER JOIN {dataset_project}.{dataset}.person as t2 USING (person_id)
            )
            SELECT person_id, visit_occurrence_id, visit_detail_id, 
            index_datetime, age_in_years,
                los_hospital_hours, los_icu_hours,  
                los_icu_3days, los_icu_7days, 
                mortality_hospital, mortality_icu
            FROM transformed_cohort
            WHERE 
                row_id = 1
                AND age_in_years BETWEEN {min_age_in_years} AND 89.0 -- MIMIC sets 90+ patients to 300 years old
                AND los_icu_hours BETWEEN {min_los_icu_hours} AND {max_los_icu_hours}
                AND hours_until_icu_disch >= {min_hours_until_icu_disch}
        """.format_map(
            self.config_dict
        )
        if not format_query:
            return query
        else:
            return query.format_map(self.config_dict)

    def get_transform_query_sampled(self):
        return """
            SELECT * EXCEPT (rnd, pos), 
            FARM_FINGERPRINT(GENERATE_UUID()) as prediction_id
            FROM (
                SELECT *, ROW_NUMBER() OVER(PARTITION BY person_id ORDER BY rnd) AS pos
                FROM (
                    SELECT 
                        *,
                        FARM_FINGERPRINT(CONCAT(CAST(person_id AS STRING), CAST(visit_occurrence_id AS STRING))) as rnd
                    FROM ({base_query})
                )
            )
            WHERE pos = 1
        """.format(
            base_query=self.get_base_query()
        )


class MIMICDemographicsCohort(BQCohort):
    """
    Cohort that defines the demographic variables
    (TODO) harmonize this with common definitions from the admissions cohort
    """

    def get_base_query(self, format_query=True):

        query = "{rs_dataset_project}.{rs_dataset}.{cohort_name}"
        if not format_query:
            return query
        else:
            return query.format_map(self.config_dict)

    def get_transform_query(self, format_query=True):
        query = """
            WITH age_labels AS (
                {age_query}
            ),
            demographics AS (
                {demographics_query}
            )
            SELECT *
            FROM {base_query}
            LEFT JOIN age_labels USING (person_id, age_in_years)
            LEFT JOIN demographics USING (person_id)
        """

        if not format_query:
            return query
        else:
            return query.format(
                base_query=self.get_base_query(), **self.get_label_query_dict()
            )

    def get_label_query_dict(self):
        query_dict = {
            "age_query": self.get_age_query(),
            "demographics_query": self.get_demographics_query(),
        }
        base_query = self.get_base_query()
        return {
            key: value.format_map({**{"base_query": base_query}, **self.config_dict})
            for key, value in query_dict.items()
        }

    def get_age_query(self):
        return """
            WITH temp AS (
                SELECT t1.person_id, t1.age_in_years,
                FROM {base_query} t1
                INNER JOIN {dataset_project}.{dataset}.person t2
                ON t1.person_id = t2.person_id
            )
            SELECT *,
            CASE 
                WHEN age_in_years >= 15.0 and age_in_years < 30.0 THEN '[15-30)'
                WHEN age_in_years >= 30.0 and age_in_years < 45.0 THEN '[30-45)'
                WHEN age_in_years >= 45.0 and age_in_years < 55.0 THEN '[45-55)'
                WHEN age_in_years >= 55.0 and age_in_years < 65.0 THEN '[55-65)'
                WHEN age_in_years >= 65.0 and age_in_years < 75.0 THEN '[65-75)'
                WHEN age_in_years >= 75.0 and age_in_years < 91.0 THEN '[75-91)'
                ELSE 'other'
            END as age_group
            FROM temp
        """

    def get_demographics_query(self):
        return """
            WITH source_concepts AS (
                SELECT person_id, race_concept_id as concept_id 
                FROM {base_query}
                INNER JOIN {dataset_project}.{dataset}.person USING (person_id)
            ),
            concept_ancestors AS (
                SELECT person_id, COALESCE(ancestor_concept_id, concept_id) as concept_id
                FROM source_concepts t1
                LEFT JOIN {dataset_project}.{dataset}.concept_ancestor as t2
                    ON t1.concept_id = t2.descendant_concept_id
            ),
            race_eth_rollup AS (
                SELECT t1.* EXCEPT(concept_id), concept_id as race_eth_concept_id, concept_name as race_eth_concept_name
                FROM concept_ancestors t1
                INNER JOIN {dataset_project}.{dataset}.concept USING (concept_id)
                --WHERE concept_id in (8527, 8515, 8516, 8557, 8657, 86571, 4188159)
                -- WHERE concept_id in (8527, 8515, 8516, 86571, 4188159)
                WHERE concept_id in (8527)
            ),
            demographics_result AS (
                SELECT 
                    t1.*, 
                    COALESCE(race_eth_concept_id, 0) as race_eth_concept_id,
                    COALESCE(race_eth_concept_name, 'Other') as race_eth_concept_name,
                    t3.race_concept_id, t3.race_source_value, t4.concept_name as gender_concept_name
                FROM {rs_dataset_project}.{rs_dataset}.{cohort_name} t1
                LEFT JOIN race_eth_rollup as t2 USING (person_id)
                LEFT JOIN {dataset_project}.{dataset}.person as t3 USING (person_id)
                INNER JOIN {dataset_project}.{dataset}.concept as t4
                    ON t3.gender_concept_id = t4.concept_id
            )
            SELECT person_id, race_eth_concept_name as race_eth, gender_concept_name 
            FROM demographics_result
        """
