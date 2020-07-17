import os
from prediction_utils.extraction_utils.database import BQDatabase


class BQCohort:
    def __init__(self, *args, **kwargs):
        self.config_dict = self.get_config_dict(**kwargs)
        self.db = BQDatabase(**self.config_dict)

    def get_base_query(self):
        """
        A reference to the data prior to transformation
        """
        raise NotImplementedError

    def get_transform_query(self):
        """
        A query that transforms the base query
        """
        raise NotImplementedError

    def get_create_query(self):
        """
        Constructs a create table query
        """
        raise NotImplementedError

    def create_cohort_table(self):
        """
        Creates the cohort table in the database
        """
        self.db.execute_sql(self.get_create_query())

    def get_defaults(self):
        return {
            "gcloud_project": "som-nero-phi-nigam-starr",
            "dataset_project": None,
            "rs_dataset_project": None,
            "dataset": "starr_omop_cdm5_deid_20200404",
            "rs_dataset": "temp_dataset",
            "cohort_name": "temp_cohort",
            "google_application_credentials": os.path.expanduser(
                "~/.config/gcloud/application_default_credentials.json"
            ),
            "limit": None,
        }

    def override_defaults(self, **kwargs):
        return {**self.get_defaults(), **kwargs}

    def get_config_dict(self, **kwargs):
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
