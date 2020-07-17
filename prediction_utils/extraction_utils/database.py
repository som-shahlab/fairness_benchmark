import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import shutil
import sqlalchemy as sq

import google.auth
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1

from prediction_utils.util import overwrite_dir, yaml_read


class BQDatabase:
    """
    A class defining a BigQuery Database
    """

    def __init__(self, **kwargs):

        self.config_dict = self.override_defaults(**kwargs)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.config_dict[
            "google_application_credentials"
        ]
        os.environ["GCLOUD_PROJECT"] = self.config_dict["gcloud_project"]

        # https://cloud.google.com/bigquery/docs/bigquery-storage-python-pandas
        credentials, your_project_id = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

        self.client = bigquery.Client(credentials=credentials, project=your_project_id)
        self.bqstorageclient = bigquery_storage_v1beta1.BigQueryStorageClient(
            credentials=credentials
        )

    def get_defaults(self):
        """
        Defaults for values in the config_dict
        """
        return {
            "gcloud_project": "som-nero-phi-nigam-starr",
            "google_application_credentials": os.path.expanduser(
                "~/.config/gcloud/application_default_credentials.json"
            ),
        }

    def override_defaults(self, **kwargs):
        return {**self.get_defaults(), **kwargs}

    def read_sql_query(
        self,
        query,
        dialect="standard",
        use_bqstorage_api=True,
        progress_bar_type=None,
        **kwargs
    ):
        """
        Read a sql query directly into a pandas DataFrame.
        Uses default project defined at instantiation.
        Args:
            query: A SQL query as a string
            dialect: BigQuery dialect to use. Default "standard"
            use_bq_storage_api: Whether to use the BigQuery Storage API
        """
        df = pd.read_gbq(
            query,
            project_id=self.config_dict["gcloud_project"],
            dialect=dialect,
            use_bqstorage_api=use_bqstorage_api,
            progress_bar_type=progress_bar_type,
            **kwargs
        )
        return df

    def stream_query(self, query, output_path, overwrite=False, combine_every=1000):
        """
        Streams a query to pandas dataframes in chunks using the Storage API.
        Results will be written as parquet files of size 1024*`combine_every` rows
        query: SQL query to execute
        output_path: a directory to write the result
        overwrite: Whether to overwrite output_path
        combine_every: The number of chunks to combine before writing a file
        """
        result = (
            self.client.query(query)
            .result(
                page_size=1024
            )  # page_size doesn't seem to do anything if using bqstorage_client?
            .to_dataframe_iterable(bqstorage_client=self.bqstorageclient)
        )
        result_dict = {}
        for i, rows in enumerate(result):
            if i == 0:
                overwrite_dir(output_path, overwrite=overwrite)
            result_dict[i] = rows
            if (i % combine_every == 0) & (i > 0):
                result_df = pd.concat(result_dict, ignore_index=True)
                result_df.to_parquet(
                    os.path.join(output_path, "features_{i}.parquet".format(i=i)),
                    engine="pyarrow",
                )
                result_dict = {}
        if len(list(result_dict.keys())) > 0:
            result_df = pd.concat(result_dict, ignore_index=True)
            result_df.to_parquet(
                os.path.join(output_path, "features_{i}.parquet".format(i=i)),
                engine="pyarrow",
            )

    def to_sql(self, *args, mode="gbq", **kwargs):
        """
        Writes results to a table, using either pandas-gbq or the google client library
        """
        if mode == "gbq":
            return self.to_sql_gbq(*args, **kwargs)
        elif mode == "client":
            return self.to_sql_client(*args, **kwargs)
        else:
            raise ValueError("Mode must be gbq or client")

    def to_sql_gbq(
        self, df, destination_table, chunksize=10000, if_exists="replace", **kwargs
    ):
        """
        Uses the pandas.to_gbq method to write the destination table to the dataframe.
        Caveats: 
            Does not support DATE
            Serializes to CSV
        """
        df.to_gbq(
            destination_table=destination_table,
            chunksize=chunksize,
            if_exists=if_exists,
            **kwargs
        )

    def to_sql_client(
        self,
        df,
        destination_table,
        date_cols=None,
        write_disposition="WRITE_TRUNCATE",
        schema=None,
    ):
        """
        Uses the the BigQuery client library to write a table to BQ.
        As of now, this method should be used to write tables with DATE columns.
        Allows serializing data with pyarrow
        (TODO): better manage alternate schemas
        Example: https://googleapis.dev/python/bigquery/latest/usage/pandas.html
        """
        if (date_cols is not None) and (schema is not None):
            schema = [
                bigquery.SchemaField(x, bigquery.enums.SqlTypeNames.DATE)
                for x in date_cols
            ]
        job_config = bigquery.LoadJobConfig(
            schema=schema, write_disposition=write_disposition
        )
        job = self.client.load_table_from_dataframe(
            df, destination_table, job_config=job_config
        )
        job.result()
        table = self.client.get_table(destination_table)  # Make an API request.
        print(
            "Loaded {} rows and {} columns to {}".format(
                table.num_rows, len(table.schema), destination_table
            )
        )

    def execute_sql(self, query):
        """
        Executes sql statement
        """
        return self.client.query(query).result()

    def execute_sql_to_destination_table(self, query, destination=None, **kwargs):
        """
        Executes a query and writes the result to a destination table
        """
        if destination is None:
            raise ValueError("destination must not be None")

        self.client.query(
            query,
            job_config=bigquery.QueryJobConfig(
                destination=destination, write_disposition="WRITE_TRUNCATE"
            ),
        ).result()


class Database:
    """
    A class defining a DBAPI database that can be connected to with SqlAlchemy.
    Support for Postgres and MySQL currently
    """

    def __init__(self, database=None, uri=None, config_path=None, dbms="postgresql"):

        # If uri is provided - overrides everything else
        if uri is not None:
            self.uri = uri
        else:
            if config_path is None:
                config_path = self.get_default_config(dbms)
            config_dict = self.parse_config(config_path)
            if database is not None:
                config_dict["database"] = database
            self.uri = self.get_db_uri(dbms=dbms, **config_dict)
        self.engine = self.init_engine(self.uri, dbms)
        self.dbapi_conn, self.cursor = self.init_raw_connection(self.engine)

    def get_default_config(self, dbms):
        if dbms == "mysql":
            return "~/.my.cnf"
        elif dbms == "postgresql":
            return "~/.pg.cnf"
        else:
            raise Exception("Default config not defined for this dbms")

    @staticmethod
    def init_engine(uri, dbms="postgresql"):
        """
        Initialize an engine for connecting
        """
        if dbms == "mysql":
            engine = sq.create_engine(uri, server_side_cursors=True)
        elif dbms == "postgresql":
            engine = sq.create_engine(
                uri, server_side_cursors=True, use_batch_mode=True
            )
        else:
            raise Exception("dbms not supported")
        return engine

    @staticmethod
    def init_raw_connection(engine):
        """
        Gives a raw connection to the database engine, rather than the wrapped SQLAlchemy version
        """
        dbapi_conn = engine.raw_connection()
        cursor = dbapi_conn.cursor()
        return dbapi_conn, cursor

    @staticmethod
    def get_db_uri(
        user=None, password=None, host=None, port=None, database=None, dbms="postgresql"
    ):
        """
        Gets a sqlalchemy uri for databases.
        """
        prefix_dict = {"mysql": "mysql+pymysql", "postgresql": "postgresql"}
        prefix = prefix_dict.get(dbms)
        if prefix is None:
            raise NotImplementedError("Only mysql and postgresql supported")

        uri = "{prefix}://{user}:{password}@{host}:{port}/{database}".format(
            prefix=prefix,
            user=user,
            password=password,
            host=host,
            port=port,
            database=database,
        )
        return uri

    @staticmethod
    def parse_config(config_path="~/.pg.cnf"):
        """
        Parses database config file
        """
        _, ext = os.path.splitext(config_path)
        if ext == ".cnf":
            with open(os.path.expanduser(config_path), "r") as f:
                config = f.readlines()
            config_dict = [x.strip().split(" = ") for x in config]
            config_dict = [x for x in config_dict if len(x) > 1]
            config_dict = {x[0]: x[1] for x in config_dict}
        elif ext == ".yaml":
            config_dict = yaml_read(os.path.expanduser(config_path))
        else:
            raise ValueError("Only .cnf and .yaml database config files supported")

        return config_dict

    def stream_query(
        self, query, write_path, chunksize=int(1e6), overwrite=False, parse_dates=None
    ):
        """
        Streams a query from a database
        Parameters:
        query: A SQL string
        write_path: a path to write the query outputs
        chunksize: The number of rows to stream at once. Each chunk will be saved in its own file
        overwrite: If this true (default) the write_path directory will be overwritten, otherwise files will be appended.
        """

        # If directory already exists and overwrite option is specified
        if overwrite and os.path.isdir(write_path):
            shutil.rmtree(write_path)

        df_stream = pd.read_sql_query(
            sql=query, con=self.engine, chunksize=chunksize, parse_dates=parse_dates
        )

        for temp in df_stream:
            temp_pq = pa.Table.from_pandas(temp)
            pq.write_to_dataset(
                table=temp_pq, root_path=write_path, preserve_index=False
            )

    def read_sql_query(self, sql, **kwargs):
        return pd.read_sql_query(sql, con=self.engine, **kwargs)

    def execute_sql(self, sql):
        self.cursor.execute(sql)
        self.dbapi_conn.commit()

    def to_sql(self, df, **kwargs):
        df.to_sql(**kwargs)
