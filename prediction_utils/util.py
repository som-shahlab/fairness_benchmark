import numpy as np
import pandas as pd
import yaml
import os
import shutil


def yaml_write(x, path):
    """
    Writes yaml to disk
    """
    with open(path, "w") as fp:
        yaml.dump(x, fp)


def yaml_read(path):
    """
    Reads yaml from disk
    """
    with open(path, "r") as fp:
        return yaml.load(fp, Loader=yaml.FullLoader)


def df_dict_concat(df_dict, outer_index_name="outer_index", drop_outer_index=False):
    """
    Concatenate a dictionary of dataframes together and remove the inner index
    """
    if isinstance(outer_index_name, str):
        reset_level = 1
    else:
        reset_level = len(outer_index_name)

    return (
        pd.concat(df_dict, sort=False)
        .reset_index(level=reset_level, drop=True)
        .rename_axis(outer_index_name)
        .reset_index(drop=drop_outer_index)
    )


def overwrite_dir(the_path, overwrite=True):
    """
    Overwrites a directory at a path.
    Will fail if overwrite=False and the_path exists
    """
    if os.path.exists(the_path):
        if not overwrite:
            raise ValueError(
                "Trying to overwrite directory {}, but `overwrite` is False".format(
                    the_path
                )
            )
        shutil.rmtree(the_path)
    os.makedirs(the_path)


def read_file(
    filename, columns=None, load_extension="parquet", mode="pandas", **kwargs
):

    if mode == "pandas":
        if load_extension == "parquet":
            return pd.read_parquet(filename, columns=columns, **kwargs)
        elif load_extension == "csv":
            return pd.read_csv(filename, usecols=columns, **kwargs)
    elif mode == "dask":
        if load_extension == "parquet":
            return pd.read_parquet(filename, columns=columns, **kwargs)
        elif load_extension == "csv":
            return pd.read_csv(filename, usecols=columns, **kwargs)
    else:
        raise ValueError('"pandas" and "dask" are the only allowable modes')


def patient_split(df, patient_col="person_id", test_frac=0.1, nfold=10, seed=657):

    assert (test_frac > 0.0) & (test_frac < 1.0)

    # Get the unique set of patients
    patient_df = df[[patient_col]].drop_duplicates()
    # Shuffle the patients
    patient_df = patient_df.sample(frac=1, random_state=seed)

    # Record the number of samples in each split
    num_test = int(np.floor(test_frac * patient_df.shape[0]))
    num_train = patient_df.shape[0] - num_test

    # Get the number of patients in each fold
    test_patient_df = patient_df.iloc[0:num_test].assign(fold_id="test")

    train_patient_df = patient_df.iloc[num_test:]

    train_patient_df = train_patient_df.assign(
        fold_id=lambda x: np.tile(
            np.arange(1, nfold + 1), int(np.ceil(num_train / nfold))
        )[: x.shape[0]]
    )
    train_patient_df["fold_id"] = train_patient_df["fold_id"].astype(str)
    patient_df = pd.concat([train_patient_df, test_patient_df], ignore_index=True)

    df = df.merge(patient_df)
    return df
