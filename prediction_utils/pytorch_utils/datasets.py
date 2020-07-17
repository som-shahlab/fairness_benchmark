import pandas as pd
import torch
import scipy as sp
import numpy as np
import dask
import dask.dataframe as dd
import random
import math
from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler
from torch.utils.data.dataloader import default_collate
from dask.distributed import Client


class LoaderGenerator:
    """
    A class that constructs data loaders
    """

    def __init__(self, *args, **kwargs):
        self.config_dict = self.get_default_config()
        self.config_dict = self.override_config(**kwargs)

    def init_loaders(self):
        """
        Returns a dictionary of dataloaders with keys indicating phases
        """
        raise NotImplementedError

    def get_default_config(self):
        """
        Defines the default config_dict
        """
        raise NotImplementedError

    def override_config(self):
        """
        Overrides the config dict with provided kwargs
        """
        raise NotImplementedError


class ArrayLoaderGenerator(LoaderGenerator):
    """
    LoaderGenerator corresponding to ArrayDataset
    """

    def __init__(
        self,
        *args,
        features=None,
        cohort=None,
        fold_id_test="test",
        train_key="train",
        eval_key="val",
        test_key="test",
        row_id_col="row_id",
        **kwargs
    ):
        super().__init__(self, *args, **kwargs)
        self.num_workers = kwargs["num_workers"]
        self.data_dict = self.get_data_dict(
            features=features,
            cohort=cohort,
            fold_id_test=fold_id_test,
            train_key=train_key,
            eval_key=eval_key,
            test_key=test_key,
            row_id_col=row_id_col,
            **kwargs
        )

    def init_datasets(self):
        """
        Creates data loaders from inputs
        """
        convert_sparse = self.config_dict.get("sparse_mode") == "convert"
        phases = self.data_dict["row_id"].keys()
        tensor_dict_dict = {
            key: {
                "features": self.data_dict["features"][key],
                "labels": self.data_dict["labels"][key],
                "row_id": torch.LongTensor(self.data_dict["row_id"][key]),
            }
            for key in phases
        }
        if self.config_dict.get("include_group_in_dataset"):
            for key in phases:
                tensor_dict_dict[key]["group"] = self.data_dict["group"][key]
        dataset_dict = {
            key: ArrayDataset(
                tensor_dict=tensor_dict_dict[key], convert_sparse=convert_sparse,
            )
            for key in phases
        }

        return dataset_dict

    def init_loaders(self, sample_keys=None):
        """
        Method that converts data and labels to instances of class torch.utils.data.DataLoader
            Returns:
                a dictionary with the same keys as data_dict and label_dict.
                    Each element of the dictionary is an instance of torch.utils.data.DataLoader
                        that yields paired elements of data and labels
        """
        # Convert the data to Dataset
        dataset_dict = self.init_datasets()

        # If the Dataset implements collate_fn, that is used. Otherwise, default_collate is used
        if hasattr(dataset_dict["train"], "collate_fn") and callable(
            getattr(dataset_dict["train"], "collate_fn")
        ):
            collate_fn = dataset_dict["train"].collate_fn
        else:
            collate_fn = default_collate

        # If 'iters_per_epoch' is defined, then a fixed number of random sample batches from the training set
        # are drawn per epoch.
        # Otherwise, an epoch is defined by a full run through all of the data in the dataloader.
        if self.config_dict.get("iters_per_epoch") is not None:
            num_samples = (
                self.config_dict["iters_per_epoch"] * self.config_dict["batch_size"]
            )
            if sample_keys is None:
                sample_keys = ["train"]
        else:
            if sample_keys is None:
                sample_keys = []

        loaders_dict = {}
        for key in dataset_dict.keys():
            if key in sample_keys:
                loaders_dict[key] = DataLoader(
                    dataset_dict[key],
                    batch_sampler=BatchSampler(
                        RandomSampler(
                            dataset_dict[key], replacement=True, num_samples=num_samples
                        ),
                        batch_size=self.config_dict["batch_size"],
                        drop_last=False,
                    ),
                    collate_fn=collate_fn,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
            else:
                loaders_dict[key] = DataLoader(
                    dataset_dict[key],
                    batch_size=self.config_dict["batch_size"],
                    collate_fn=collate_fn,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )

        return loaders_dict

    def init_loaders_predict(self, *args):
        """
        Creates data loaders from inputs - for use at prediction time
        """

        # Convert the data to Dataset
        dataset_dict = self.init_datasets()

        # If the Dataset implements collate_fn, that is used. Otherwise, default_collate is used
        if hasattr(dataset_dict["train"], "collate_fn") and callable(
            getattr(dataset_dict["train"], "collate_fn")
        ):
            collate_fn = dataset_dict["train"].collate_fn
        else:
            collate_fn = default_collate

        loaders_dict = {
            key: DataLoader(
                dataset_dict[key],
                batch_size=self.config_dict["batch_size"],
                collate_fn=collate_fn,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            for key in dataset_dict.keys()
        }

        return loaders_dict

    def get_data_dict(
        self,
        features=None,
        cohort=None,
        fold_id_test="test",
        train_key="train",
        eval_key="val",
        test_key="test",
        row_id_col="row_id",
        label_col="outcome",
        sensitive_attribute=None,
        load_features=True,
        **kwargs
    ):
        """
        Generates a data_dict from a features array and a cohort dataframe.
        Args:
            features: The input feature matrix
            cohort: A dataframe with a column called "fold_id" that maps to fold_id
            fold_id: The fold_id corresponding to the validation set
            fold_id_test: The fold_id corresponding to the test set
            train_key: A string that will be used to refer to the training set in the result
            eval_key: A string that will be used to refer to the validation set in the result
            test_key: A string that will be used to refer to the test set in the result
        """

        # Get the validation fold
        fold_id = self.config_dict.get("fold_id")

        if fold_id is None:
            # raise Warning("fold_id not provided")
            fold_id = ""

        fold_id = str(fold_id)
        train_eval_df = cohort.query("fold_id != @fold_id_test")
        # Partition the cohort data into the training phases
        cohort_dict = {
            train_key: train_eval_df.query("fold_id != @fold_id"),
            eval_key: train_eval_df.query("fold_id == @fold_id"),
            test_key: cohort.query("fold_id == @fold_id_test"),
        }

        # # Ensure that each partition is sorted and not empty
        cohort_dict = {
            key: value.sort_values(row_id_col)
            for key, value in cohort_dict.items()
            if value.shape[0] > 0
        }

        # # Initialize the data_dict
        data_dict = {}
        # Save the row_id corresponding to unique predictions
        data_dict["row_id"] = {
            key: value[row_id_col].values for key, value in cohort_dict.items()
        }

        # store the sensitive_attribute
        if sensitive_attribute is not None:
            categories = cohort[sensitive_attribute].sort_values().unique()
            print(categories)
            data_dict["group"] = {
                key: pd.Categorical(
                    value[sensitive_attribute], categories=categories
                ).codes
                for key, value in cohort_dict.items()
            }

        # If features should be loaded
        if load_features:
            data_dict["features"] = {}
            for key in cohort_dict.keys():
                data_dict["features"][key] = features[data_dict["row_id"][key], :]

        data_dict["labels"] = {
            key: np.int64((value[self.config_dict["label_col"]] > 0).values)
            for key, value in cohort_dict.items()
        }

        return data_dict

    def get_default_config(self):
        return {"batch_size": 256, "iters_per_epoch": 100}

    def override_config(self, **override_dict):
        return {**self.config_dict, **override_dict}


class ArrayDataset(Dataset):
    """Dataset wrapping arrays (tensor, numpy, or scipy CSR sparse)

    Each sample will be retrieved by indexing arrays along the first dimension.

    Arguments:
        tensor_dict: a dictionary of array inputs that have the same size in the first dimension
        convert_sparse: whether CSR inputs should be converted to torch.SparseTensor
    """

    def __init__(self, tensor_dict, convert_sparse=False):
        self.convert_sparse = convert_sparse
        self.the_len = list(tensor_dict.values())[0].shape[0]
        assert all(self.the_len == tensor.shape[0] for tensor in tensor_dict.values())
        self.tensor_dict = tensor_dict

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensor_dict.items()}

    def __len__(self):
        return self.the_len

    def collate_fn(self, batch):
        """
        Called by Dataloader to aggregate elements into a batch.
        Delegates to collate_helper for typed aggregation
        Arguments:
            batch: a list of dictionaries with same keys as self.tensor_dict
        """
        result = {}
        keys = batch[0].keys()
        for key in keys:
            result[key] = self.collate_helper(tuple(element[key] for element in batch))
        return result

    def collate_helper(self, batch):
        """
        Aggregates a tuple of elements of the same type
        """
        if isinstance(batch[0], sp.sparse.csr_matrix):
            batch_concat = sp.sparse.vstack(batch)
            if not self.convert_sparse:
                return batch_concat
            else:
                return self.csr_to_tensor(batch_concat)
        else:
            return default_collate(batch)

    def csr_to_tensor(self, x):
        """
        Converts CSR matrix to torch.sparse.Tensor
        """
        x = x.tocoo()
        return torch.sparse.FloatTensor(
            torch.LongTensor([x.row, x.col]),
            torch.FloatTensor(x.data),
            torch.Size(x.shape),
        )

