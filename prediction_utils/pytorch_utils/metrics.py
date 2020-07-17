import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import warnings
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    recall_score,
    precision_score,
)
import scipy
from sklearn.linear_model import LogisticRegression
from prediction_utils.util import df_dict_concat


class MetricComparator:
    def __init__(self, metric_type="min"):
        self.metric_type = metric_type

    def is_better(self, value, other):
        if self.metric_type == "min":
            return value < other
        elif self.metric_type == "max":
            return value > other

    def get_inital_value(self):
        if self.metric_type == "min":
            return 1e18
        elif self.metric_type == "max":
            return -1e18


class MetricDict:
    """
    Accumulates outputs and computes metrics
    """

    def __init__(self, metrics=None, init_value=0.0):

        if metrics is None:
            metrics = ["auc", "auprc", "brier", "loss_bce"]

        self.metric_fns = self.get_metric_fns(metrics)
        self.init_metric_dict(metrics=metrics, init_value=0.0)
        self.init_output_dict()

    def init_metric_dict(self, metrics=None, init_value=None):
        """
        Initialize a dict of metrics
        """
        if metrics is None:
            metrics = [""]

        if init_value is None:
            init_value = 0.0

        self.metric_dict = {metric: init_value for metric in metrics}

    def get_metric_fns(self, metrics=None):
        """
        Sklearn metric functions that operate on labels and pred_probs
        """
        metric_fn_dict = {
            "auc": lambda labels, pred_probs: 0.0
            if (labels.sum() == len(labels)) or (labels.sum() == 0)
            else roc_auc_score(labels, pred_probs),
            "auprc": average_precision_score,
            "brier": brier_score_loss,
            "loss_bce": log_loss,
        }
        if metrics is None:
            return metric_fn_dict
        else:
            return {
                key: value for key, value in metric_fn_dict.items() if key in metrics
            }

    def print_metrics(self):
        """
        Print method
        """
        print(
            "".join([" {}: {:4f},".format(k, v) for k, v in self.metric_dict.items()])
        )

    def compute_metrics(self):
        """
        Compute epoch statistics after an epoch of training using an output_dict.
        """
        self.finalize_output_dict()
        self.metric_dict = {
            key: value(self.output_dict["labels"], self.output_dict["pred_probs"])
            for key, value in self.metric_fns.items()
        }

    def init_output_dict(self, keys=None):
        if keys is None:
            keys = ["outputs", "pred_probs", "labels", "row_id"]

        self.output_dict = {key: [] for key in keys}

    def update_output_dict(self, **kwargs):
        kwargs["pred_probs"] = F.softmax(kwargs["outputs"], dim=1)[:, 1]
        for key, value in kwargs.items():
            self.output_dict[key].append(value.detach().cpu())

    def finalize_output_dict(self):
        """
        Convert an output_dict to numpy
        """
        self.output_dict = {
            key: torch.cat(value).numpy() for key, value in self.output_dict.items()
        }


class LossDict(MetricDict):
    # (TODO, rewrite this so it is a sibling of MetricDict)

    def __init__(self, metrics=["loss"], init_value=0.0, mode="mean"):
        super().__init__(metrics=metrics, init_value=init_value)
        self.running_batch_size = 0
        self.mode = mode

    def update_loss_dict(self, update_dict, batch_size=None):
        if self.mode == "mean":
            self.running_batch_size += batch_size
        for key in self.metric_dict.keys():
            if self.mode == "mean":
                self.metric_dict[key] += update_dict[key].item() * batch_size
            else:
                self.metric_dict[key] += update_dict[key].item()

    def compute_metrics(self):
        if self.mode == "mean":
            for key in self.metric_dict.keys():
                self.metric_dict[key] = self.metric_dict[key] / float(
                    self.running_batch_size
                )


class MetricLogger:
    def __init__(self, metrics=None, losses=None, phases=None):
        if metrics is None:
            metrics = ["auc", "auprc", "brier", "loss_bce"]
        if losses is None:
            losses = ["loss"]
        if phases is None:
            phases = ["train", "val"]

        self.metrics = metrics
        self.losses = losses
        self.phases = phases

        self.metric_dict_overall = self.init_metric_dict_overall(metric_names=metrics)
        self.loss_dict_overall = self.init_metric_dict_overall(metric_names=losses)
        self.init_metric_dicts()

    def init_metric_dict_overall(self, metric_names):
        metric_dict = {
            phase: {metric: [] for metric in metric_names} for phase in self.phases
        }
        return metric_dict

    def compute_metrics_epoch(self, phase=None):
        if phase is None:
            raise ValueError("Must provide phase to compute_metrics_epoch")

        self.metric_dict.compute_metrics()
        self.loss_dict.compute_metrics()

        self.loss_dict_overall = self.update_metric_dict_overall(
            self.loss_dict_overall, self.loss_dict.metric_dict, phase=phase
        )
        self.metric_dict_overall = self.update_metric_dict_overall(
            self.metric_dict_overall, self.metric_dict.metric_dict, phase=phase
        )

    def get_output_dict(self):
        return self.metric_dict.output_dict

    def get_output_df(self):
        return self.process_output_dict(self.metric_dict.output_dict)

    def get_metrics_overall(self):
        return self.process_metrics_overall(
            {
                phase: {
                    **self.metric_dict_overall[phase],
                    **self.loss_dict_overall[phase],
                }
                for phase in self.phases
            }
        )

    def process_metrics_overall(
        self, the_dict, names=["metric", "phase", "epoch", "performance"]
    ):
        """
        Converts the overall metrics to a dataframe
        """
        result = (
            pd.DataFrame(the_dict)
            .reset_index()
            .melt(id_vars="index")
            .set_index(["index", "variable"])
            .value.apply(pd.Series)
            .stack()
            .reset_index()
        )
        result.columns = names
        return result

    def process_output_dict(self, output_dict, metadata_dict=None):
        return (
            pd.concat(
                {
                    phase: pd.DataFrame(
                        {
                            "outputs": output_dict[phase]["outputs"][:, 1],
                            "pred_probs": output_dict[phase]["pred_probs"],
                            "labels": output_dict[phase]["labels"],
                            "row_id": output_dict[phase]["row_id"],
                        }
                        if metadata_dict is None
                        else {
                            **{
                                "outputs": output_dict[phase]["outputs"][:, 1],
                                "pred_probs": output_dict[phase]["pred_probs"],
                                "labels": output_dict[phase]["labels"],
                            },
                            **{
                                key: metadata_dict[phase][key]
                                for key in metadata_dict[phase].keys()
                            },
                        }
                    )
                    for phase in output_dict.keys()
                }
            )
            .rename_axis(["phase", "index"])
            .reset_index(0)
            .reset_index(drop=True)
        )

    def update_metric_dict_overall(self, metric_dict, update_dict, phase):
        """
        Updates a metric dict with metrics from an epoch
        """
        for key in update_dict.keys():
            metric_dict[phase][key].append(update_dict[key])
        return metric_dict

    def init_metric_dicts(self):
        self.metric_dict = MetricDict(metrics=self.metrics)
        self.loss_dict = LossDict(metrics=self.losses)

    def update_loss_dict(self, *args, **kwargs):
        self.loss_dict.update_loss_dict(*args, **kwargs)

    def update_output_dict(self, *args, **kwargs):
        self.metric_dict.update_output_dict(*args, **kwargs)

    def get_metrics_epoch(self):
        return {**self.metric_dict.metric_dict, **self.loss_dict.metric_dict}

    def print_metrics(self):
        self.metric_dict.print_metrics()
        self.loss_dict.print_metrics()


class StandardEvaluator:
    def __init__(self, metrics=None, thresholds=None):
        # default behavior: use all metrics, do not use any threshold metrics
        self.metric_fns = self.get_metric_fns(metrics=metrics, thresholds=thresholds)

    def get_result_df(self, df, group_vars=["task", "attribute", "phase"]):
        group_vars = [var for var in group_vars if var in df.columns]
        result_df_by_group = self.evaluate_by_group(
            df, group_vars + ["group"], "performance"
        )
        result_df_by_group = result_df_by_group.query(
            '(not performance.isnull()) & (not (metric == "auc" & (performance < 0.0))) & (not (metric == "loss_bce" & (performance == 1e18)))'
        )
        result_df_overall = self.evaluate_by_group(
            df, group_vars, "performance_overall"
        )
        result_df = result_df_by_group.merge(result_df_overall)
        return result_df

    def evaluate_by_group(self, df, group_vars=None, result_name="performance"):
        if self.metric_fns is None:
            raise ValueError("Please provide metric_fn_dict")
        if group_vars is None:
            group_vars = ["task", "attribute", "phase", "group"]

        group_vars = [var for var in group_vars if var in df.columns]

        result_df = df_dict_concat(
            {
                metric: df.groupby(group_vars)
                .apply(lambda x: metric_func(x.labels.values, x.pred_probs.values))
                .rename(index=result_name)
                .rename_axis(group_vars)
                .reset_index()
                for metric, metric_func in self.metric_fns.items()
            },
            "metric",
        )
        return result_df

    def get_metric_fns(self, metrics=None, thresholds=None):
        threshold_free_metrics = self.get_threshold_free_metrics(metrics=metrics)
        threshold_metrics = self.get_threshold_metrics(thresholds=thresholds)
        return {**threshold_free_metrics, **threshold_metrics}

    def get_threshold_free_metrics(self, metrics=None):
        base_metric_dict = {
            "auc": lambda labels, pred_probs: -1
            if (labels.sum() == len(labels)) or (labels.sum() == 0)
            else roc_auc_score(labels, pred_probs),
            "auprc": average_precision_score,
            "brier": brier_score_loss,
            "loss_bce": lambda labels, pred_probs: 1e18
            if (labels.sum() == len(labels)) or (labels.sum() == 0)
            else log_loss(labels, pred_probs),
        }
        if metrics is None:
            return base_metric_dict
        else:
            return {
                key: base_metric_dict[key]
                for key in metrics
                if key in base_metric_dict.keys()
            }

    def get_threshold_metrics(self, thresholds=[0.01, 0.05, 0.1, 0.2]):
        if thresholds is None:
            return {}
        return {
            **{
                "recall_{}".format(threshold): self.recall_at_threshold(threshold)
                for threshold in thresholds
            },
            **{
                "precision_{}".format(threshold): self.precision_at_threshold(threshold)
                for threshold in thresholds
            },
            **{
                "specificity_{}".format(threshold): self.specificity_at_threshold(
                    threshold
                )
                for threshold in thresholds
            },
        }

    @staticmethod
    def recall_at_threshold(threshold):
        return lambda x, y: recall_score(x, 1.0 * (y >= threshold))

    @staticmethod
    def precision_at_threshold(threshold):
        return lambda x, y: precision_score(x, 1.0 * (y >= threshold), zero_division=0)

    @staticmethod
    def specificity_at_threshold(threshold):
        return (
            lambda labels, pred_probs: (
                (labels == 0) & (labels == (pred_probs >= threshold))
            ).sum()
            / (labels == 0).sum()
            if (labels == 0).sum() > 0
            else 0.0
        )


class FairOVAEvaluator:
    def __init__(self, metrics=None):
        self.metric_fns = self.get_metric_fns(metrics=metrics)

    def get_result_df(self, df, group_vars=["attribute", "task", "phase"]):
        group_vars = [var for var in group_vars if var in df.columns]
        grouped = df.groupby(group_vars)
        result = {}
        for (meta, df_group) in grouped:
            groups = df_group.group.unique()
            for group in groups:
                result[meta + tuple([group])] = pd.DataFrame(
                    {
                        key: value(df_group, group)
                        for key, value in self.metric_fns.items()
                    }
                )

        result_df = (
            pd.concat(result)
            .reset_index(level=len(group_vars) + 1, drop=True)
            .rename_axis(group_vars + ["group"])
            .reset_index()
        )
        id_vars = group_vars + ["group"]
        result_df_long = result_df.melt(
            id_vars=id_vars,
            value_vars=set(result_df.columns) - set(id_vars),
            var_name="metric",
            value_name="performance",
        )

        result_df_long = result_df_long.query(
            '(not performance.isnull()) & (not (metric in ["auc", "xauc_0", "xauc_1", "xauc_ova_1", "xauc_ova_0"] & (performance < 0.0)))'
        )
        return result_df_long

    @staticmethod
    def try_wasserstein_distance(*args, **kwargs):
        try:
            return scipy.stats.wasserstein_distance(*args, **kwargs)
        except ValueError:
            warnings.warn("Error in EMD, filling with -1")
            return -1.0  # flag value to filter out at the point of aggregation

    @staticmethod
    def try_roc_auc_score(*args, **kwargs):
        try:
            return roc_auc_score(*args, **kwargs)
        except ValueError:
            warnings.warn("Error in AUC, filling with -1")
            return -1.0

    def get_metric_fns(self, metrics=None):
        base_metric_dict = {
            "emd_ova": lambda x, group: pd.Series(
                self.try_wasserstein_distance(
                    x.query("group == @group").pred_probs, x.pred_probs,
                )
            ),
            "emd_ova_1": lambda x, group: pd.Series(
                self.try_wasserstein_distance(
                    x.query("group == @group & labels == 1").pred_probs,
                    x.query("labels == 1").pred_probs,
                )
            ),
            "emd_ova_0": lambda x, group: pd.Series(
                self.try_wasserstein_distance(
                    x.query("group == @group & labels == 0").pred_probs,
                    x.query("labels == 0").pred_probs,
                )
            ),
            "mean_prediction": lambda x, group: pd.Series(
                (x.query("group == @group").pred_probs).mean() - (x.pred_probs).mean()
            ),
            "mean_prediction_1": lambda x, group: pd.Series(
                (x.query("group == @group & labels == 1").pred_probs).mean()
                - (x.query("labels == 1").pred_probs).mean()
            ),
            "mean_prediction_0": lambda x, group: pd.Series(
                (x.query("group == @group & labels == 0").pred_probs).mean()
                - (x.query("labels == 0").pred_probs).mean()
            ),
            "xauc_ova_1": lambda x, group: pd.Series(
                self.try_roc_auc_score(
                    pd.concat(
                        [
                            x.query("group == @group & labels == 1").labels,
                            x.query("labels == 0").labels,
                        ]
                    ),
                    pd.concat(
                        [
                            x.query("group == @group & labels == 1").pred_probs,
                            x.query("labels == 0").pred_probs,
                        ]
                    ),
                )
            ),
            "xauc_ova_0": lambda x, group: pd.Series(
                self.try_roc_auc_score(
                    pd.concat(
                        [
                            x.query("group == @group & labels == 0").labels,
                            x.query("labels == 1").labels,
                        ]
                    ),
                    pd.concat(
                        [
                            x.query("group == @group & labels == 0").pred_probs,
                            x.query("labels == 1").pred_probs,
                        ]
                    ),
                )
            ),
            "xauc_1": lambda x, group: pd.Series(
                self.try_roc_auc_score(
                    pd.concat(
                        [
                            x.query("group == @group & labels == 1").labels,
                            x.query("labels == 0 & group != @group").labels,
                        ]
                    ),
                    pd.concat(
                        [
                            x.query("group == @group & labels == 1").pred_probs,
                            x.query("labels == 0 & group != @group").pred_probs,
                        ]
                    ),
                )
            ),
            "xauc_0": lambda x, group: pd.Series(
                self.try_roc_auc_score(
                    pd.concat(
                        [
                            x.query("group == @group & labels == 0").labels,
                            x.query("labels == 1 & group != @group").labels,
                        ]
                    ),
                    pd.concat(
                        [
                            x.query("group == @group & labels == 0").pred_probs,
                            x.query("labels == 1 & group != @group").pred_probs,
                        ]
                    ),
                )
            ),
        }
        if metrics is None:
            return base_metric_dict
        else:
            return {
                key: base_metric_dict[key]
                for key in metrics
                if key in base_metric_dict.keys()
            }


class CalibrationEvaluator:
    def get_calibration_df(
        self, df, group_vars=["config_filename", "phase", "task", "attribute", "group"],
    ):

        group_vars = [var for var in group_vars if var in df.columns]
        model_dict = {}
        calibration_dict = {}

        for group, df_grouped in df.groupby(group_vars):
            if df_grouped.labels.sum() == 0:
                continue
            print(group)
            df_grouped = df_grouped.query("pred_probs > 0")
            log_probs = np.log(df_grouped.pred_probs.values.reshape(-1, 1))
            model_dict[group] = LogisticRegression(
                solver="lbfgs", penalty="none", max_iter=1000
            )
            model_dict[group].fit(log_probs, df_grouped.labels.values)
            calibration_dict[group] = df_grouped.assign(
                density_conditional_y1=model_dict[group].predict_proba(log_probs)[:, -1]
            ).reset_index(drop=True)
        calibration_df = pd.concat(calibration_dict).reset_index(drop=True)

        return calibration_df

    def get_calibration_df_combined(
        self,
        df,
        group_vars=[
            "sensitive_attribute",
            "config_filename",
            "phase",
            "task",
            "attribute",
        ],
    ):
        group_vars = [var for var in group_vars if var in df.columns]
        df = df[group_vars + ["pred_probs", "labels", "group"]]

        calibration_df_group = self.get_calibration_df(
            df, group_vars=group_vars + ["group"]
        )
        calibration_df_overall = self.get_calibration_df(df, group_vars=group_vars)

        # Joining on a floating point value is very slow
        sort_columns = list(
            set(calibration_df_group.columns) - set(["density_conditional_y1"])
        )

        calibration_df_group = calibration_df_group.sort_values(
            sort_columns
        ).reset_index(drop=True)
        calibration_df_overall = calibration_df_overall.sort_values(
            sort_columns
        ).reset_index(drop=True)
        calibration_df = calibration_df_group
        calibration_df["density_conditional_y1_overall"] = calibration_df_overall[
            "density_conditional_y1"
        ]

        return calibration_df

    def get_calibration_result(
        self, df, group_vars=["config_filename", "phase", "task", "attribute", "group"],
    ):
        group_vars = [var for var in group_vars if var in df.columns]
        return (
            df.assign(
                brier_diff_signed=lambda x: x.labels - x.pred_probs,
                brier_diff_squared=lambda x: x.brier_diff_signed ** 2,
                calib_diff_signed=lambda x: x.density_conditional_y1 - x.pred_probs,
                calib_diff_squared=lambda x: x.calib_diff_signed ** 2,
                calib_density_diff_signed=lambda x: x.density_conditional_y1
                - x.density_conditional_y1_overall,
                calib_density_diff_squared=lambda x: x.calib_density_diff_signed ** 2,
            )
            .groupby(group_vars)
            .agg(
                brier=("brier_diff_squared", lambda x: x.mean()),
                brier_signed=("brier_diff_signed", lambda x: x.mean()),
                calib_error=("calib_diff_squared", lambda x: x.mean()),
                calib_error_signed=("calib_diff_signed", lambda x: x.mean()),
                calib_group_error=("calib_density_diff_squared", lambda x: x.mean()),
                calib_group_error_signed=(
                    "calib_density_diff_signed",
                    lambda x: x.mean(),
                ),
            )
            .reset_index()
        )

    def filter_by_group_spec(self, df, group_vars, group_values):
        for group_var, group_value in zip(group_vars, group_values):
            df = df.loc[df[group_var] == group_value]
        return df