import copy
import torch
import torch.nn.functional as F
import itertools

from sklearn.metrics import roc_auc_score, average_precision_score
from prediction_utils.pytorch_utils.models import TorchModel
from prediction_utils.pytorch_utils.layers import FeedforwardNet
from prediction_utils.pytorch_utils.metrics import MetricLogger


class GroupRegularizedModel(TorchModel):
    """
    A model that penalizes differences in a quantity across groups
    """

    def compute_group_regularization_loss(self, outputs, labels, group):
        """
        Computes a regularization term defined in terms of model outputs, labels, and group.
        This class should be overriden and this regularization term defined.
        """
        raise NotImplementedError

    def get_default_config(self):
        """
        Default parameters
        """
        config_dict = super().get_default_config()
        update_dict = {
            "num_hidden": 1,
            "hidden_dim": 128,
            "drop_prob": 0.0,
            "normalize": False,
            "sparse": True,
            "sparse_mode": "csr",  # alternatively, "convert"
            "resnet": False,
            "temperature": 1.0,
        }
        return {**config_dict, **update_dict}

    def init_model(self):
        model = FeedforwardNet(
            in_features=self.config_dict["input_dim"],
            hidden_dim_list=self.config_dict["num_hidden"]
            * [self.config_dict["hidden_dim"]],
            output_dim=self.config_dict["output_dim"],
            drop_prob=self.config_dict["drop_prob"],
            normalize=self.config_dict["normalize"],
            sparse=self.config_dict["sparse"],
            sparse_mode=self.config_dict["sparse_mode"],
            resnet=self.config_dict["resnet"],
        )
        return model

    def get_transform_batch_keys(self):
        """
        Returns the names of the list of tensors that sent to device
        """
        return ["features", "labels", "group"]

    def get_loss_names(self):
        return ["loss", "supervised", "group_regularization"]

    def forward_on_batch(self, the_data):
        """
        Run the forward pass, returning a batch_loss_dict and outputs
        """
        loss_dict_batch = {}
        inputs, labels, group = (
            the_data["features"],
            the_data["labels"],
            the_data["group"],
        )
        outputs = self.model(inputs)
        # Compute the loss
        loss_dict_batch["supervised"] = self.criterion(outputs, labels)
        loss_dict_batch[
            "group_regularization"
        ] = self.compute_group_regularization_loss(outputs, labels, group)
        loss_dict_batch["loss"] = loss_dict_batch["supervised"] + (
            self.config_dict["lambda_group_regularization"]
            * loss_dict_batch["group_regularization"]
        )
        return loss_dict_batch, outputs


def group_regularized_model(model_type="loss"):
    """
    A switch function that returns an instance of GroupRegularizedModel
    """
    class_dict = {
        "mmd": MMDModel,
        "mean_prediction": EqualMeanPredictionModel,
    }
    the_class = class_dict.get(model_type, None)
    if the_class is None:
        raise ValueError("model_type not defined in group_regularized_model")
    return the_class


class MMDModel(GroupRegularizedModel):
    """
    Model that minimizes distributional discrepancy between predictions belonging to different groups.
    In the default case, corresponds to threshold-free demographic parity.
    If made conditional on the outcome, corresponds to equalized odds.
    """

    def get_default_config(self):
        config_dict = super().get_default_config()
        update_dict = {
            "mmd_mode": "conditional"
            # "conditional" -> eq_odds,
            # "conditional_pos" -> eq_opportunity_pos,
            # "conditional_neg" -> eq_opportunity_neg,
            # "unconditional" -> demographic_parity
        }
        return {**config_dict, **update_dict}

    def compute_mmd(self, x, y):
        """
        Compute an MMD
        Based on: https://github.com/napsternxg/pytorch-practice/blob/master/Pytorch%20-%20MMD%20VAE.ipynb
        """
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
        return mmd

    @staticmethod
    def compute_kernel(x, y):
        """
        Gaussian RBF kernel for use in an MMD
        # Based on https://github.com/napsternxg/pytorch-practice/blob/master/Pytorch%20-%20MMD%20VAE.ipynb
        """
        dim = x.size(1)
        assert dim == y.size(1)
        kernel_input = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).mean(2) / float(
            dim  # scale-invariant gamma
        )
        return torch.exp(-kernel_input)  # (x_size, y_size)

    def compute_mmd_group(self, x, group):
        """
        Compute the MMD between data for each group
        """
        unique_groups = group.unique()
        mmd = torch.FloatTensor([0.0]).to(self.device)
        if len(unique_groups) == 1:
            return mmd
        i = 0
        if self.config_dict["group_regularization_mode"] == "overall":
            for the_group in unique_groups:
                mmd = mmd + self.compute_mmd(x[group == the_group], x)
                i = i + 1
        elif self.config_dict["group_regularization_mode"] == "group":
            for comb in itertools.combinations(unique_groups, 2):
                mmd = mmd + self.compute_mmd(x[group == comb[0]], x[group == comb[1]])
                i = i + 1
        return mmd / i

    def compute_group_regularization_loss(self, outputs, labels, group):
        """
        Partition the data on the labels and compute the group MMD
        """
        mmd = torch.FloatTensor([0.0]).to(self.device)
        outputs = F.log_softmax(outputs, dim=1)[:, -1].unsqueeze(1)
        if self.config_dict["mmd_mode"] == "unconditional":
            mmd = mmd + self.compute_mmd_group(outputs, group)
        else:
            if self.config_dict["mmd_mode"] == "conditional":
                unique_labels = labels.unique()
            elif self.config_dict["mmd_mode"] == "conditional_pos":
                unique_labels = [1]
            elif self.config_dict["mmd_mode"] == "conditional_neg":
                unique_labels = [0]
            else:
                raise ValueError("Invalid option provided to mmd_mode")
            for the_label in unique_labels:
                if (labels == the_label).sum() > 0:
                    mmd = mmd + self.compute_mmd_group(
                        outputs[labels == the_label], group[labels == the_label]
                    )
                else:
                    print("Skipping regularization due to no samples")
        return mmd


class GroupMetricRegularizedModel(GroupRegularizedModel):
    """
    A model that minimizes the difference in a metric across groups.
    """

    def compute_group_regularization_loss(self, outputs, labels, group):
        """
        Computes the group regularization.
        Note: Returns 0 if all labels are negative
        """
        unique_groups = group.unique()
        result = torch.FloatTensor([0.0]).to(self.device)
        if (len(unique_groups) == 1) or (labels.sum() == 0):
            return result

        if self.config_dict["group_regularization_mode"] == "overall":
            # Regularize discrepancy in the value of the metric on groups vs. whole population
            overall_metric = self.compute_metric_overall(outputs, labels)

            for i, the_group in enumerate(unique_groups):
                if labels[group == the_group].sum() > 0:
                    result = result + (
                        (
                            self.compute_metric_group(
                                outputs[group == the_group], labels[group == the_group]
                            )
                            - overall_metric
                        )
                        ** 2
                    )
        elif self.config_dict["group_regularization_mode"] == "group":
            # Regularize discrepancy in the value of the metric between groups pairwise
            for group_0, group_1 in itertools.combinations(unique_groups, 2):
                if (labels[group == group_0].sum() > 0) and (
                    labels[group == group_1].sum() > 0
                ):
                    result = (
                        result
                        + (
                            (
                                self.compute_metric_group(
                                    outputs[group == group_0], labels[group == group_0]
                                )
                                - self.compute_metric_group(
                                    outputs[group == group_1], labels[group == group_1]
                                )
                            )
                        )
                        ** 2
                    )
        return result

    def compute_metric_overall(self, outputs, labels):
        """
        Computes the value of the metric on a batch. Does not need to be differentiable.
        """
        raise NotImplementedError

    def compute_metric_group(self, outputs, labels):
        """
        Computes the value of the metric on a group. Needs to be differentiable
        """
        raise NotImplementedError


class EqualMeanPredictionModel(GroupMetricRegularizedModel):
    def get_default_config(self):
        config_dict = super().get_default_config()
        update_dict = {
            "mean_prediction_mode": "conditional"  # conditional -> eq_odds; unconditional -> demographic parity
        }
        return {**config_dict, **update_dict}

    def compute_group_regularization_loss(self, outputs, labels, group):
        """
        Partition the data on the labels and compute the group MMD
        """
        result = torch.FloatTensor([0.0]).to(self.device)
        outputs = F.log_softmax(outputs, dim=1)[:, -1].unsqueeze(1)
        if self.config_dict["mean_prediction_mode"] == "unconditional":
            result = result + self.compute_group_regularization_loss_helper(
                outputs, labels, group
            )
        else:
            if self.config_dict["mean_prediction_mode"] == "conditional":
                unique_labels = labels.unique()
            elif self.config_dict["mean_prediction_mode"] == "conditional_pos":
                unique_labels = [1]
            elif self.config_dict["mean_prediction_mode"] == "conditional_neg":
                unique_labels = [0]
            else:
                raise ValueError("Invalid option provided to mean_prediction_mode")
            for the_label in unique_labels:
                if (labels == the_label).sum() > 0:
                    result = result + self.compute_group_regularization_loss_helper(
                        outputs[labels == the_label],
                        labels[labels == the_label],
                        group[labels == the_label],
                    )
                else:
                    print("Skipping regularization due to no samples")

        return result

    def compute_group_regularization_loss_helper(self, outputs, labels, group):
        result = torch.FloatTensor([0.0]).to(self.device)
        unique_groups = group.unique()
        for i, the_group in enumerate(unique_groups):
            result = result + (
                (outputs[group == the_group].mean() - outputs.mean()) ** 2
            )
        return result
