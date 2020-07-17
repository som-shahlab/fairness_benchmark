import copy
import torch
import torch.nn as nn
from torchcontrib.optim import SWA

from prediction_utils.pytorch_utils.metrics import MetricComparator, MetricLogger
from prediction_utils.pytorch_utils.layers import (
    SparseLinear,
    LinearLayer,
    SequentialLayers,
    FeedforwardNet,
    EmbeddingBagLinear,
)


class TorchModel:
    """
    This is the upper level class that provides training and logging code for a Pytorch model.
    To initialize the model, provide a config_dict with relevant parameters.
    The default model is logistic regression. Subclass and override init_model() for custom usage.

    The user is intended to interact with this class primarily through the train method.
    """

    def __init__(self, *args, **kwargs):
        self.config_dict = self.get_default_config()
        self.config_dict = self.override_config(**kwargs)
        if self.config_dict.get("input_dim") is None:
            raise ValueError("Must provide input_dim")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model = self.init_model()
        self.model.apply(self.weights_init)
        self.model.to(self.device)
        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_scheduler()
        if self.config_dict.get("SWA"):
            self.optimizer = self.init_SWA(self.optimizer)
        self.criterion = self.init_loss()
        self.metric_comparator = self.init_metric_comparator()

    def get_default_config(self):
        return {
            "input_dim": None,
            "lr": 1e-4,
            "num_epochs": 10,
            "selection_metric": "loss",
            "batch_size": 256,
            "output_dim": 2,
            "iters_per_epoch": 100,
            "gamma": None,
            "SWA": False,
            "early_stopping": False,
            "early_stopping_patience": 5,
        }

    def override_config(self, **override_dict):
        return {**self.config_dict, **override_dict}

    def transform_batch(self, batch, keys=None):
        """
        Sends a batch to the device
        """
        if keys is None:
            keys = batch.keys()

        result = {}
        for key in batch.keys():
            if (isinstance(batch[key], torch.Tensor)) and (key in keys):
                result[key] = batch[key].to(self.device, non_blocking=True)
            elif isinstance(batch[key], dict):
                result[key] = self.transform_batch(batch[key])
            else:
                result[key] = batch[key]

        return result

    def get_transform_batch_keys(self):
        """
        Returns the names of the list of tensors that sent to device
        """
        return ["features", "labels"]

    @staticmethod
    def weights_init(m):
        """
        Initialize the weights with Glorot initilization
        """
        if (
            isinstance(m, nn.Linear)
            or isinstance(m, nn.EmbeddingBag)
            or isinstance(m, nn.Embedding)
            or isinstance(m, SparseLinear)
        ):
            nn.init.xavier_normal_(m.weight)

    def init_model(self):
        """
        Initializes the model
        """
        return LinearLayer(
            self.config_dict["input_dim"], self.config_dict["output_dim"]
        )

    def init_optimizer(self):
        """
        Initialize an optimizer
        """
        params = [{"params": self.model.parameters()}]
        optimizer = torch.optim.Adam(params, lr=self.config_dict["lr"])
        return optimizer

    def init_scheduler(self):
        """
        A learning rate scheduler
        """
        gamma = self.config_dict.get("gamma")
        if gamma is None:
            return None
        else:
            return torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

    def init_SWA(self, optimizer):
        print("Using SWA")
        opt = SWA(
            optimizer,
            swa_start=self.config_dict["iters_per_epoch"] * 5,
            swa_freq=self.config_dict["iters_per_epoch"] * 2,
            swa_lr=self.config_dict["lr"] * 1e-1,
        )
        return opt

    def init_loss(self, reduction="mean"):
        """
        Cross entropy
        """
        return nn.CrossEntropyLoss(reduction="mean")

    def init_metric_comparator(self):
        if self.config_dict["selection_metric"] in ["auc", "auprc", "brier"]:
            comparator = MetricComparator("max")
        else:
            comparator = MetricComparator("min")
        return comparator

    def get_loss_names(self):
        return ["loss"]

    def forward_on_batch(self, the_data):
        """
        Run the forward pass, returning a batch_loss_dict and outputs
        """
        outputs = self.model(the_data["features"])
        loss_dict_batch = {"loss": self.criterion(outputs, the_data["labels"])}
        return loss_dict_batch, outputs

    def train(self, loaders, **kwargs):
        """
        Method that trains the model.
            Args:
                loaders: A dictionary of DataLoaders with keys corresponding to phases
                kwargs: Additional arguments to override in the config_dict
            Returns:
                result_dict: A dictionary with metrics recorded every epoch

        """
        self.config_dict = self.override_config(**kwargs)
        best_performance = self.metric_comparator.get_inital_value()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        metric_logger = MetricLogger(losses=self.get_loss_names())
        phases = kwargs.get("phases", ["train", "val"])
        epochs_since_improvement = 0
        best_epoch = 0
        for epoch in range(self.config_dict["num_epochs"]):
            if self.config_dict["early_stopping"] & (
                epochs_since_improvement >= self.config_dict["early_stopping_patience"]
            ):
                print(
                    "Early stopping at epoch {epoch} with best epoch {best_epoch}".format(
                        epoch=epoch - 1, best_epoch=best_epoch
                    )
                )
                break
            print("Epoch {}/{}".format(epoch, self.config_dict["num_epochs"] - 1))
            print("-" * 10)
            for phase in phases:
                self.model.train(phase == "train")
                metric_logger.init_metric_dicts()
                for i, the_data in enumerate(loaders[phase]):
                    self.optimizer.zero_grad()
                    the_data = self.transform_batch(
                        the_data, keys=self.get_transform_batch_keys()
                    )
                    loss_dict_batch, outputs = self.forward_on_batch(the_data)

                    if phase == "train":
                        loss_dict_batch["loss"].backward()
                        self.optimizer.step()

                    metric_logger.update_loss_dict(
                        loss_dict_batch, batch_size=the_data["labels"].shape[0]
                    )
                    metric_logger.update_output_dict(
                        outputs=outputs,
                        labels=the_data["labels"],
                        row_id=the_data["row_id"],
                    )
                if phase == "train":
                    if self.scheduler is not None:
                        self.scheduler.step()

                metric_logger.compute_metrics_epoch(phase=phase)

                print("Phase: {}:".format(phase))
                metric_logger.print_metrics()
                epoch_performance = metric_logger.get_metrics_epoch()

                if phase == "val":
                    if self.metric_comparator.is_better(
                        epoch_performance[self.config_dict["selection_metric"]],
                        best_performance,
                    ):
                        print("Best model updated")
                        best_epoch = epoch
                        best_performance = epoch_performance[
                            self.config_dict["selection_metric"]
                        ]
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        best_optimizer_state = copy.deepcopy(
                            self.optimizer.state_dict()
                        )
                        best_scheduler_state = copy.deepcopy(
                            self.scheduler.state_dict()
                        )
                        epochs_since_improvement = 0
                    else:
                        epochs_since_improvement += 1

        if self.config_dict.get("SWA"):
            print("Applying SWA")
            self.optimizer.swap_swa_sgd()
        elif "val" in phases:
            print("Best performance: {:4f}".format(best_performance))
            self.model.load_state_dict(best_model_wts)
            self.optimizer.load_state_dict(best_optimizer_state)
            self.scheduler.load_state_dict(best_scheduler_state)

        self.epoch = epoch
        return {"performance": metric_logger.get_metrics_overall()}

    def predict(self, loaders, phases=["test"], return_outputs=True):
        """
        Method that trains the model.
            Args:
                loaders: A dictionary of DataLoaders with keys corresponding to phases
                kwargs: Additional arguments to override in the config_dict
            Returns:
                result_dict: A dictionary with metrics recorded every epoch

        """
        metric_logger = MetricLogger(phases=phases, losses=self.get_loss_names())
        self.model.train(False)
        output_dict = {}
        with torch.no_grad():
            for phase in phases:
                print("Evaluating on phase: {phase}".format(phase=phase))
                metric_logger.init_metric_dicts()
                for i, the_data in enumerate(loaders[phase]):
                    the_data = self.transform_batch(
                        the_data, keys=self.get_transform_batch_keys()
                    )
                    loss_dict_batch, outputs = self.forward_on_batch(the_data)
                    metric_logger.update_loss_dict(
                        loss_dict_batch, batch_size=the_data["labels"].shape[0]
                    )
                    metric_logger.update_output_dict(
                        outputs=outputs,
                        labels=the_data["labels"],
                        row_id=the_data["row_id"],
                    )

                metric_logger.compute_metrics_epoch(phase=phase)
                metric_logger.print_metrics()
                if return_outputs:
                    output_dict[phase] = metric_logger.get_output_dict()

        result_dict = {"performance": metric_logger.get_metrics_overall()}
        if return_outputs:
            result_dict["outputs"] = metric_logger.process_output_dict(output_dict)
        return result_dict

    def load_weights(self, the_path):
        """
        Save the model weights to a file
        """
        self.model.load_state_dict(torch.load(the_path))

    def save_weights(self, the_path):
        """
        Load model weights from a file
        """
        torch.save(self.model.state_dict(), the_path)


class FeedforwardNetModel(TorchModel):
    """
    The primary class for a feedforward network.
    Has options for sparse inputs, residual connections, dropout, and layer normalization.
    """

    def get_default_config(self):
        config_dict = super().get_default_config()
        update_dict = {
            "hidden_dim_list": [
                128
            ],  # To-do figure out how to harmonize hidden_dim_list and num_hidden/hidden_dim
            "drop_prob": 0.0,
            "normalize": False,
            "sparse": True,
            "sparse_mode": "csr",  # alternatively, "convert"
            "resnet": False,
        }

        return {**config_dict, **update_dict}

    def init_model(self):
        model = FeedforwardNet(
            in_features=self.config_dict["input_dim"],
            hidden_dim_list=self.config_dict["hidden_dim_list"],
            output_dim=self.config_dict["output_dim"],
            drop_prob=self.config_dict["drop_prob"],
            normalize=self.config_dict["normalize"],
            sparse=self.config_dict["sparse"],
            sparse_mode=self.config_dict["sparse_mode"],
            resnet=self.config_dict["resnet"],
        )
        return model


class FixedWidthModel(FeedforwardNetModel):
    """
    The primary class for a feedforward network with a fixed number of hidden layers of equal size.
    Has options for sparse inputs, residual connections, dropout, and layer normalization.
    """

    def get_default_config(self):
        config_dict = super().get_default_config()
        update_dict = {"num_hidden": 1, "hidden_dim": 128}
        update_dict["hidden_dim_list"] = update_dict["num_hidden"] * [
            update_dict["hidden_dim"]
        ]

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


class BottleneckModel(FeedforwardNetModel):
    def get_default_config(self):
        config_dict = super().get_default_config()
        update_dict = {"bottleneck_size": 128, "num_hidden": 2}
        return {**config_dict, **update_dict}

    def init_model(self):

        hidden_dim_list = [
            self.config_dict["bottleneck_size"] * (2 ** (i))
            for i in reversed(range(self.config_dict["num_hidden"]))
        ]

        model = FeedforwardNet(
            in_features=self.config_dict["input_dim"],
            hidden_dim_list=hidden_dim_list,
            output_dim=self.config_dict["output_dim"],
            drop_prob=self.config_dict["drop_prob"],
            normalize=self.config_dict["normalize"],
            sparse=self.config_dict["sparse"],
            sparse_mode=self.config_dict["sparse_mode"],
            resnet=self.config_dict["resnet"],
        )
        return model


class SparseLogisticRegression(TorchModel):
    """
    A model that perform sparse logistic regression
    """

    def init_model(self):
        layer = SparseLinear(
            self.config_dict["input_dim"], self.config_dict["output_dim"]
        )
        model = SequentialLayers([layer])
        return model


class SparseLogisticRegressionEmbed(TorchModel):
    """
    A model that performs sparse logistic regression with an EmbeddingBag encoder
    """

    def init_model(self):
        layer = EmbeddingBagLinear(
            self.config_dict["input_dim"], self.config_dict["output_dim"]
        )
        model = SequentialLayers([layer])
        return model
