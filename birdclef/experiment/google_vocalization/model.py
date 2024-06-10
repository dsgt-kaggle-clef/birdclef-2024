import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score

from birdclef.torch.losses import AsymmetricLossOptimized, ROCStarLoss, SigmoidF1


class LossFunctions:
    def get_hyperparameter_config(self):
        loss_params = {
            "bce": nn.BCEWithLogitsLoss,
            "asl": AsymmetricLossOptimized,
            "sigmoidf1": SigmoidF1,
        }
        return loss_params


class LinearClassifier(pl.LightningModule):
    def __init__(
        self,
        num_features: int,
        num_labels: int,
        loss: str = "bce",
        hp_kwargs: dict = {},
    ):
        super().__init__()
        self.num_features = num_features
        self.num_labels = num_labels
        self.hp_kwargs = hp_kwargs
        self.learning_rate = 0.002
        self.save_hyperparameters()  # Saves hyperparams in the checkpoints
        loss_fn = LossFunctions()
        loss_params = loss_fn.get_hyperparameter_config()
        self.loss = loss_params[loss](**hp_kwargs)
        self.model = nn.Linear(num_features, num_labels)
        self.f1_score = MultilabelF1Score(num_labels=num_labels, average="macro")
        self.auroc_score = MultilabelAUROC(num_labels=num_labels, average="weighted")

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _run_step(self, batch, batch_idx, step_name):
        x, y = batch["features"], batch["label"].to_dense()
        logits = self(x)
        # sigmoid the label and apply a threshold
        y_sigmoid = torch.sigmoid(y)
        y_threshold = (y_sigmoid > 0.5).float()
        loss = self.loss(logits, y_threshold)
        self.log(f"{step_name}_loss", loss, prog_bar=True)
        self.log(
            f"{step_name}_f1",
            self.f1_score(logits, y_threshold),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{step_name}_auroc",
            self.auroc_score(logits, y_threshold.to(torch.long)),
            on_step=False,
            on_epoch=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._run_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._run_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._run_step(batch, batch_idx, "test")


class TwoLayerClassifier(LinearClassifier):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_layer_size: int = 64,
        **kwargs,
    ):
        super().__init__(num_features, num_classes, **kwargs)
        self.model = nn.Sequential(
            nn.Linear(num_features, hidden_layer_size),
            nn.BatchNorm1d(hidden_layer_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layer_size, num_classes),
        )


class LinearClassifier(pl.LightningModule):
    def __init__(
        self,
        num_features: int,
        num_labels: int,
        loss: str = "bce",
        hp_kwargs: dict = {},
    ):
        super().__init__()
        self.num_features = num_features
        self.num_labels = num_labels
        self.hp_kwargs = hp_kwargs
        self.learning_rate = 0.002
        self.save_hyperparameters()  # Saves hyperparams in the checkpoints
        loss_fn = LossFunctions()
        loss_params = loss_fn.get_hyperparameter_config()
        self.loss = loss_params[loss](**hp_kwargs)
        self.model = nn.Linear(num_features, num_labels)
        self.f1_score = MultilabelF1Score(num_labels=num_labels, average="macro")
        self.auroc_score = MultilabelAUROC(num_labels=num_labels, average="weighted")

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _run_step(self, batch, batch_idx, step_name):
        x, y = batch["features"], batch["label"].to_dense()
        logits = self(x)
        # sigmoid the label and apply a threshold
        y_sigmoid = torch.sigmoid(y)
        y_threshold = (y_sigmoid > 0.5).float()
        loss = self.loss(logits, y_threshold)
        self.log(f"{step_name}_loss", loss, prog_bar=True)
        self.log(
            f"{step_name}_f1",
            self.f1_score(logits, y_threshold),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{step_name}_auroc",
            self.auroc_score(logits, y_threshold.to(torch.long)),
            on_step=False,
            on_epoch=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._run_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._run_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._run_step(batch, batch_idx, "test")
