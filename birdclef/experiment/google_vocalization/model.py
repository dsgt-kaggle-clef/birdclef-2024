import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.classification import MultilabelAUROC


class LinearClassifier(pl.LightningModule):
    def __init__(self, num_features: int, num_labels: int, loss=nn.BCEWithLogitsLoss()):
        super().__init__()
        self.num_features = num_features
        self.num_labels = num_labels
        self.loss = loss
        self.save_hyperparameters()  # Saves hyperparams in the checkpoints
        self.model = nn.Linear(num_features, num_labels)
        self.learning_rate = 0.002
        self.auroc_score = MultilabelAUROC(num_labels=num_labels, average="weighted")

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _run_step(self, batch, batch_idx, step_name):
        x, y = batch["features"], batch["label"]
        logits = self(x)
        # sigmoid the label and apply a threshold
        y_sigmoid = torch.sigmoid(y)
        y_threshold = (y_sigmoid > 0.5).float()
        loss = self.loss(logits, y_threshold)
        self.log(f"{step_name}_loss", loss, prog_bar=True)
        y_threshold = y_threshold.to(
            torch.long
        )  # AUROC expects the target tensor of type long
        self.log(
            f"{step_name}_auroc",
            self.auroc_score(logits, y_threshold),
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
