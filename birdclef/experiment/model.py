import warnings

import lightning as pl
import torch
from torch import nn
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score

from birdclef.torch.losses import AsymmetricLossOptimized, ROCStarLoss, SigmoidF1

LOSS_PARAMS = {
    "bce": nn.BCEWithLogitsLoss,
    "asl": AsymmetricLossOptimized,
    "sigmoidf1": SigmoidF1,
}


class LinearClassifier(pl.LightningModule):
    def __init__(
        self,
        num_features: int,
        num_labels: int,
        loss: str = "bce",
        hp_kwargs: dict = {},
        species_label: bool = False,
        should_threshold: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_labels = num_labels
        self.hp_kwargs = hp_kwargs
        self.species_label = species_label
        self.learning_rate = 0.002
        self.should_threshold = should_threshold

        self.save_hyperparameters()  # Saves hyperparams in the checkpoints
        self.loss = LOSS_PARAMS[loss](**hp_kwargs)
        self.model = nn.Linear(num_features, num_labels)
        self.f1_score = MultilabelF1Score(num_labels=num_labels, average="macro")
        self.auroc_score = MultilabelAUROC(num_labels=num_labels, average="weighted")

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _run_step(self, batch, batch_idx, step_name):
        # sigmoid the label and apply a threshold
        logit = batch["label"]
        y_threshold = (
            (torch.sigmoid(logit.to_dense()) > 0.5) if self.should_threshold else logit
        )

        if self.species_label:
            # compute z: row-wise sum of elements in y, cast as boolean
            indicator_call = y_threshold.sum(dim=1, keepdim=True) > 0
            # compute s: one-hot encoded species matrix (NxK)
            indicator_species = torch.zeros_like(y_threshold, dtype=torch.bool).scatter(
                1, batch["species_index"].to(torch.int64).unsqueeze(1), 1
            )
            # compute r: r = y + (s * z)
            # multiply the indicator by the species matrix and then add it to the original
            # update logits for the loss computation
            y_threshold = torch.logical_or(
                y_threshold,
                torch.logical_and(indicator_call, indicator_species),
            )

        label = y_threshold.to(torch.float)
        logits_pred = self(batch["features"])

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=UserWarning)

            loss = self.loss(logits_pred, label)
            f1_score = self.f1_score(logits_pred, label)
            auroc_score = self.auroc_score(logits_pred, label.to(torch.int))

            self.log(f"{step_name}_loss", loss, prog_bar=True)
            self.log(f"{step_name}_f1", f1_score, on_step=False, on_epoch=True)
            self.log(f"{step_name}_auroc", auroc_score, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._run_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._run_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._run_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        # NOTE: it's a pain to rename everything in the tensorflow dataloader to feature,
        # so instead we just pass the name from the soundscape dataloader instead
        batch["prediction"] = torch.sigmoid(self(batch["embedding"]))
        return batch


class TwoLayerClassifier(LinearClassifier):
    def __init__(
        self,
        num_features: int,
        num_labels: int,
        hidden_layer_size: int = 64,
        **kwargs,
    ):
        super().__init__(num_features, num_labels, **kwargs)
        self.model = nn.Sequential(
            nn.Linear(num_features, hidden_layer_size),
            nn.BatchNorm1d(hidden_layer_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layer_size, num_labels),
        )


class LSTMClassifier(LinearClassifier):
    # TODO: in progress
    def __init__(
        self,
        num_features: int,
        num_labels: int,
        lstm_size: int = 64,
        **kwargs,
    ):
        super().__init__(num_features, num_labels, **kwargs)
        self.seq_features = 4
        self.seq_len = num_features // self.seq_features

        self.lstm = nn.LSTM(self.seq_features, lstm_size)
        self.fc = nn.Linear(lstm_size, num_labels)

    def forward(self, x):
        x = x.reshape(self.seq_len, -1, self.seq_features)
        x, _ = self.lstm(x)
        x = self.fc(x[-1])
        return x


class ConvLSTMClassifier(LinearClassifier):
    # TODO: in progress
    def __init__(
        self,
        num_features: int,
        num_labels: int,
        conv_size: int = 128,
        conv_kernel: int = 7,
        lstm_size: int = 128,
        **kwargs,
    ):
        super().__init__(num_features, num_labels, **kwargs)
        self.conv = nn.Conv1d(num_features, conv_size, conv_kernel)
        self.lstm = nn.LSTM(num_features, lstm_size)
        self.fc = nn.Linear(lstm_size, num_labels)

    def forward(self, x):
        x = x.reshape(self.seq_len, -1, self.seq_features)
        x, _ = self.lstm(x)
        x = self.fc(x[-1])
        return x
