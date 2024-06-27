import lightning as pl
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
        species_label: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_labels = num_labels
        self.hp_kwargs = hp_kwargs
        self.species_label = species_label
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
        x, y, spidx = (
            batch["features"],
            batch["label"].to_dense(),
            batch["species_index"].to_dense(),
        )
        logits = self(x)
        # sigmoid the label and apply a threshold
        y_threshold = torch.sigmoid(logits) > 0.5
        if self.species_label:
            # compute z: row-wise sum of elements in y, cast as boolean
            indicator_call = y_threshold.sum(dim=1, keepdim=True) > 0
            # compute s: one-hot encoded species matrix (NxK)
            indicator_species = torch.zeros_like(logits, dtype=torch.bool).scatter(
                1, spidx.to(torch.int).unsqueeze(1), 1
            )
            # compute r: r = y + (s * z)
            # multiply the indicator by the species matrix and then add it to the original
            # update logits for the loss computation
            label = torch.logical_or(
                y_threshold, torch.logical_and(indicator_call, indicator_species)
            )
        else:
            label = y_threshold
        label = label.to(torch.float)

        loss = self.loss(logits, label)
        self.log(f"{step_name}_loss", loss, prog_bar=True)
        self.log(
            f"{step_name}_f1",
            self.f1_score(logits, label),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{step_name}_auroc",
            self.auroc_score(logits, label.to(torch.int)),
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
