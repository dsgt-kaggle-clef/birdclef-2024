import os
from pathlib import Path

import luigi
import luigi.contrib.gcs
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from birdclef.utils import spark_resource

from .data import PetastormDataModule
from .model import LinearClassifier, TwoLayerClassifier


class TrainClassifier(luigi.Task):
    input_path = luigi.Parameter()
    feature_col = luigi.Parameter()
    label_col = luigi.Parameter()
    default_root_dir = luigi.Parameter()
    batch_size = luigi.IntParameter(default=32)
    num_partitions = luigi.IntParameter(default=32)
    two_layer = luigi.OptionalBoolParameter(default=False)

    def output(self):
        # save the model run
        return luigi.contrib.gcs.GCSTarget(f"{self.default_root_dir}/_SUCCESS")

    def run(self):
        with spark_resource() as spark:
            # data module
            data_module = PetastormDataModule(
                spark,
                self.input_path,
                self.feature_col,
                self.label_col,
                self.batch_size,
                self.num_partitions,
            )
            data_module.setup()

            # get parameters for the model
            num_features = int(
                len(data_module.train_data.select("features").first()["features"])
            )
            num_classes = int(data_module.train_data.select("label").distinct().count())

            # model module
            if self.two_layer:
                model = TwoLayerClassifier(num_features, num_classes)
            else:
                model = LinearClassifier(num_features, num_classes)

            # initialise the wandb logger and name your wandb project
            wandb_logger = WandbLogger(
                project="plantclef-2024",
                name=Path(self.default_root_dir).name,
                save_dir=self.default_root_dir,
            )

            # add your batch size to the wandb config
            wandb_logger.experiment.config["batch_size"] = self.batch_size

            model_checkpoint = ModelCheckpoint(
                dirpath=os.path.join(self.default_root_dir, "checkpoints"),
                monitor="val_loss",
                save_last=True,
            )

            # trainer
            trainer = pl.Trainer(
                max_epochs=10,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                reload_dataloaders_every_n_epochs=1,
                default_root_dir=self.default_root_dir,
                logger=wandb_logger,
                callbacks=[
                    EarlyStopping(monitor="val_loss", mode="min"),
                    model_checkpoint,
                ],
            )

            # fit model
            trainer.fit(model, data_module)

        # write the output
        with self.output().open("w") as f:
            f.write("")
