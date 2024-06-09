import os
from argparse import ArgumentParser
from pathlib import Path

import luigi
import luigi.contrib.gcs
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import LearningRateFinder
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import nn

from birdclef.torch.losses import AsymmetricLossOptimized, SigmoidF1
from birdclef.utils import spark_resource

from .data import PetastormDataModule
from .model import LinearClassifier, TwoLayerClassifier


class TrainClassifier(luigi.Task):
    input_path = luigi.Parameter()
    default_root_dir = luigi.Parameter()
    label_col = luigi.Parameter()
    feature_col = luigi.Parameter()
    loss = luigi.Parameter()
    model = luigi.Parameter()
    hidden_layer_size = luigi.OptionalIntParameter(default=64)
    batch_size = luigi.IntParameter(default=500)
    num_partitions = luigi.IntParameter(default=os.cpu_count())
    two_layer = luigi.OptionalBoolParameter(default=False)

    def output(self):
        # save the model run
        return luigi.contrib.gcs.GCSTarget(f"{self.default_root_dir}/_SUCCESS")

    def run(self):
        # Hyperparameters
        hp = HyperparameterGrid()
        model_params, loss_params, _ = hp.get_hyperparameter_config()
        # get model and loss objects
        torch_model = model_params[self.model]

        with spark_resource() as spark:
            # data module
            data_module = PetastormDataModule(
                spark,
                self.input_path,
                self.label_col,
                self.feature_col,
                self.batch_size,
                self.num_partitions,
            )
            data_module.setup()

            # get parameters for the model
            num_features = int(
                len(data_module.train_data.select("features").first()["features"])
            )
            num_labels = int(
                len(data_module.train_data.select("label").first()["label"])
            )

            # model module
            if self.two_layer:
                model = torch_model(
                    num_features,
                    num_labels,
                    loss=self.loss,
                    hidden_layer_size=self.hidden_layer_size,
                )
            else:
                model = torch_model(num_features, num_labels, loss=self.loss)

            # initialise the wandb logger and name your wandb project
            print(f"\nwanb name: {Path(self.default_root_dir).name}")
            print(f"wanb save dir: {self.default_root_dir}\n")
            wandb_logger = WandbLogger(
                project="birdclef-2024",
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
                max_epochs=20,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                reload_dataloaders_every_n_epochs=1,
                default_root_dir=self.default_root_dir,
                logger=wandb_logger,
                callbacks=[
                    EarlyStopping(monitor="val_auroc", mode="max"),
                    model_checkpoint,
                    LearningRateFinder(),
                ],
            )
            trainer.fit(model, data_module)

            # finish W&B
            wandb.finish()

        # write the output
        with self.output().open("w") as f:
            f.write("")


class HyperparameterGrid:
    def get_hyperparameter_config(self):
        # Model and Loss mappings
        model_params = {
            "linear": LinearClassifier,
            "two_layer": TwoLayerClassifier,
        }
        loss_params = [
            "bce",
            "asl",
            "sigmoidf1",
        ]
        hidden_layers = [64, 128, 256]
        return model_params, loss_params, hidden_layers


class Workflow(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    default_root_dir = luigi.Parameter()

    def run(self):
        label_col, feature_col = "logits", "embedding"
        # get hyperparameter config
        hp = HyperparameterGrid()
        _, loss_params, hidden_layers = hp.get_hyperparameter_config()

        # Linear model grid search
        model = "linear"
        yield [
            TrainClassifier(
                input_path=self.input_path,
                default_root_dir=f"{self.default_root_dir}-{model}-{loss}",
                label_col=label_col,
                feature_col=feature_col,
                loss=loss,
                model=model,
            )
            for loss in loss_params
        ]

        # TwoLayer model grid search
        model, loss = "two_layer", "bce"
        yield [
            TrainClassifier(
                input_path=self.input_path,
                default_root_dir=f"{self.default_root_dir}-twolayer-{loss}-hidden{hidden_layer_size}",
                label_col=label_col,
                feature_col=feature_col,
                loss=loss,
                model=model,
                hidden_layer_size=hidden_layer_size,
                two_layer=True,
            )
            for hidden_layer_size in hidden_layers
        ]


def parse_args():
    parser = ArgumentParser(description="Luigi pipeline")
    parser.add_argument(
        "--gcs-root-path",
        type=str,
        default="gs://dsgt-clef-birdclef-2024",
        help="Root directory for birdclef-2024 in GCS",
    )
    parser.add_argument(
        "--train-data-path",
        type=str,
        default="data/processed/google_embeddings/v1",
        help="Root directory for training data in GCS",
    )
    parser.add_argument(
        "--output-name-path",
        type=str,
        default="data/processed/google_embeddings/v1-transformed",
        help="GCS path for output Parquet files",
    )
    parser.add_argument(
        "--model-dir-path",
        type=str,
        default="models/torch-v1-google",
        help="Default root directory for storing the pytorch classifier runs",
    )
    parser.add_argument(
        "--scheduler-host",
        type=str,
        default="services.us-central1-a.c.dsgt-clef-2024.internal",
        help="scheduler host for Luigi",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Input and output paths for training workflow
    input_path = f"{args.gcs_root_path}/{args.train_data_path}"
    output_path = f"{args.gcs_root_path}/{args.output_name_path}"
    default_root_dir = f"{args.gcs_root_path}/{args.model_dir_path}"

    luigi.build(
        [
            Workflow(
                input_path=input_path,
                output_path=output_path,
                default_root_dir=default_root_dir,
            )
        ],
        scheduler_host=args.scheduler_host,
        workers=1,
    )
