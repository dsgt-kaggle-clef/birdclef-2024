import itertools
import os
from argparse import ArgumentParser
from pathlib import Path

import lightning as pl
import luigi
import luigi.contrib.gcs
import torch
import wandb
from lightning.pytorch.callbacks import LearningRateFinder
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from birdclef.experiment.google_vocalization.data import PetastormDataModule
from birdclef.experiment.model import LinearClassifier, TwoLayerClassifier
from birdclef.utils import spark_resource


class TrainClassifier(luigi.Task):
    input_path = luigi.Parameter()
    default_root_dir = luigi.Parameter()
    label_col = luigi.Parameter()
    feature_col = luigi.Parameter()
    loss = luigi.Parameter()
    model = luigi.Parameter()
    hidden_layer_size = luigi.OptionalIntParameter(default=64)
    hyper_params = luigi.OptionalDictParameter(default={})
    species_label = luigi.OptionalBoolParameter(default=False)
    batch_size = luigi.IntParameter(default=1000)
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
                    hp_kwargs=self.hyper_params,
                    species_label=self.species_label,
                )
            else:
                model = torch_model(
                    num_features,
                    num_labels,
                    loss=self.loss,
                    species_label=self.species_label,
                )

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
        loss_params = {
            "bce": {},
            "asl": {
                "gamma_neg": [0, 2, 4],
                "gamma_pos": [0, 1],
            },
            "sigmoidf1": {
                "S": [-1, -15, -30],
                "E": [0, 1, 2],
            },
        }
        hidden_layers = [64, 128, 256]
        return model_params, loss_params, hidden_layers


class Workflow(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    default_root_dir = luigi.Parameter()

    def generate_loss_hp_params(self, loss_params):
        """Generate all combinations of hyperparameters for a given loss function."""
        if not loss_params:
            return [{}]

        keys, values = zip(*loss_params.items())
        combinations = [
            dict(zip(keys, combination)) for combination in itertools.product(*values)
        ]
        return combinations

    def run(self):
        label_col, feature_col = "logits", "embedding"
        # get hyperparameter config
        hp = HyperparameterGrid()
        _, loss_params, hidden_layers = hp.get_hyperparameter_config()

        # Linear model grid search
        model, species_label = "linear", True
        for loss in loss_params:
            default_dir = f"{self.default_root_dir}-{model}-{loss}"
            if species_label:
                default_dir = f"{self.default_root_dir}-{model}-{loss}-species-label"
            yield TrainClassifier(
                input_path=self.input_path,
                default_root_dir=default_dir,
                label_col=label_col,
                feature_col=feature_col,
                loss=loss,
                model=model,
                species_label=species_label,
            )

        # TwoLayer model grid search
        model, hidden_layer_size, species_label = "two_layer", 256, False
        for loss in loss_params:
            for hp_params in self.generate_loss_hp_params(loss_params[loss]):
                param_log = [f"{k}{v}" for k, v in hp_params.items()]
                if len(param_log) > 0:
                    param_name = "-".join(param_log)
                    yield TrainClassifier(
                        input_path=self.input_path,
                        default_root_dir=f"{self.default_root_dir}-twolayer-{loss}-{param_name}-hidden{hidden_layer_size}",
                        label_col=label_col,
                        feature_col=feature_col,
                        loss=loss,
                        model=model,
                        hidden_layer_size=hidden_layer_size,
                        hyper_params=hp_params,
                        species_label=species_label,
                        two_layer=True,
                    )


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
