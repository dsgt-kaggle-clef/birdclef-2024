import itertools
import os
from argparse import ArgumentParser
from pathlib import Path

import lightning as pl
import luigi
import luigi.contrib.gcs
import pandas as pd
import torch
import wandb
from lightning.pytorch.callbacks import LearningRateFinder
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from birdclef.experiment.birdnet.data import PetastormDataModule
from birdclef.experiment.model import LinearClassifier, TwoLayerClassifier
from birdclef.inference.birdnet import BirdNetInference
from birdclef.tasks import RsyncGCSFiles, maybe_gcs_target
from birdclef.utils import spark_resource


class EmbedSpeciesAudio(luigi.Task):
    """Embed all audio files for a species and save to a parquet file."""

    remote_root = luigi.Parameter()
    local_root = luigi.Parameter()

    audio_path = luigi.Parameter()
    metadata_path = luigi.Parameter()
    species = luigi.Parameter()
    output_path = luigi.Parameter()

    def output(self):
        return maybe_gcs_target(
            f"{self.remote_root}/{self.output_path}/{self.species}.parquet"
        )

    def run(self):
        yield RsyncGCSFiles(
            src_path=f"{self.remote_root}/{self.audio_path}/{self.species}",
            dst_path=f"{self.local_root}/{self.audio_path}/{self.species}",
        )

        inference = self.get_inference()
        out_path = f"{self.remote_root}/{self.output_path}/{self.species}.parquet"
        df = inference.predict_species_df(
            f"{self.local_root}/{self.audio_path}",
            self.species,
            out_path,
        )
        print(df.head())

    def get_inference(self):
        return BirdNetInference(
            metadata_path=f"{self.remote_root}/{self.metadata_path}",
        )


class EmbedWorkflow(luigi.Task):
    remote_root = luigi.Parameter()
    local_root = luigi.Parameter()

    audio_path = luigi.Parameter()
    metadata_path = luigi.Parameter()
    intermediate_path = luigi.Parameter()
    output_path = luigi.Parameter()

    partitions = luigi.IntParameter(default=16)

    def output(self):
        return maybe_gcs_target(f"{self.remote_root}/{self.output_path}/_SUCCESS")

    def run(self):
        metadata = pd.read_csv(f"{self.remote_root}/{self.metadata_path}")
        species_list = metadata["primary_label"].unique()

        tasks = []
        for species in species_list:
            task = EmbedSpeciesAudio(
                remote_root=self.remote_root,
                local_root=self.local_root,
                audio_path=self.audio_path,
                metadata_path=self.metadata_path,
                species=species,
                output_path=self.intermediate_path,
            )
            tasks.append(task)
        yield tasks

        with spark_resource(memory="16g") as spark:
            spark.read.parquet(
                f"{self.remote_root}/{self.intermediate_path}/*.parquet"
            ).repartition(self.partitions).write.parquet(
                f"{self.remote_root}/{self.output_path}", mode="overwrite"
            )


class TrainClassifier(luigi.Task):
    input_birdnet_path = luigi.Parameter()
    input_google_path = luigi.Parameter()
    default_model_dir = luigi.Parameter()
    label_col = luigi.Parameter(default="logits")
    feature_col = luigi.Parameter(default="embedding")
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
        return luigi.contrib.gcs.GCSTarget(f"{self.default_model_dir}/_SUCCESS")

    def run(self):
        # Hyperparameters
        hp = HyperparameterGrid()
        model_params, _, _ = hp.get_hyperparameter_config()
        # get model and loss objects
        torch_model = model_params[self.model]

        with spark_resource() as spark:
            # data module
            data_module = PetastormDataModule(
                spark,
                self.input_birdnet_path,
                self.input_google_path,
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
                    hp_kwargs=self.hyper_params,
                )

            # initialise the wandb logger and name your wandb project
            print(f"\nwanb name: {Path(self.default_model_dir).name}")
            print(f"wanb save dir: {self.default_model_dir}\n")
            wandb_logger = WandbLogger(
                project="birdclef-2024",
                name=Path(self.default_model_dir).name,
                save_dir=self.default_model_dir,
            )

            # add your batch size to the wandb config
            wandb_logger.experiment.config["batch_size"] = self.batch_size

            model_checkpoint = ModelCheckpoint(
                dirpath=os.path.join(self.default_model_dir, "checkpoints"),
                monitor="val_loss",
                save_last=True,
            )

            # trainer
            print(f"\ndevice: {'gpu' if torch.cuda.is_available() else 'cpu'}\n")
            trainer = pl.Trainer(
                max_epochs=20,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                reload_dataloaders_every_n_epochs=1,
                default_root_dir=self.default_model_dir,
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
            "twolayer": TwoLayerClassifier,
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
    remote_root = luigi.Parameter()
    local_root = luigi.Parameter()
    audio_path = luigi.Parameter()
    metadata_path = luigi.Parameter()
    intermediate_path = luigi.Parameter()
    output_path = luigi.Parameter()
    input_birdnet_path = luigi.Parameter()
    input_google_path = luigi.Parameter()
    default_model_dir = luigi.Parameter()

    def generate_loss_hp_params(self, loss_params):
        """Generate all combinations of hyperparameters for a given loss function."""
        if not loss_params:
            return [{}]

        keys, values = zip(*loss_params.items())
        combinations = [
            dict(zip(keys, combination)) for combination in itertools.product(*values)
        ]
        return combinations

    def generate_hp_parameters(self, model, species_label, loss_params, **kwargs):
        for loss in loss_params:
            for hp_params in self.generate_loss_hp_params(loss_params[loss]):
                param_log = [f"{k}{v}" for k, v in hp_params.items()]
                if len(param_log) > 0:
                    param_name = "-".join(param_log)
                else:
                    param_name = "default"
                default_dir = f"{self.default_model_dir}-{model}-{loss}-{param_name}"
                if species_label:
                    default_dir = f"{default_dir}-species-label"
                yield dict(
                    input_birdnet_path=self.input_birdnet_path,
                    input_google_path=self.input_google_path,
                    default_model_dir=default_dir,
                    loss=loss,
                    model=model,
                    hyper_params=hp_params,
                    species_label=species_label,
                    **kwargs,
                )

    def run(self):
        yield EmbedWorkflow(
            remote_root=self.remote_root,
            local_root=self.local_root,
            audio_path=self.audio_path,
            metadata_path=self.metadata_path,
            intermediate_path=self.intermediate_path,
            output_path=self.output_path,
        )

        hp = HyperparameterGrid()
        _, loss_params, hidden_layers = hp.get_hyperparameter_config()

        tasks = []
        for model, species_label in [("linear", False)]:
            for kwargs in self.generate_hp_parameters(
                model, species_label, loss_params
            ):
                tasks.append(TrainClassifier(**kwargs))
        yield tasks

        # now test the default two layer model over the different hidden layer sizes
        tasks = []
        for model, species_label in [("twolayer", False), ("twolayer", True)]:
            for hidden_layer_size in hidden_layers:
                for kwargs in self.generate_hp_parameters(
                    model,
                    species_label,
                    {"bce": {}},
                    hidden_layer_size=hidden_layer_size,
                ):
                    tasks.append(TrainClassifier(**kwargs))
        yield tasks

        # tasks = []
        # model, hidden_layer_size = "two_layer", 256
        # for species_label in [False, True]:
        #     for kwargs in self.generate_hp_parameters(
        #         model,
        #         species_label,
        #         loss_params,
        #         hidden_layer_size=hidden_layer_size,
        #     ):
        #         tasks.append(TrainClassifier(**kwargs))
        # yield tasks


def parse_args():
    parser = ArgumentParser()

    default_out_folder = "birdnet"
    gcs_root_path = "gs://dsgt-clef-birdclef-2024"
    train_data_google_path = "data/processed/google_embeddings/v1"
    train_data_birdnet_path = f"data/processed/{default_out_folder}/v1"
    model_dir_path = f"models/torch-v1-{default_out_folder}"
    defaults = {
        "remote-root": "gs://dsgt-clef-birdclef-2024/data",
        "local-root": "/mnt/data",
        "audio-path": "raw/birdclef-2024/train_audio",
        "metadata-path": "raw/birdclef-2024/train_metadata.csv",
        "intermediate-path": f"intermediate/{default_out_folder}/v1",
        "output-path": f"processed/{default_out_folder}/v1",
        "scheduler-host": "services.us-central1-a.c.dsgt-clef-2024.internal",
        "workers": 1,
        "input_birdnet_path": f"{gcs_root_path}/{train_data_birdnet_path}",
        "input_google_path": f"{gcs_root_path}/{train_data_google_path}",
        "default_model_dir": f"{gcs_root_path}/{model_dir_path}",
    }

    for arg, default in defaults.items():
        parser.add_argument(f"--{arg}", type=type(default), default=default)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    luigi.build(
        [
            Workflow(
                **{
                    k: v
                    for k, v in vars(args).items()
                    if k not in ["scheduler_host", "workers"]
                }
            )
        ],
        scheduler_host=args.scheduler_host,
        workers=args.workers,
    )
