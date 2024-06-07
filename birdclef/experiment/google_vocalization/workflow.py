import itertools
import os
from argparse import ArgumentParser
from pathlib import Path

import luigi
import luigi.contrib.gcs
import pytorch_lightning as pl
import torch
from pyspark.ml import Pipeline
from pyspark.ml.feature import SQLTransformer
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import nn

from birdclef.torch.losses import ASLSingleLabel, SigmoidF1
from birdclef.transforms import TransformEmbedding
from birdclef.utils import spark_resource

from .data import PetastormDataModule
from .model import LinearClassifier, TwoLayerClassifier


class ProcessBase(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    sample_col = luigi.Parameter(default="species_id")
    num_partitions = luigi.OptionalIntParameter(default=500)
    sample_id = luigi.OptionalIntParameter(default=None)
    num_sample_id = luigi.OptionalIntParameter(default=10)

    def output(self):
        if self.sample_id is None:
            # save both the model pipeline and the dataset
            return luigi.contrib.gcs.GCSTarget(f"{self.output_path}/data/_SUCCESS")
        else:
            return luigi.contrib.gcs.GCSTarget(
                f"{self.output_path}/data/sample_id={self.sample_id}/_SUCCESS"
            )

    @property
    def feature_columns(self) -> list:
        raise NotImplementedError()

    def pipeline(self) -> Pipeline:
        raise NotImplementedError()

    def transform(self, model, df, features) -> DataFrame:
        transformed = model.transform(df)

        if self.sample_id is not None:
            transformed = (
                transformed.withColumn(
                    "sample_id",
                    F.crc32(F.col(self.sample_col).cast("string")) % self.num_sample_id,
                )
                .where(F.col("sample_id") == self.sample_id)
                .drop("sample_id")
            )

        for c in features:
            # check if the feature is a vector and convert it to an array
            if "array" in transformed.schema[c].simpleString():
                continue
            transformed = transformed.withColumn(c, vector_to_array(F.col(c)))
        return transformed

    def run(self):
        with spark_resource(
            **{"spark.sql.shuffle.partitions": self.num_partitions}
        ) as spark:
            df = spark.read.parquet(self.input_path)

            model = self.pipeline().fit(df)
            transformed = self.transform(model, df, self.feature_columns)

            if self.sample_id is None:
                output_path = f"{self.output_path}/data"
            else:
                output_path = f"{self.output_path}/data/sample_id={self.sample_id}"

            transformed.repartition(self.num_partitions).write.mode(
                "overwrite"
            ).parquet(output_path)


class ProcessEmbeddings(ProcessBase):
    sql_statement = luigi.Parameter()

    @property
    def feature_columns(self) -> list:
        return ["sigmoid_logits"]

    def pipeline(self):
        transform = TransformEmbedding(input_col="embedding", output_col="embedding")
        return Pipeline(
            stages=[transform, SQLTransformer(statement=self.sql_statement)]
        )


class TrainClassifier(luigi.Task):
    input_path = luigi.Parameter()
    default_root_dir = luigi.Parameter()
    label_col = luigi.Parameter()
    feature_col = luigi.Parameter()
    loss = luigi.Parameter()
    model = luigi.Parameter()
    hidden_layer_size = luigi.OptionalIntParameter(default=64)
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
            num_classes = int(
                len(data_module.train_data.select("label").first()["label"])
            )

            # model module
            if self.two_layer:
                model = self.model(
                    num_features, num_classes, self.loss, self.hidden_layer_size
                )
            else:
                model = self.model(num_features, num_classes, self.loss)

            # initialise the wandb logger and name your wandb project
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


class Workflow(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    default_root_dir = luigi.Parameter()

    def run(self):
        # training workflow parameters
        train_model = True
        sample_col = "sigmoid_logits"
        sql_statement = "SELECT id, sigmoid_logits, embedding FROM __THIS__"
        # process bird embeddings
        yield [
            ProcessEmbeddings(
                input_path=self.input_path,
                output_path=self.output_path,
                sample_id=i,
                num_sample_id=10,
                sample_col=sample_col,
                sql_statement=sql_statement,
            )
            for i in range(10)
        ]

        # train classifier
        if train_model:
            label_col, feature_col = "sigmoid_logits", "embedding"
            # Parameters
            layer_params = {
                "linear": LinearClassifier,
                "two_layer": TwoLayerClassifier,
            }
            loss_params = {
                "bce": nn.BCEWithLogitsLoss,
                "asl": ASLSingleLabel,
                "sigmoid_f1": SigmoidF1,
            }
            hidden_layers = [64, 128, 256]

            # Linear model grid search
            linear_grid_search = list(itertools.product(["linear"], loss_params.keys()))
            for model_name, loss_name in linear_grid_search:
                model = layer_params[model_name]
                loss = loss_params[loss_name]
                yield TrainClassifier(
                    input_path=self.output_path,
                    default_root_dir=f"{self.default_root_dir}-{model_name}-{loss_name}",
                    label_col=label_col,
                    feature_col=feature_col,
                    loss=loss,
                    model=model,
                )

            # TwoLayer model grid search
            two_layer_grid_search = itertools.product(
                ["two_layer"],
                ["bce"],  # Only BCE loss for initial search
                hidden_layers,
            )
            for model_name, loss_name, hidden_layer_size in two_layer_grid_search:
                model = layer_params[model_name]
                loss = loss_params[loss_name]
                yield TrainClassifier(
                    input_path=self.output_path,
                    default_root_dir=f"{self.default_root_dir}-{model_name}-{loss_name}",
                    label_col=label_col,
                    feature_col=feature_col,
                    loss=loss,
                    model=model,
                    hidden_layer_size=hidden_layer_size,
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
        default="data/intermediate/google_embeddings/v1",
        help="Root directory for training data in GCS",
    )
    parser.add_argument(
        "--output-name-path",
        type=str,
        default="data/intermediate/google_embeddings/v1-transformed",
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
    )
