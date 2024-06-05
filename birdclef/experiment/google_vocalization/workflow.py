from argparse import ArgumentParser

import luigi
import luigi.contrib.gcs
from pyspark.ml import Pipeline
from pyspark.ml.feature import SQLTransformer
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from birdclef.utils import spark_resource

from .classifier import TrainClassifier


class Workflow(luigi.Task):
    input_path = luigi.Parameter()
    default_root_dir = luigi.Parameter()

    def run(self):
        # Train classifier
        input_path = self.input_path
        feature_col, label_col = "embedding", "name"
        final_default_dir = self.default_root_dir
        two_layer = False
        # train model
        yield TrainClassifier(
            input_path=input_path,
            feature_col=feature_col,
            label_col=label_col,
            default_root_dir=final_default_dir,
            two_layer=two_layer,
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
        default="data/processed/birdclef-2024/asbfly.parquet",
        help="Root directory for training data in GCS",
    )
    parser.add_argument(
        "--model-dir-path",
        type=str,
        default="models/torch-petastorm-v1",
        help="Default root directory for storing the pytorch classifier runs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Input and output paths for training workflow
    input_path = f"{args.gcs_root_path}/{args.train_data_path}"
    default_root_dir = f"{args.gcs_root_path}/{args.model_dir_path}"

    luigi.build(
        [
            Workflow(
                input_path=input_path,
                default_root_dir=default_root_dir,
            )
        ],
        scheduler_host="services.us-central1-a.c.dsgt-clef-2024.internal",
    )
