from argparse import ArgumentParser

import luigi
import luigi.contrib.gcs
from pyspark.ml import Pipeline
from pyspark.ml.feature import SQLTransformer
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from birdclef.transforms import TransformEmbedding
from birdclef.utils import spark_resource

from .classifier import TrainClassifier


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
            model.write().overwrite().save(f"{self.output_path}/model")
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
        return ["dino_embedding"]

    def pipeline(self):
        dino = TransformEmbedding(input_col="embedding", output_col="bird_embedding")
        return Pipeline(stages=[dino, SQLTransformer(statement=self.sql_statement)])


class Workflow(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    default_root_dir = luigi.Parameter()

    def run(self):
        # training workflow parameters
        train_model = True
        sample_col = "species"
        sql_statement = "SELECT species, embedding FROM __THIS__"
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
        "--output-name-path",
        type=str,
        default="data/processed/birdclef-2024-train-embedding",
        help="GCS path for output Parquet files",
    )
    parser.add_argument(
        "--model-dir-path",
        type=str,
        default="models/torch-petastorm-v1",
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
