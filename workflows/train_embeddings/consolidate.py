import os
from argparse import ArgumentParser
from pathlib import Path

import luigi

from birdclef.utils import spark_resource
from workflows.utils.gcs import single_file_target
from workflows.utils.rsync import GSUtilRsyncTask


class ExtractEmbedding(luigi.Task):
    input_path = luigi.Parameter()
    output_name = luigi.Parameter()
    parallelism = luigi.IntParameter(default=2)
    partitions = luigi.IntParameter(default=32)

    def output(self):
        return single_file_target(Path(self.input_path) / self.output_name / "_SUCCESS")

    def run(self):
        root = self.input_path
        with spark_resource(self.parallelism, memory="16g") as spark:
            df = spark.read.parquet(f"{root}/embeddings/*/*/*.parquet")
            df.repartition(self.partitions).write.parquet(
                f"{root}/{self.output_name}", mode="overwrite"
            )


class ConsolidateEmbeddings(luigi.Task):
    input_path = luigi.Parameter()
    remote_path = luigi.Parameter()
    output_name = luigi.Parameter()
    parallelism = luigi.IntParameter(default=2)
    partitions = luigi.IntParameter(default=32)

    def run(self):
        emb = yield ExtractEmbedding(
            input_path=self.input_path,
            output_name=self.output_name,
            parallelism=self.parallelism,
            partitions=self.partitions,
        )
        yield GSUtilRsyncTask(
            input_path=f"{self.input_path}/{self.output_name}",
            output_path=f"{self.remote_path}/{self.output_name}",
            is_dir=True,
            touch_success=False,
            dynamic_requires=[emb],
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/processed/birdclef-2023/train_embeddings",
    )
    parser.add_argument(
        "--remote-path",
        type=str,
        default="gs://birdclef-2023/data/processed/birdclef-2023/train_embeddings",
    )
    parser.add_argument("--output-name", type=str, required=True)
    parser.add_argument("--parallelism", type=int, default=os.cpu_count())
    parser.add_argument("--partitions", type=int, default=64)
    args = parser.parse_args()

    luigi.build(
        [
            ConsolidateEmbeddings(
                input_path=args.input_path,
                remote_path=args.remote_path,
                output_name=args.output_name,
                parallelism=args.parallelism,
                partitions=args.partitions,
            )
        ],
        scheduler_host="luigi.us-central1-a.c.birdclef-2023.internal",
    )
