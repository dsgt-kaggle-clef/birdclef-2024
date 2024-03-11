import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import luigi
import numpy as np
import pandas as pd
from luigi.parameter import ParameterVisibility
from pynndescent import NNDescent
from pyspark.sql import Window
from pyspark.sql import functions as F
from tqdm import tqdm

from birdclef.utils import spark_resource


class BirdNetAnalyzeConcatTask(luigi.Task):
    taxonomy_path = luigi.Parameter()
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    parallelism = luigi.IntParameter(default=os.cpu_count(), significant=False)
    dynamic_requires = luigi.Parameter(
        default=[], visibility=ParameterVisibility.HIDDEN
    )

    def requires(self):
        return self.dynamic_requires

    def output(self):
        return luigi.LocalTarget(f"{self.output_path}/birdnet-analyze.parquet")

    def read_path(self, path: Path, lookup=lambda x: x) -> pd.DataFrame:
        df = pd.read_csv(path)
        name = path.name.split(".")[0]
        df["filename"] = f"{path.parent.name}/{name}.ogg"
        df["birdnet_label"] = df["Common name"].apply(lambda x: lookup(x, "unknown"))
        df["birdnet_common_name"] = df["Common name"]
        return df

    def run(self):
        taxonomy_df = pd.read_csv(self.taxonomy_path)
        common_to_code = dict(
            list(zip(taxonomy_df["PRIMARY_COM_NAME"], taxonomy_df["SPECIES_CODE"]))
        )

        root = Path(self.input_path)
        paths = sorted(root.glob("**/*.csv"))
        with Pool(self.parallelism) as pool:
            res = list(
                tqdm(
                    pool.imap(
                        partial(self.read_path, lookup=common_to_code.get), paths
                    ),
                    total=len(paths),
                )
            )
        df = pd.concat(res).rename(
            columns={
                "Start (s)": "start_sec",
                "End (s)": "end_sec",
                "Confidence": "confidence",
            }
        )[
            [
                "start_sec",
                "end_sec",
                "confidence",
                "birdnet_label",
                "birdnet_common_name",
                "filename",
            ]
        ]
        print(df.head())
        # ensure the output directory exists
        output_path = Path(self.output().path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)


class BirdNetEmbeddingsConcatTask(luigi.Task):
    """Concatenate embeddings into a single file using PySpark."""

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    parallelism = luigi.IntParameter(default=os.cpu_count(), significant=False)
    partitions = luigi.IntParameter(default=8, significant=False)
    dynamic_requires = luigi.Parameter(
        default=[], visibility=ParameterVisibility.HIDDEN
    )

    def requires(self):
        return self.dynamic_requires

    def output(self):
        return luigi.LocalTarget(f"{self.output_path}/birdnet-embeddings.parquet")

    def run(self):
        def format_filename(path: str) -> str:
            species, name = Path(path).parts[-2:]
            # name is like {ID}.birdnet.embeddings.txt want {species}/{ID}.ogg
            id = name.split(".")[0]
            return f"{species}/{id}.ogg"

        def parse_emb_text(emb_text: str) -> np.ndarray:
            return np.fromstring(emb_text, dtype=float, sep=",").tolist()

        with spark_resource(cores=self.parallelism) as spark:
            df = (
                spark.read.csv(
                    f"{self.input_path}/*/*.txt",
                    sep="\t",
                    schema="start_sec FLOAT, end_sec FLOAT, emb_text STRING",
                )
                .withColumn(
                    "filename",
                    F.udf(format_filename, "string")(F.input_file_name()),
                )
                .select(
                    "start_sec",
                    "end_sec",
                    "filename",
                    F.udf(parse_emb_text, "array<float>")("emb_text").alias("emb"),
                )
                .orderBy("filename", "start_sec")
            )
            df.printSchema()
            df.show(3, vertical=True)
            df.toPandas().to_parquet(self.output().path, index=False)


class BirdNetEmbeddingsWithNeighbors(luigi.Task):
    """Combine all metadata into a single file, with nearest neighbors."""

    birdnet_analyze_path = luigi.Parameter()
    birdnet_embeddings_path = luigi.Parameter()
    train_metadata_path = luigi.Parameter()
    output_path = luigi.Parameter()
    train_n_neighbors = luigi.IntParameter(default=30)
    query_n_neighbors = luigi.IntParameter(default=50)

    partitions = luigi.IntParameter(default=16, significant=False)
    parallelism = luigi.IntParameter(default=os.cpu_count(), significant=False)

    dynamic_requires = luigi.Parameter(
        default=[], visibility=ParameterVisibility.HIDDEN
    )

    def requires(self):
        return self.dynamic_requires

    def output(self):
        return luigi.LocalTarget(
            f"{self.output_path}/birdnet-embeddings-with-neighbors/v1"
        )

    def run(self):
        with spark_resource(cores=self.parallelism) as spark:
            birdnet_analyze = spark.read.parquet(self.birdnet_analyze_path)
            birdnet_embeddings = spark.read.parquet(self.birdnet_embeddings_path)
            train_metadata = spark.read.csv(self.train_metadata_path, header=True)

            joined = (
                birdnet_analyze.join(
                    birdnet_embeddings, on=["filename", "start_sec", "end_sec"]
                )
                .join(
                    train_metadata.select(
                        "filename", "primary_label", "secondary_labels", "type"
                    ),
                    on="filename",
                )
                .orderBy("filename", "start_sec")
                # we want to ensure that we can correspond between neighbors
                # found by nn-descent and the entries here, though bringing
                # everything into the driver is generally bad.
                .withColumn(
                    "id", F.row_number().over(Window.orderBy("filename", "start_sec"))
                )
            ).cache()

            X = np.stack(joined.select("emb").toPandas().emb)
            index = NNDescent(X, n_neighbors=self.train_n_neighbors, verbose=True)
            neighbors, distances = index.query(X, k=self.query_n_neighbors)
            query_df = pd.DataFrame(
                dict(
                    id=joined.select("id").toPandas().id,
                    neighbors=neighbors.tolist(),
                    distances=distances.tolist(),
                )
            )

            joined = joined.join(spark.createDataFrame(query_df), on="id")

            joined.printSchema()

            # we're not going to be using this dataset in pandas, so let's just
            # write it out partitioned
            joined.repartition(self.partitions).write.parquet(self.output().path)
