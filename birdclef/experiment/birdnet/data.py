import math
import os
import warnings
from pathlib import Path

import lightning as pl
import numpy as np
import torch
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pyspark.sql import functions as F
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import v2

from birdclef.config import SPECIES
from birdclef.inference.birdnet import BirdNetInference


class BirdNetSoundscapeDataset(IterableDataset):
    """Dataset meant for inference on soundscape data."""

    def __init__(
        self,
        soundscape_path: str,
        metadata_path: str,
        max_length: int = 4 * 60 / 5,
        limit=None,
    ):
        """Initialize the dataset.

        :param soundscape_path: The path to the soundscape data.
        :param metadata_path: The path to the metadata.
        :param max_length: The maximum length of the soundscape.
        :param limit: The number of files to limit the dataset to.
        """
        self.soundscapes = sorted(Path(soundscape_path).glob("**/*.ogg"))
        self.max_length = int(max_length)
        if limit is not None:
            self.soundscapes = self.soundscapes[:limit]
        self.metadata_path = metadata_path

    def _load_data(self, iter_start, iter_end):
        model = BirdNetInference(self.metadata_path)
        for i in range(iter_start, iter_end):
            path = self.soundscapes[i]
            embeddings, _ = model.predict(path)
            embeddings = embeddings[: self.max_length].float()
            n_chunks = embeddings.shape[0]
            indices = range(n_chunks)

            # now we yield a dictionary
            for idx, embedding in zip(indices, embeddings):
                yield {
                    "row_id": f"{path.stem}_{(idx+1)*5}",
                    "embedding": embedding,
                }

    def __iter__(self):
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        worker_info = torch.utils.data.get_worker_info()
        start, end = 0, len(self.soundscapes)
        if worker_info is None:
            iter_start = start
            iter_end = end
        else:
            per_worker = int(math.ceil((end - start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, end)

        return self._load_data(iter_start, iter_end)


class BirdNetSoundscapeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        soundscape_path: str,
        metadata_path: str,
        batch_size: int = 32,
        num_workers: int = 0,
        limit=None,
    ):
        """Initialize the data module.

        :param soundscape_path: The path to the soundscape data.
        :param metadata_path: The path to the metadata.
        :param model_path: The path to the model.
        :param batch_size: The batch size.
        :param limit: The number of files to limit the dataset to.
        """
        super().__init__()
        self.dataloader = DataLoader(
            BirdNetSoundscapeDataset(
                soundscape_path,
                metadata_path,
                limit=limit,
            ),
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def predict_dataloader(self):
        return self.dataloader


# create a transform to convert a list of numbers into a sparse tensor
class ToSparseTensor(v2.Transform):
    def forward(self, batch):
        if "label" in batch:
            batch["label"] = batch["label"].to_sparse()
        return batch


class PetastormDataModule(pl.LightningDataModule):
    def __init__(
        self,
        spark,
        input_birdnet_path,
        label_col,
        feature_col,
        species_label=True,
        # optional: will be used to join if it exists
        input_google_path=None,
        batch_size=64,
        num_partitions=os.cpu_count(),
        workers_count=os.cpu_count(),
    ):
        super().__init__()
        cache_dir = "file:///mnt/data/tmp"
        spark.conf.set(
            SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, Path(cache_dir).as_posix()
        )
        self.spark = spark
        self.input_birdnet_path = input_birdnet_path
        self.input_google_path = input_google_path
        self.label_col = label_col
        self.feature_col = feature_col
        self.species_label = species_label
        self.batch_size = batch_size
        self.num_partitions = num_partitions
        self.workers_count = workers_count

    def _prepare_dataframe(self, df):
        """Prepare the DataFrame for training by ensuring correct types and repartitioning"""

        @F.udf("array<boolean>")
        def sigmoid_udf(x):
            z = 1 / (1 + np.exp(-np.array(x)))
            return (z > 0.5).tolist()

        return df.select(
            F.col(self.feature_col).cast("array<float>").alias("features"),
            sigmoid_udf(self.label_col).cast("array<short>").alias("label"),
            *([F.col("species_index")] if self.species_label else []),
        ).repartition(self.num_partitions)

    def _train_valid_split(self, df):
        """
        Perform train/valid random split
        :return: train_df, valid_df Spark DataFrames
        """
        train_df, valid_df = df.randomSplit([0.8, 0.2], seed=42)
        train_df = self._prepare_dataframe(train_df)
        valid_df = self._prepare_dataframe(valid_df)
        return train_df.cache(), valid_df.cache()

    def setup(self, stage=None):
        """Setup dataframe for petastorm spark converter"""
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)

            # UDF to retrieve species index
            @F.udf("integer")
            def species_index_udf(name):
                species = Path(name).parent.name
                species_idx = SPECIES.index(species)
                return species_idx

            # read data
            birdnet_df = self.spark.read.parquet(self.input_birdnet_path).cache()
            google_df = self.spark.read.parquet(self.input_google_path).cache()
            df = birdnet_df.join(
                google_df.select("name", "chunk_5s", self.label_col),
                on=["name", "chunk_5s"],
                how="inner",
            )
            if self.species_label:
                df = df.withColumn("species_index", species_index_udf("name"))

            # train/valid Split
            self.train_data, self.valid_data = self._train_valid_split(df=df)
            print(
                f"\ntrain count: {self.train_data.count()}\n"
                f"valid count: {self.valid_data.count()}\n",
            )

            # setup petastorm data conversion from Spark to PyTorch
            self.converter_train = make_spark_converter(self.train_data)
            self.converter_valid = make_spark_converter(self.valid_data)

    def _dataloader(self, converter):
        # transform = v2.Compose([ToSparseTensor()])
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            with converter.make_torch_dataloader(
                batch_size=self.batch_size,
                num_epochs=1,
                workers_count=self.workers_count,
            ) as dataloader:
                for batch in dataloader:
                    yield batch
        torch.cuda.empty_cache()

    def train_dataloader(self):
        for batch in self._dataloader(self.converter_train):
            yield batch

    def val_dataloader(self):
        for batch in self._dataloader(self.converter_valid):
            yield batch
