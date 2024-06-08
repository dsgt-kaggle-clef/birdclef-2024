import os
from pathlib import Path

import pytorch_lightning as pl
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pyspark.sql import functions as F


class PetastormDataModule(pl.LightningDataModule):
    def __init__(
        self,
        spark,
        input_path,
        feature_col,
        label_col,
        batch_size=64,
        num_partitions=32,
        workers_count=4,  # os.cpu_count(),
    ):
        super().__init__()
        cache_dir = "file:///mnt/data/tmp"
        spark.conf.set(
            SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, Path(cache_dir).as_posix()
        )
        self.spark = spark
        self.input_path = input_path
        self.feature_col = feature_col
        self.label_col = label_col
        self.batch_size = batch_size
        self.num_partitions = num_partitions
        self.workers_count = workers_count

    def _prepare_data(self):
        """
        Prepare data for petastorm loading.
        :return: DataFrame of filtered and indexed species data
        """
        df = self.spark.read.parquet(self.input_path).cache()
        return df

    def _prepare_dataframe(self, df, partitions=32):
        """Prepare the DataFrame for training by ensuring correct types and repartitioning"""
        return (
            df.withColumnRenamed(self.feature_col, "features")
            .withColumnRenamed(self.label_col, "label")
            .select(
                F.col("features").cast("array<float>").alias("features"),
                F.col("label").cast("array<float>").alias("label"),
            )
            .repartition(partitions)
        )

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
        # Get prepared data
        prepared_df = self._prepare_data()
        # train/valid Split
        self.train_data, self.valid_data = self._train_valid_split(df=prepared_df)

        # setup petastorm data conversion from Spark to PyTorch
        self.converter_train = make_spark_converter(self.train_data)
        self.converter_valid = make_spark_converter(self.valid_data)

    def _dataloader(self, converter):
        with converter.make_torch_dataloader(
            batch_size=self.batch_size,
            num_epochs=1,
            workers_count=self.workers_count,
        ) as dataloader:
            for batch in dataloader:
                yield batch

    def train_dataloader(self):
        for batch in self._dataloader(self.converter_train):
            yield batch

    def val_dataloader(self):
        for batch in self._dataloader(self.converter_valid):
            yield batch
