from unittest.mock import MagicMock, patch

import pytest
import pytorch_lightning as pl
from pyspark.sql import Row
from pyspark.sql.types import ArrayType, FloatType, LongType, StructField, StructType

from birdclef.experiment.google_vocalization.data import PetastormDataModule
from birdclef.experiment.google_vocalization.model import LinearClassifier
from birdclef.utils import get_spark


# Function to create a mock Spark DataFrame
def create_mock_dataframe(spark, num_rows=10):
    data = [
        Row(features=[float(i) for i in range(10)], label=i % 2)
        for i in range(num_rows)
    ]
    schema = StructType(
        [
            StructField("features", ArrayType(FloatType()), False),
            StructField("label", LongType(), False),
        ]
    )
    return spark.createDataFrame(data, schema=schema)


# Mock the spark session and DataFrame
@pytest.fixture(scope="session")
def spark():
    spark = get_spark()
    return spark


# test dataframe
@pytest.fixture(scope="session")
def mock_df(spark):
    return create_mock_dataframe(spark)


def test_petastorm_data_module_setup(spark, mock_df):
    input_path = "mack_path"
    feature_col = "features"

    # reading the dataframe
    spark.read.parquet = MagicMock(return_value=mock_df)

    # init the DataModule
    data_module = PetastormDataModule(
        spark=spark,
        input_path=input_path,
        feature_col=feature_col,
    )

    # run setup
    data_module.setup()

    # assertions
    assert data_module.train_data is not None
    assert data_module.valid_data is not None
    assert data_module.converter_train is not None
    assert data_module.converter_valid is not None
    assert data_module.train_data.count() > 0
    assert data_module.valid_data.count() > 0

    print(f"train_data count: {data_module.train_data.count()}")
    print(f"valid_data count: {data_module.valid_data.count()}")
