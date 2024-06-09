import pytest
import pytorch_lightning as pl
from pyspark.sql import Row
from pyspark.sql.types import ArrayType, FloatType, LongType, StructField, StructType

from birdclef.experiment.google_vocalization.data import PetastormDataModule
from birdclef.experiment.google_vocalization.model import LinearClassifier
from birdclef.utils import get_spark


# Function to create a mock Spark DataFrame
# Create a mock Spark DataFrame
@pytest.fixture(scope="session")
def temp_spark_data_path(spark, tmp_path_factory):
    spark = get_spark()
    data = [Row(features=[float(i) for i in range(10)], label=i % 2) for i in range(10)]
    schema = StructType(
        [
            StructField("embeddings", ArrayType(FloatType()), False),
            StructField("species_name", LongType(), False),
        ]
    )
    # create DF
    df = spark.createDataFrame(data, schema=schema)
    temp_path = tmp_path_factory.mktemp("spark_data")
    temp_path_posix = temp_path.as_posix()
    df.write.mode("overwrite").parquet(temp_path_posix)
    return temp_path_posix


# Test Function
def test_petastorm_data_module_setup(spark, temp_spark_data_path):
    input_path = temp_spark_data_path
    feature_col = "embeddings"
    label_col = "species_name"

    data_module = PetastormDataModule(
        spark=spark,
        input_path=input_path,
        feature_col=feature_col,
        label_col=label_col,
    )
    data_module.setup()
    print(f"train data shape: {data_module.train_data.count()}")
    print(f"valid data shape: {data_module.valid_data.count()}")
    # Assertions
    assert data_module.train_data is not None
    assert data_module.valid_data is not None
    assert data_module.converter_train is not None
    assert data_module.converter_valid is not None
    assert data_module.train_data.count() > 0
    assert data_module.valid_data.count() > 0
