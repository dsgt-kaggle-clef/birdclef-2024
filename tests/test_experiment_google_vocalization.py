import itertools
import random

import lightning as pl
import pytest
import torch
from pyspark.sql import Row
from pyspark.sql.types import ArrayType, FloatType, StringType, StructField, StructType

from birdclef.config import SPECIES
from birdclef.experiment.google_vocalization.data import PetastormDataModule
from birdclef.experiment.model import LinearClassifier, TwoLayerClassifier
from birdclef.utils import get_spark


def test_gpu_available():
    assert torch.cuda.is_available(), "CUDA is not available"
    assert torch.cuda.device_count() > 0, "No CUDA devices found"
    assert torch.cuda.get_device_name(0) != "", "CUDA device name is empty"


# Function to create a mock Spark DataFrame
# Create a mock Spark DataFrame
@pytest.fixture(scope="session")
def temp_spark_data_path(spark, tmp_path_factory):
    spark = get_spark()
    data = [
        Row(
            features=[float(i) for i in range(10)],
            label=[float(random.randint(0, 1)) for _ in range(10)],
            name=f"{random.choice(SPECIES[:10])}/XC123456.ogg",  # use only first 10 species
        )
        for _ in range(10)
    ]
    schema = StructType(
        [
            StructField("embeddings", ArrayType(FloatType()), False),
            StructField("species_name", ArrayType(FloatType()), False),
            StructField("name", StringType(), False),
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
    data_module = PetastormDataModule(
        spark=spark,
        input_path=temp_spark_data_path,
        feature_col="embeddings",
        label_col="species_name",
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


# run this both gpu and cpu, but only the gpu if it's available
# pytest parametrize
@pytest.mark.parametrize(
    "device, loss, species_label",
    # itertools.product(["cpu", "gpu"], ["bce", "asl", "sigmoidf1"], [False, True]),
    itertools.product(["cpu", "gpu"], ["asl"], [False, True]),
)
def test_linear_torch_model(spark, temp_spark_data_path, device, loss, species_label):
    if device == "gpu" and not torch.cuda.is_available():
        pytest.skip()

    data_module = PetastormDataModule(
        spark=spark,
        input_path=temp_spark_data_path,
        feature_col="embeddings",
        label_col="species_name",
    )
    data_module.setup()

    # get parameters for the model
    num_features = int(
        len(data_module.train_data.select("features").first()["features"])
    )
    num_labels = int(len(data_module.train_data.select("label").first()["label"]))

    # test losses
    model = LinearClassifier(
        num_features, num_labels, loss=loss, species_label=species_label
    )

    trainer = pl.Trainer(
        accelerator=device,
        default_root_dir=temp_spark_data_path,
        fast_dev_run=True,
    )
    trainer.fit(model, data_module)


# run this both gpu and cpu, but only the gpu if it's available
# pytest parametrize
@pytest.mark.parametrize(
    "device, loss, hp_kwargs",
    itertools.product(
        ["cpu", "gpu"],
        ["asl"],
        [{"gamma_neg": 0, "gamma_pos": 0}, {"gamma_neg": 2, "gamma_pos": 1}],
    ),
)
def test_two_layer_torch_model(spark, temp_spark_data_path, device, loss, hp_kwargs):
    if device == "gpu" and not torch.cuda.is_available():
        pytest.skip()
    # Params
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

    # get parameters for the model
    num_features = int(
        len(data_module.train_data.select("features").first()["features"])
    )
    num_labels = int(len(data_module.train_data.select("label").first()["label"]))

    # test losses
    model = TwoLayerClassifier(
        num_features, num_labels, loss=loss, hidden_layer_size=64, hp_kwargs=hp_kwargs
    )

    trainer = pl.Trainer(
        accelerator=device,
        default_root_dir=temp_spark_data_path,
        fast_dev_run=True,
    )
    trainer.fit(model, data_module)
