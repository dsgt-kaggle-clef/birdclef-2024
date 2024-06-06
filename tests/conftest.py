import numpy as np
import pytest

from birdclef.utils import spark_resource


@pytest.fixture(scope="session")  # Change scope to "session"
def spark(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("spark_data")
    with spark_resource(local_dir=tmp_path.as_posix()) as spark:
        yield spark
