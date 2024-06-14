import pandas as pd

from birdclef.inference.birdnet import BirdNetInference


def test_birdnet_inference_init(metadata_path):
    bi = BirdNetInference(metadata_path)
    assert bi is not None


def test_birdnet_inference_predict(metadata_path):
    bi = BirdNetInference(metadata_path)
    embedding, _ = bi.predict(metadata_path.parent / "file_0.ogg")
    # 10 seconds of audio means there are 10 rows
    assert embedding.shape == (2, 1024)


def test_birdnet_inference_predict_df(metadata_path):
    bi = BirdNetInference(metadata_path)
    df = bi.predict_df(metadata_path.parent, "file_0.ogg")
    assert len(df) == 2
    assert set(df.columns) == {"name", "chunk_5s", "embedding"}


def test_birdnet_inference_predict_species_df(metadata_path):
    bi = BirdNetInference(metadata_path)
    out_path = metadata_path.parent / "file_0.parquet"
    df = bi.predict_species_df(metadata_path.parent, "asbfly", out_path)
    assert len(df) == 2
    assert set(df.columns) == {"name", "chunk_5s", "embedding"}
    assert out_path.exists()
