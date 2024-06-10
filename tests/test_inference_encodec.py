import pandas as pd

from birdclef.inference.encodec import EncodecInference


def test_encodec_inference_init(metadata_path):
    ei = EncodecInference(metadata_path)
    assert ei is not None
    assert isinstance(ei.metadata, pd.DataFrame)
    assert isinstance(ei.chunk_size, int)


def test_encodec_inference_predict(metadata_path):
    ei = EncodecInference(metadata_path)
    embedding, _ = ei.predict(metadata_path.parent / "file_0.ogg")
    # 10 seconds of audio means there are 10 rows
    assert embedding.shape == (2, 150 * 5)


def test_encodec_inference_predict_df(metadata_path):
    ei = EncodecInference(metadata_path)
    df = ei.predict_df(metadata_path.parent, "file_0.ogg")
    assert len(df) == 2
    assert set(df.columns) == {"name", "chunk_5s", "embedding"}


def test_encodec_inference_predict_species_df(metadata_path):
    ei = EncodecInference(metadata_path)
    out_path = metadata_path.parent / "file_0.parquet"
    df = ei.predict_species_df(metadata_path.parent, "asbfly", out_path)
    assert len(df) == 2
    assert set(df.columns) == {"name", "chunk_5s", "embedding"}
    assert out_path.exists()
