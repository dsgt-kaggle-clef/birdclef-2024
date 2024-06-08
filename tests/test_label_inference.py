import pandas as pd

from birdclef.label.google_vocalization.inference import GoogleVocalizationInference


def test_google_vocalization_inference_init(metadata_path):
    gvi = GoogleVocalizationInference(metadata_path)
    assert gvi is not None
    assert isinstance(gvi.metadata, pd.DataFrame)
    assert isinstance(gvi.model_labels_df, pd.DataFrame)
    assert isinstance(gvi.model_indices, list)
    assert len(gvi.model_indices) == 3


def test_google_vocalization_inference_predict(metadata_path):
    gvi = GoogleVocalizationInference(metadata_path)
    embedding, logits = gvi.predict(metadata_path.parent / "file_0.ogg")
    # 10 seconds of audio means there are 2 rows
    # and a single species means there's only a single logit
    assert embedding.shape == (2, 1280)
    assert logits.shape == (2, 3)


def test_google_vocalization_inference_predict_df(metadata_path):
    gvi = GoogleVocalizationInference(metadata_path)
    df = gvi.predict_df(metadata_path.parent, "file_0.ogg")
    assert len(df) == 2
    assert set(df.columns) == {"name", "chunk_5s", "embedding", "logits"}


def test_google_vocalization_inference_predict_species_df(metadata_path):
    gvi = GoogleVocalizationInference(metadata_path)
    out_path = metadata_path.parent / "file_0.parquet"
    df = gvi.predict_species_df(metadata_path.parent, "asbfly", out_path)
    assert len(df) == 1 * 2
    assert set(df.columns) == {"name", "chunk_5s", "embedding", "logits"}
    assert out_path.exists()
