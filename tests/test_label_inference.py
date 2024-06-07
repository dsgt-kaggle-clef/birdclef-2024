import numpy as np
import pandas as pd
import pytest
import torch
import torchaudio

from birdclef.label.inference import GoogleVocalizationInference

"""
Example metadata:
$ gcloud storage cat gs://dsgt-clef-birdclef-2024/data/raw/birdclef-2024/train_metadata.csv | head
primary_label,secondary_labels,type,latitude,longitude,scientific_name,common_name,author,license,rating,url,filename
asbfly,[],['call'],39.2297,118.1987,Muscicapa dauurica,Asian Brown Flycatcher,Matt Slaymaker,Creative Commons Attribution-NonCommercial-ShareAlike 3.0,5.0,https://www.xeno-canto.org/134896,asbfly/XC134896.ogg
asbfly,[],['song'],51.403,104.6401,Muscicapa dauurica,Asian Brown Flycatcher,Magnus Hellström,Creative Commons Attribution-NonCommercial-ShareAlike 3.0,2.5,https://www.xeno-canto.org/164848,asbfly/XC164848.ogg
asbfly,[],['song'],36.3319,127.3555,Muscicapa dauurica,Asian Brown Flycatcher,Stuart Fisher,Creative Commons Attribution-NonCommercial-ShareAlike 4.0,2.5,https://www.xeno-canto.org/175797,asbfly/XC175797.ogg
asbfly,[],['call'],21.1697,70.6005,Muscicapa dauurica,Asian Brown Flycatcher,vir joshi,Creative Commons Attribution-NonCommercial-ShareAlike 4.0,4.0,https://www.xeno-canto.org/207738,asbfly/XC207738.ogg
"""


@pytest.fixture
def metadata_path(tmp_path):
    """Subset of the metadata for testing."""
    metadata = tmp_path / "metadata.csv"
    rows = []
    for i in range(3):
        filename = f"file_{i}.ogg"
        primary_label = f"asbfly"
        rows.append(
            {
                "primary_label": primary_label,
                "filename": filename,
            }
        )
        sr = 32_000
        # create 2 channel audio from random noise
        y = torch.tensor(np.random.rand(2 * 10 * sr), dtype=torch.float32).reshape(
            2, -1
        )
        torchaudio.save(str(tmp_path / filename), y, sr)
    df = pd.DataFrame(rows)
    df.to_csv(metadata, index=False)
    return metadata


def test_google_vocalization_inference_init(metadata_path):
    gvi = GoogleVocalizationInference(metadata_path)
    assert gvi is not None
    assert isinstance(gvi.metadata, pd.DataFrame)
    assert isinstance(gvi.model_labels_df, pd.DataFrame)
    assert isinstance(gvi.model_indices, list)
    assert len(gvi.model_indices) == 1


def test_google_vocalization_inference_predict(metadata_path):
    gvi = GoogleVocalizationInference(metadata_path)
    embedding, logits = gvi.predict(metadata_path.parent / "file_0.ogg")
    # 10 seconds of audio means there are 2 rows
    # and a single species means there's only a single logit
    assert embedding.shape == (2, 1280)
    assert logits.shape == (2, 1)


def test_google_vocalization_inference_predict_df(metadata_path):
    gvi = GoogleVocalizationInference(metadata_path)
    df = gvi.predict_df(metadata_path.parent, "file_0.ogg")
    assert len(df) == 2
    assert set(df.columns) == {"name", "chunk_5s", "embedding", "logits"}


def test_google_vocalization_inference_predict_species_df(metadata_path):
    gvi = GoogleVocalizationInference(metadata_path)
    out_path = metadata_path.parent / "file_0.parquet"
    df = gvi.predict_species_df(metadata_path.parent, "asbfly", out_path)
    assert len(df) == 3 * 2
    assert set(df.columns) == {"name", "chunk_5s", "embedding", "logits"}
    assert out_path.exists()
