import numpy as np
import pandas as pd
import pytest
import torch
import torchaudio

from birdclef.label.data import GoogleVocalizationSoundscapeDataset


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


@pytest.fixture
def soundscape_path(tmp_path):
    """Subset of the metadata for testing."""
    soundscape = tmp_path / "soundscape"
    soundscape.mkdir()
    for i in range(3):
        filename = f"file_{i}.ogg"
        sr = 32_000
        # create 2 channel audio from random noise
        y = torch.tensor(np.random.rand(2 * 10 * sr), dtype=torch.float32).reshape(
            2, -1
        )
        torchaudio.save(str(soundscape / filename), y, sr)
    return soundscape


def test_google_vocalization_soundscape_dataset(metadata_path, soundscape_path):
    dataset = GoogleVocalizationSoundscapeDataset(soundscape_path, metadata_path)
    row = next(iter(dataset))
    assert row.keys() == {"row_id", "embedding", "logits"}
    assert row["embedding"].shape == (1280,)
    assert row["logits"].shape == (1,)
