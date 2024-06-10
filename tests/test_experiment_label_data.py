from birdclef.experiment.label.data import (
    GoogleVocalizationSoundscapeDataModule,
    GoogleVocalizationSoundscapeDataset,
)


def test_google_vocalization_soundscape_dataset(metadata_path, soundscape_path):
    dataset = GoogleVocalizationSoundscapeDataset(soundscape_path, metadata_path)
    row = next(iter(dataset))
    assert row.keys() == {"row_id", "embedding", "logits"}
    assert row["embedding"].shape == (1280,)
    assert row["logits"].shape == (3,)


def test_google_vocalization_soundscape_data_module(metadata_path, soundscape_path):
    data_module = GoogleVocalizationSoundscapeDataModule(
        soundscape_path,
        metadata_path,
        batch_size=4,
    )
    data_module.setup()
    # check that our batch size is correct
    batch = next(iter(data_module.predict_dataloader()))
    assert batch.keys() == {"row_id", "embedding", "logits"}
    assert batch["embedding"].shape == (4, 1280)
    assert batch["logits"].shape == (4, 3)
