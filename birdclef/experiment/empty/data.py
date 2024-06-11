from pathlib import Path

import lightning as pl
import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, IterableDataset


class SoundscapeDataset(IterableDataset):
    """Dataset meant for inference on soundscape data."""

    def __init__(
        self,
        soundscape_path: str,
        limit=None,
    ):
        """Initialize the dataset.

        :param soundscape_path: The path to the soundscape data.
        :param metadata_path: The path to the metadata.
        :param model_path: The path to the model.
        """
        self.soundscapes = sorted(Path(soundscape_path).glob("**/*.ogg"))
        if limit is not None:
            self.soundscapes = self.soundscapes[:limit]

    def load(self, path: str, window: int = 5 * 32_000) -> np.ndarray:
        """Load an audio file.

        :param path: The absolute path to the audio file.
        """
        audio, _ = torchaudio.load(path)
        audio = audio.squeeze()
        # right pad the audio so we can reshape into a rectangle
        n = audio.shape[0]
        if n % window != 0:
            audio = audio[: n - (n % window)]
        # reshape the audio into windowsize chunks
        audio = audio.reshape(-1, window)
        return audio

    def _load_data(self, iter_start, iter_end):
        for i in range(iter_start, iter_end):
            path = self.soundscapes[i]
            audio = self.load(path)
            n_chunks = audio.shape[0]
            indices = range(n_chunks)

            # now we yield a dictionary
            for idx, audio_row in zip(indices, audio):
                yield {"row_id": f"{path.stem}_{(idx+1)*5}", "audio": audio_row}

    def __iter__(self):
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.soundscapes)
        else:
            per_worker = int(len(self.soundscapes) / worker_info.num_workers)
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.soundscapes))

        return self._load_data(iter_start, iter_end)


class SoundscapeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        soundscape_path: str,
        batch_size: int = 32,
        num_workers: int = 0,
        limit=None,
    ):
        """Initialize the data module.

        :param soundscape_path: The path to the soundscape data.
        :param batch_size: The batch size.
        :param use_compiled: Whether to use the compiled model.
        :param num_workers: The number of workers.
        :param limit: The number of files to limit the dataset to.
        """
        super().__init__()
        self.dataloader = DataLoader(
            SoundscapeDataset(soundscape_path, limit),
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def predict_dataloader(self):
        return self.dataloader
