import math
from pathlib import Path

import lightning as pl
import torch
from torch.utils.data import DataLoader, IterableDataset

from birdclef.inference.birdnet import BirdNetInference


class BirdNetSoundscapeDataset(IterableDataset):
    """Dataset meant for inference on soundscape data."""

    def __init__(
        self,
        soundscape_path: str,
        metadata_path: str,
        max_length: int = 4 * 60 / 5,
        limit=None,
    ):
        """Initialize the dataset.

        :param soundscape_path: The path to the soundscape data.
        :param metadata_path: The path to the metadata.
        :param max_length: The maximum length of the soundscape.
        :param limit: The number of files to limit the dataset to.
        """
        self.soundscapes = sorted(Path(soundscape_path).glob("**/*.ogg"))
        self.max_length = int(max_length)
        if limit is not None:
            self.soundscapes = self.soundscapes[:limit]
        self.metadata_path = metadata_path

    def _load_data(self, iter_start, iter_end):
        model = BirdNetInference(self.metadata_path)
        for i in range(iter_start, iter_end):
            path = self.soundscapes[i]
            embeddings, _ = model.predict(path)
            embeddings = embeddings[: self.max_length].float()
            n_chunks = embeddings.shape[0]
            indices = range(n_chunks)

            # now we yield a dictionary
            for idx, embedding in zip(indices, embeddings):
                yield {
                    "row_id": f"{path.stem}_{(idx+1)*5}",
                    "embedding": embedding,
                }

    def __iter__(self):
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        worker_info = torch.utils.data.get_worker_info()
        start, end = 0, len(self.soundscapes)
        if worker_info is None:
            iter_start = start
            iter_end = end
        else:
            per_worker = int(math.ceil((end - start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, end)

        return self._load_data(iter_start, iter_end)


class BirdNetSoundscapeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        soundscape_path: str,
        metadata_path: str,
        batch_size: int = 32,
        num_workers: int = 0,
        limit=None,
    ):
        """Initialize the data module.

        :param soundscape_path: The path to the soundscape data.
        :param metadata_path: The path to the metadata.
        :param model_path: The path to the model.
        :param batch_size: The batch size.
        :param limit: The number of files to limit the dataset to.
        """
        super().__init__()
        self.dataloader = DataLoader(
            BirdNetSoundscapeDataset(
                soundscape_path,
                metadata_path,
                limit=limit,
            ),
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def predict_dataloader(self):
        return self.dataloader
