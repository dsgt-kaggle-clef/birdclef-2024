from pathlib import Path

import lightning as pl
import torch
from torch.utils.data import DataLoader, IterableDataset

from birdclef.config import DEFAULT_VOCALIZATION_MODEL_PATH
from birdclef.label.google_inference import GoogleVocalizationInference


class GoogleVocalizationSoundscapeDataset(IterableDataset):
    """Dataset meant for inference on soundscape data."""

    def __init__(
        self,
        soundscape_path: str,
        metadata_path: str,
        model_path: str = DEFAULT_VOCALIZATION_MODEL_PATH,
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
        self.metadata_path = metadata_path
        self.model_path = model_path

    def _load_data(self, iter_start, iter_end):
        model = GoogleVocalizationInference(self.metadata_path, self.model_path)
        for i in range(iter_start, iter_end):
            path = self.soundscapes[i]
            embeddings, logits = model.predict(path)
            n_chunks = embeddings.shape[0]
            indices = range(n_chunks)

            # now we yield a dictionary
            for idx, embedding, logit in zip(indices, embeddings, logits):
                yield {
                    "row_id": f"{path.stem}_{idx*5}",
                    "embedding": embedding,
                    "logits": logit,
                }

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


class GoogleVocalizationSoundscapeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        soundscape_path: str,
        metadata_path: str,
        model_path: str = DEFAULT_VOCALIZATION_MODEL_PATH,
        batch_size: int = 32,
        num_workers: int = 0,
        limit=None,
    ):
        """Initialize the data module.

        :param soundscape_path: The path to the soundscape data.
        :param metadata_path: The path to the metadata.
        :param model_path: The path to the model.
        :param batch_size: The batch size.
        :param num_workers: The number of workers.
        """
        super().__init__()
        self.soundscape_path = soundscape_path
        self.metadata_path = metadata_path
        self.model_path = model_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.limit = limit

    def setup(self, stage=None):
        self.dataloader = DataLoader(
            GoogleVocalizationSoundscapeDataset(
                self.soundscape_path, self.metadata_path, self.model_path, self.limit
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return self.dataloader
