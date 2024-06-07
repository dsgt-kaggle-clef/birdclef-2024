from pathlib import Path

import torch
from torch.utils.data import IterableDataset

from birdclef.label.inference import GoogleVocalizationInference


class GoogleVocalizationSoundscapeDataset(IterableDataset):
    """Dataset meant for inference on soundscape data."""

    def __init__(
        self,
        soundscape_path: str,
        metadata_path: str,
        model_path: str = "https://kaggle.com/models/google/bird-vocalization-classifier/frameworks/TensorFlow2/variations/bird-vocalization-classifier/versions/4",
    ):
        """Initialize the dataset.

        :param soundscape_path: The path to the soundscape data.
        :param metadata_path: The path to the metadata.
        :param model_path: The path to the model.
        """
        self.soundscapes = sorted(Path(soundscape_path).glob("**/*.ogg"))
        self.model = GoogleVocalizationInference(metadata_path, model_path)

    def _load_data(self, iter_start, iter_end):
        for i in range(iter_start, iter_end):
            path = self.soundscapes[i]
            df = self.model.predict_df(path.parent, path.name)

            # now we yield a dictionary
            for row in df.itertuples():
                yield {
                    "row_id": f"{Path(row.name).stem}_{row.chunk_5s*5}",
                    "embedding": torch.from_numpy(row.embedding),
                    "logits": torch.from_numpy(row.logits),
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
