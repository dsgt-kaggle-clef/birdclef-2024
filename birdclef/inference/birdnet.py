import numpy as np
import pandas as pd
import torch
import torchaudio
from birdnetlib import RecordingBuffer
from birdnetlib.analyzer import Analyzer
from torchaudio.transforms import Resample

from birdclef.inference.base import Inference


class BirdNetInference(Inference):
    """Class to perform inference on audio files using a Google Vocalization model."""

    def __init__(
        self,
        metadata_path: str,
        max_length: int = 0,
    ):
        self.metadata = pd.read_csv(metadata_path)
        self.max_length = max_length
        self.resampler = Resample(32_000, 48_000)
        self.source_sr = 32_000
        self.target_sr = 48_000
        self.analyzer = Analyzer(verbose=False)

    def load(self, path: str, window_sec: int = 5) -> np.ndarray:
        """Load an audio file.

        :param path: The absolute path to the audio file.
        """
        audio, _ = torchaudio.load(path)
        audio = audio[0]
        # right pad the audio so we can reshape into a rectangle
        n = audio.shape[0]
        window = window_sec * self.source_sr
        if (n % window) != 0:
            audio = torch.concatenate([audio, torch.zeros(window - (n % window))])

        audio = self.resampler(audio)
        window = window_sec * self.target_sr
        # reshape the audio into windowsize chunks
        audio = audio.reshape(-1, window)
        if self.max_length > 0:
            audio = audio[: self.max_length]
        return audio

    def _infer(self, audio):
        recording = RecordingBuffer(
            self.analyzer,
            audio.squeeze(),
            self.target_sr,
            overlap=1,
            verbose=False,
        )
        recording.extract_embeddings()
        # concatenate the embeddings together, this should only be two of them
        return torch.stack(
            [torch.from_numpy(np.array(r["embeddings"])) for r in recording.embeddings],
        ).mean(axis=0)

    def predict(
        self, path: str, window: int = 5, **kwargs
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Get embeddings and logits for a single audio file.

        :param path: The absolute path to the audio file.
        :param window: The size of the window to split the audio into.
        """
        audio = self.load(path, window).numpy()
        embeddings = []
        for row in audio:
            embeddings.append(self._infer(row))
        return torch.stack(embeddings), None
