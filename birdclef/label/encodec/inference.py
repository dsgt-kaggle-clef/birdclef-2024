from functools import partial
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow_hub as hub
import torchaudio
from tqdm import tqdm

from encodec import EncodecModel
from encodec.utils import convert_audio

from birdclef.label.inference import Inference


class EncodecInference(Inference):
    """Class to perform inference on audio files using an Encodec model."""

    def __init__(
        self,
        metadata_path: str,
        chunk_size: int = 1,
    ):
        self.metadata = pd.read_csv(metadata_path)
        self.chunk_size = chunk_size
        self.model = EncodecModel.encodec_model_24khz().model.set_target_bandwidth(3.0)

    def predict(
        self,
        path: str,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Get embeddings for a single audio file.

        :param path: The absolute path to the audio file.
        """
        audio, sr = torchaudio.load(path)
        audio = convert_audio(audio, sr, self.model.sample_rate, self.model.channels)
        audio = audio.unsqueeze(0)

        true_chunk_size = chunk_size * model.sample_rate
        chunks = torch.split(audio, true_chunk_size, dim=-1)
        last_padded = F.pad(chunks[-1], (0, true_chunk_size - chunks[-1].shape[-1]))
        chunks = chunks[:-1] + (last_padded,)
        embeddings = [self.encode(chunk) for chunk in chunks]
        return embeddings, None
