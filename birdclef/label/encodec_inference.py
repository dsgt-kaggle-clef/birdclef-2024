import numpy as np
import pandas as pd
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
import torch.nn.functional as F

from birdclef.label.inference import Inference


class EncodecInference(Inference):
    """Class to perform inference on audio files using an Encodec model."""

    def __init__(
        self,
        metadata_path: str,
        chunk_size: int = 1,
    ):
        device = torch.cuda.device(0)
        self.metadata = pd.read_csv(metadata_path)
        self.chunk_size = chunk_size
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(3.0)
        self.model = self.model.to(device)
        
    def encode(self, audio):
        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = self.model.encode(audio)
        embeddings = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
        return embeddings[0].flatten().numpy()

    def predict(
        self,
        path: str,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Get embeddings for a single audio file.

        :param path: The absolute path to the audio file.
        """
        print(torch.cuda.is_available())
        
        audio, sr = torchaudio.load(path)
        audio = convert_audio(audio, sr, self.model.sample_rate, self.model.channels)
        audio = audio.unsqueeze(0)

        true_chunk_size = self.chunk_size * self.model.sample_rate
        chunks = torch.split(audio, true_chunk_size, dim=-1)
        last_padded = F.pad(chunks[-1], (0, true_chunk_size - chunks[-1].shape[-1]))
        chunks = chunks[:-1] + (last_padded,)
        embeddings = [self.encode(chunk) for chunk in chunks]
        return embeddings, None
