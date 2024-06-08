from functools import partial

import numpy as np
import pandas as pd
import tensorflow_hub as hub
import torchaudio
from tqdm import tqdm

from birdclef.label.inference import Inference


class GoogleVocalizationInference(Inference):
    """Class to perform inference on audio files using a Google Vocalization model."""

    def __init__(
        self,
        metadata_path: str,
        model_path: str = "https://kaggle.com/models/google/bird-vocalization-classifier/frameworks/TensorFlow2/variations/bird-vocalization-classifier/versions/4",
    ):
        self.model = hub.load(model_path)
        self.metadata = pd.read_csv(metadata_path)
        self.model_labels_df = pd.read_csv(
            hub.resolve(model_path) + "/assets/label.csv"
        )
        self.model_indices = self._get_index_subset(
            self.metadata,
            self.model_labels_df,
        )

    def _get_index_subset(
        self, metadata: pd.DataFrame, model_labels: pd.DataFrame
    ) -> list:
        """Get the subset of labels that are in the model.

        We use the species names from the metadata to get the subset of labels that are in the model.
        """
        index_to_label = sorted(metadata.primary_label.unique())
        model_labels = {v: k for k, v in enumerate(model_labels.ebird2021)}
        return sorted(
            [
                model_labels[label] if label in model_labels else -1
                for label in index_to_label
            ]
        )

    def predict(
        self,
        path: str,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Get embeddings and logits for a single audio file.

        :param path: The absolute path to the audio file.
        :param window: The size of the window to split the audio into.
        """
        audio = torchaudio.load(path)[0].numpy()[0]
        embeddings = []
        logits = []
        window = 5 * 32_000
        for i in range(0, len(audio), window):
            clip = audio[i : i + window]
            if len(clip) < window:
                clip = np.concatenate([clip, np.zeros(window - len(clip))])
            result = self.model.infer_tf(clip[None, :])
            embeddings.append(result[1][0].numpy())
            clip_logits = np.concatenate([result[0].numpy(), -np.inf], axis=None)
            logits.append(clip_logits[self.model_indices])
        return embeddings, logits    
