from functools import partial
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow_hub as hub
import torchaudio
from tqdm import tqdm

from birdclef.config import DEFAULT_VOCALIZATION_MODEL_PATH


class GoogleVocalizationInference:
    """Class to perform inference on audio files using a Google Vocalization model."""

    def __init__(
        self,
        metadata_path: str,
        model_path: str = DEFAULT_VOCALIZATION_MODEL_PATH,
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
        window: int = 5 * 32_000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get embeddings and logits for a single audio file.

        :param path: The absolute path to the audio file.
        :param window: The size of the window to split the audio into.
        """
        audio = torchaudio.load(path)[0].numpy()[0]
        embeddings = []
        logits = []
        for i in range(0, len(audio), window):
            clip = audio[i : i + window]
            if len(clip) < window:
                clip = np.concatenate([clip, np.zeros(window - len(clip))])
            result = self.model.infer_tf(clip[None, :])
            embeddings.append(result[1][0].numpy())
            clip_logits = np.concatenate([result[0].numpy(), -np.inf], axis=None)
            logits.append(clip_logits[self.model_indices])
        embeddings = np.stack(embeddings)
        logits = np.stack(logits)
        return embeddings, logits

    def predict_df(self, root, suffix) -> pd.DataFrame:
        """Embed a single audio file.

        :param root: The root directory of the audio files.
        :param suffix: The filename of the audio file.
        """
        path = f"{root}/{suffix}"
        embeddings, logits = self.predict(path)
        n_chunks = embeddings.shape[0]
        indices = range(n_chunks)
        df = pd.DataFrame(
            {
                "name": f"{suffix}",
                "chunk_5s": indices,
                "embedding": list(embeddings),
                "logits": list(logits),
            }
        )
        return df

    def predict_species_df(
        self,
        root: str,
        species: str,
        out_path: str,
    ) -> pd.DataFrame:
        """Helper function to embed all the training data for a species in the training dataset.

        :param root: The root directory of the audio files.
        :param species: The species to embed.
        :param out_path: The path to save the embeddings.
        """
        tqdm.pandas()
        subset = self.metadata[self.metadata["primary_label"] == species]
        dfs = subset.filename.progress_apply(partial(self.predict_df, root)).tolist()
        df = pd.concat(dfs)
        df.to_parquet(out_path, index=False)
        return df
