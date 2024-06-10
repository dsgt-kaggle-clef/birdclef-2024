from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

from birdclef.config import DEFAULT_VOCALIZATION_MODEL_PATH


class Inference:
    """Class to perform inference on audio files."""

    def predict(
        self,
        path: str,
        **kwargs,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Get embeddings and logits for a single audio file.

        :param path: The absolute path to the audio file.
        :param window: The size of the window to split the audio into.
        """
        raise NotImplementedError

    def predict_df(self, root, suffix) -> pd.DataFrame:
        """Embed a single audio file.

        :param root: The root directory of the audio files.
        :param suffix: The filename of the audio file.
        """
        path = f"{root}/{suffix}"
        embeddings, logits = self.predict(path)
        indices = range(embeddings.shape[0])
        df = pd.DataFrame(
            {"name": f"{suffix}", "chunk_5s": indices, "embedding": embeddings.tolist()}
        )
        if logits is not None:
            df["logits"] = logits.tolist()
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
