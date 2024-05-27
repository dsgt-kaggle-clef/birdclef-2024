from argparse import ArgumentParser
from functools import partial
from itertools import chain
from pathlib import Path

import luigi
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import torchaudio
from tqdm import tqdm

from birdclef.tasks import maybe_gcs_target


class EmbedSpeciesAudio(luigi.Task):
    metadata_path = luigi.Parameter()
    species = luigi.Parameter()
    output_path = luigi.Parameter()
    model_path = luigi.Parameter(
        default="https://kaggle.com/models/google/bird-vocalization-classifier/frameworks/TensorFlow2/variations/bird-vocalization-classifier/versions/4"
    )

    def output(self):
        maybe_gcs_target(Path(self.output_path) / f"{self.species}.parquet")

    def run(self):
        metadata = pd.read_csv(self.metadata_path)
        model_labels_df = pd.read_csv(
            hub.resolve(self.model_path) + "/assets/label.csv"
        )
        df = self.embed_species(
            metadata,
            self.species,
            Path(self.output().path),
            model=hub.load(self.model_path),
            model_labels=self.get_index_subset(metadata, model_labels_df),
        )
        print(df.head())

    def get_index_subset(
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

    def embed_species(
        self,
        metadata: pd.DataFrame,
        species: str,
        out_path: Path,
        model: hub.KerasLayer,
        model_indices: list,
    ) -> pd.DataFrame:
        tqdm.pandas()
        files = metadata[metadata["primary_label"] == species]
        func = partial(self.embed_single, model=model, model_indices=model_indices)
        cols = zip(*files.progress_apply(func, axis=1))
        names, indices, embeddings, logits = [chain(*col) for col in cols]
        df = pd.DataFrame(
            {
                "name": names,
                "chunk_5s": indices,
                "embedding": embeddings,
                "logits": logits,
            }
        )

        out_path = out_path / f"{species}.parquet"
        df.to_parquet(out_path, index=False)
        return df

    def embed_single(self, row, model, model_indices):
        path = row["filename"]
        embeddings, logits = self.get_embeddings_and_logits(path, model, model_indices)
        n_chunks = embeddings.shape[0]
        indices = range(n_chunks)
        names = [path.split("/")[1]] * n_chunks
        return names, indices, list(embeddings), list(logits)

    def get_embeddings_and_logits(
        self,
        path,
        model,
        model_indices,
        window=5 * 32_000,
    ):
        audio = torchaudio.load(path)[0].numpy()[0]
        embeddings = []
        logits = []
        for i in range(0, len(audio), window):
            clip = audio[i : i + window]
            if len(clip) < window:
                clip = np.concatenate([clip, np.zeros(window - len(clip))])
            result = model.infer_tf(clip[None, :])
            embeddings.append(result[1][0].numpy())
            clip_logits = np.concatenate([result[0].numpy(), -np.inf], axis=None)
            logits.append(clip_logits[model_indices])
        embeddings = np.stack(embeddings)
        logits = np.stack(logits)
        return embeddings, logits


class Workflow(luigi.Task):
    remote_root = luigi.Parameter(default="gs://dsgt-clef-geolifeclef-2024/data")
    local_root = luigi.Parameter(default="/mnt/data/geolifeclef-2024/data")

    def run(self):
        pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scheduler-host", default="localhost")
    args = parser.parse_args()

    luigi.build(
        [Workflow()],
        scheduler_host=args.scheduler_host,
        workers=1,
    )
