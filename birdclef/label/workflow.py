from argparse import ArgumentParser
from functools import partial
from itertools import chain

import luigi
import numpy as np
import pandas as pd
import tensorflow_hub as hub
import torchaudio
from tqdm import tqdm

from birdclef.tasks import RsyncGCSFiles, maybe_gcs_target
from birdclef.utils import spark_resource


class EmbedSpeciesAudio(luigi.Task):
    """Embed all audio files for a species and save to a parquet file."""

    # used to pull audio data from GCS down locally
    remote_root = luigi.Parameter()
    local_root = luigi.Parameter()

    audio_path = luigi.Parameter()
    metadata_path = luigi.Parameter()
    species = luigi.Parameter()
    output_path = luigi.Parameter()
    model_path = luigi.Parameter()

    def output(self):
        return maybe_gcs_target(
            f"{self.remote_root}/{self.output_path}/{self.species}.parquet"
        )

    def run(self):
        yield RsyncGCSFiles(
            src_path=f"{self.remote_root}/{self.audio_path}",
            dst_path=f"{self.local_root}/{self.audio_path}",
        )

        metadata = pd.read_csv(f"{self.remote_root}/{self.metadata_path}")
        model_labels_df = pd.read_csv(
            hub.resolve(self.model_path) + "/assets/label.csv"
        )
        df = self.embed_species(
            metadata,
            self.species,
            f"{self.remote_root}/{self.output_path}/{self.species}.parquet",
            model=hub.load(self.model_path),
            model_indices=self.get_index_subset(metadata, model_labels_df),
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
        out_path: str,
        model: hub.KerasLayer,
        model_indices: list,
    ) -> pd.DataFrame:
        """Embed all audio files for a species and save to a parquet file."""
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
        df.to_parquet(out_path, index=False)
        return df

    def embed_single(self, row, model, model_indices):
        """Embed a single audio file."""
        path = f"{self.local_root}/{self.audio_path}/{row.filename}"
        embeddings, logits = self.get_embeddings_and_logits(path, model, model_indices)
        n_chunks = embeddings.shape[0]
        indices = range(n_chunks)
        names = [row.filename] * n_chunks
        return names, indices, list(embeddings), list(logits)

    def get_embeddings_and_logits(
        self,
        path: str,
        model: hub.KerasLayer,
        model_indices: list,
        window: int = 5 * 32_000,
    ):
        """Get embeddings and logits for a single audio file."""
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
    remote_root = luigi.Parameter()
    local_root = luigi.Parameter()

    audio_path = luigi.Parameter()
    metadata_path = luigi.Parameter()
    intermediate_path = luigi.Parameter()
    output_path = luigi.Parameter()

    model_path = luigi.Parameter()
    partitions = luigi.IntParameter(default=64)

    def requires(self):
        metadata = pd.read_csv(f"{self.remote_root}/{self.metadata_path}")
        species_list = metadata["primary_label"].unique()

        tasks = []
        for species in species_list[:1]:
            task = EmbedSpeciesAudio(
                remote_root=self.remote_root,
                local_root=self.local_root,
                audio_path=self.audio_path,
                metadata_path=self.metadata_path,
                species=species,
                output_path=self.intermediate_path,
                model_path=self.model_path,
            )
            tasks.append(task)
        yield tasks

    def run(self):
        with spark_resource(memory="16g") as spark:
            spark.read.parquet(
                f"{self.remote_root}/{self.intermediate_path}/*.parquet"
            ).repartition(self.partitions).write.parquet(
                f"{self.remote_root}/{self.output_path}", mode="overwrite"
            )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--remote-root",
        type=str,
        default="gs://dsgt-clef-birdclef-2024/data",
    )
    parser.add_argument(
        "--local-root",
        type=str,
        default="/mnt/data",
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        default="raw/birdclef-2024/train_audio",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="raw/birdclef-2024/train_metadata.csv",
    )
    parser.add_argument(
        "--intermediate-path",
        type=str,
        default="intermediate/google_embeddings/v1",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="processed/google_embeddings/v1",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="https://kaggle.com/models/google/bird-vocalization-classifier/frameworks/TensorFlow2/variations/bird-vocalization-classifier/versions/4",
    )
    parser.add_argument(
        "--scheduler-host",
        type=str,
        default="services.us-central1-a.c.dsgt-clef-2024.internal",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    luigi.build(
        [
            Workflow(
                **{
                    k: v
                    for k, v in vars(args).items()
                    if k not in ["scheduler_host", "workers"]
                }
            )
        ],
        scheduler_host=args.scheduler_host,
        workers=args.workers,
    )
