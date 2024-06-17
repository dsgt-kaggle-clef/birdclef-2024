import binascii
from argparse import ArgumentParser
from pathlib import Path

import luigi
import pandas as pd
import tqdm

from birdclef.config import DEFAULT_VOCALIZATION_MODEL_PATH
from birdclef.inference.encodec import EncodecInference
from birdclef.inference.vocalization import GoogleVocalizationInference
from birdclef.tasks import RsyncGCSFiles, maybe_gcs_target
from birdclef.utils import spark_resource


class BaseEmbedSpeciesAudio(luigi.Task):
    """Embed all audio files for a species and save to a parquet file."""

    # used to pull audio data from GCS down locally
    remote_root = luigi.Parameter()
    local_root = luigi.Parameter()

    audio_path = luigi.Parameter()
    metadata_path = luigi.Parameter()
    species = luigi.Parameter()
    output_path = luigi.Parameter()

    def output(self):
        return maybe_gcs_target(
            f"{self.remote_root}/{self.output_path}/{self.species}.parquet"
        )

    def run(self):
        yield RsyncGCSFiles(
            src_path=f"{self.remote_root}/{self.audio_path}/{self.species}",
            dst_path=f"{self.local_root}/{self.audio_path}/{self.species}",
        )

        inference = self.get_inference()
        out_path = f"{self.remote_root}/{self.output_path}/{self.species}.parquet"
        df = inference.predict_species_df(
            f"{self.local_root}/{self.audio_path}",
            self.species,
            out_path,
        )
        print(df.head())

    def get_inference(self):
        raise NotImplementedError()


class VocalizationEmbedSpeciesAudio(BaseEmbedSpeciesAudio):
    model_path = luigi.Parameter()

    def get_inference(self):
        metadata_path = f"{self.remote_root}/{self.metadata_path}"
        return GoogleVocalizationInference(
            metadata_path=metadata_path,
            model_path=self.google_model_path,
            use_compiled=True,
        )


class EncodecEmbedSpeciesAudio(BaseEmbedSpeciesAudio):
    chunk_size = luigi.IntParameter(default=5)
    bandwidth = luigi.FloatParameter(default=1.5)

    def get_inference(self):
        metadata_path = f"{self.remote_root}/{self.metadata_path}"
        return EncodecInference(
            metadata_path=metadata_path,
            chunk_size=self.chunk_size,
            target_bandwidth=self.bandwidth,
        )


class BaseSpeciesWorkflow(luigi.Task):
    remote_root = luigi.Parameter()
    local_root = luigi.Parameter()

    audio_path = luigi.Parameter()
    metadata_path = luigi.Parameter()
    intermediate_path = luigi.Parameter()
    output_path = luigi.Parameter()

    partitions = luigi.IntParameter(default=16)

    def get_species_list(self):
        metadata = pd.read_csv(f"{self.remote_root}/{self.metadata_path}")
        return metadata["primary_label"].unique()

    def get_task(self, species):
        raise NotImplementedError()

    def output(self):
        return maybe_gcs_target(f"{self.remote_root}/{self.output_path}/_SUCCESS")

    def run(self):
        species_list = self.get_species_list()
        tasks = []
        for species in species_list:
            task = self.get_task(species)
            tasks.append(task)
        yield tasks

        with spark_resource(memory="16g") as spark:
            spark.read.parquet(
                f"{self.remote_root}/{self.intermediate_path}/*.parquet"
            ).repartition(self.partitions).write.parquet(
                f"{self.remote_root}/{self.output_path}", mode="overwrite"
            )


class VocalizationWorkflow(BaseSpeciesWorkflow):
    def get_task(self, species):
        return VocalizationEmbedSpeciesAudio(
            remote_root=self.remote_root,
            local_root=self.local_root,
            audio_path=self.audio_path,
            metadata_path=self.metadata_path,
            species=species,
            output_path=self.intermediate_path,
            model_path=DEFAULT_VOCALIZATION_MODEL_PATH,
        )


class EncodecWorkflow(BaseSpeciesWorkflow):
    chunk_size = luigi.IntParameter(default=5)
    bandwidth = luigi.FloatParameter(default=1.5)

    def get_task(self, species):
        return EncodecEmbedSpeciesAudio(
            remote_root=self.remote_root,
            local_root=self.local_root,
            audio_path=self.audio_path,
            metadata_path=self.metadata_path,
            species=species,
            output_path=self.intermediate_path,
            chunk_size=self.chunk_size,
            bandwidth=self.bandwidth,
        )


class EmbedSoundscapesAudio(luigi.Task):
    """Embed soundscapes embeddings"""

    remote_root = luigi.Parameter()
    local_root = luigi.Parameter()

    audio_path = luigi.Parameter()
    metadata_path = luigi.Parameter()
    output_path = luigi.Parameter()

    total_batches = luigi.IntParameter(default=100)
    batch_number = luigi.IntParameter()

    def output(self):
        return maybe_gcs_target(
            f"{self.remote_root}/{self.output_path}/{self.batch_number:03d}/_SUCCESS"
        )

    def run(self):
        yield RsyncGCSFiles(
            src_path=f"{self.remote_root}/{self.audio_path}/",
            dst_path=f"{self.local_root}/{self.audio_path}/",
        )

        paths = sorted(Path(f"{self.local_root}/{self.audio_path}").glob("*.ogg"))
        # now only keep the audio files that belong to the same hash
        paths = [
            path
            for path in paths
            if (binascii.crc32(path.stem.encode()) % self.total_batches)
            == self.batch_number
        ]

        inference = self.get_inference()
        for path in tqdm.tqdm(paths):
            out_path = f"{self.remote_root}/{self.output_path}/{self.batch_number:03d}/{path.stem}.parquet"
            if maybe_gcs_target(out_path).exists():
                continue
            df = inference.predict_df(path.parent, path.name)
            df.to_parquet(out_path, index=False)

        # write success
        with self.output().open("w") as f:
            f.write("")

    def get_inference(self):
        metadata_path = f"{self.remote_root}/{self.metadata_path}"
        return GoogleVocalizationInference(
            metadata_path=metadata_path,
            use_compiled=True,
        )


class EmbedSoundscapesAudioWorkflow(luigi.Task):
    remote_root = luigi.Parameter()
    local_root = luigi.Parameter()

    audio_path = luigi.Parameter()
    metadata_path = luigi.Parameter()
    intermediate_path = luigi.Parameter()
    output_path = luigi.Parameter()

    total_batches = luigi.IntParameter(default=100)
    num_partitions = luigi.IntParameter(default=16)

    def output(self):
        return maybe_gcs_target(f"{self.remote_root}/{self.output_path}/_SUCCESS")

    def run(self):
        yield [
            EmbedSoundscapesAudio(
                remote_root=self.remote_root,
                local_root=self.local_root,
                audio_path=self.audio_path,
                metadata_path=self.metadata_path,
                output_path=self.intermediate_path,
                total_batches=self.total_batches,
                batch_number=i,
            )
            for i in range(self.total_batches)
        ]

        with spark_resource(memory="16g") as spark:
            spark.read.parquet(
                f"{self.remote_root}/{self.intermediate_path}/*/*.parquet"
            ).repartition(self.partitions).write.parquet(
                f"{self.remote_root}/{self.output_path}", mode="overwrite"
            )


class Workflow(luigi.Task):
    remote_root = luigi.Parameter(default="gs://dsgt-clef-birdclef-2024/data")
    local_root = luigi.Parameter(default="/mnt/data")
    train_audio_path = luigi.Parameter(default="raw/birdclef-2024/train_audio")
    unlabeled_soundscapes_path = luigi.Parameter(
        default="raw/birdclef-2024/unlabeled_soundscapes"
    )
    metadata_path = luigi.Parameter(default="raw/birdclef-2024/train_metadata.csv")

    def run(self):
        common_args = dict(
            remote_root=self.remote_root,
            local_root=self.local_root,
            metadata_path=self.metadata_path,
        )
        yield [
            VocalizationWorkflow(
                **common_args,
                audio_path=self.train_audio_path,
                intermediate_path=f"intermediate/google_embeddings/v1",
                output_path=f"processed/google_embeddings/v1",
            ),
            EmbedSoundscapesAudioWorkflow(
                **common_args,
                audio_path=self.unlabeled_soundscapes_path,
                intermediate_path=f"intermediate/google_soundscape_embeddings/v1",
                output_path=f"processed/google_soundscape_embeddings/v1",
            ),
            EncodecWorkflow(
                **common_args,
                audio_path=self.train_audio_path,
                intermediate_path=f"intermediate/encodec_embeddings/v1",
                output_path=f"processed/encodec_embeddings/v1",
                bandwidth=1.5,
            ),
            EncodecWorkflow(
                **common_args,
                audio_path=self.train_audio_path,
                intermediate_path=f"intermediate/encodec_embeddings/v2",
                output_path=f"processed/encodec_embeddings/v2",
                bandwidth=3.0,
            ),
        ]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--scheduler-host",
        type=str,
        default="services.us-central1-a.c.dsgt-clef-2024.internal",
    )
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    luigi.build(
        [Workflow()],
        scheduler_host=args.scheduler_host,
        workers=args.workers,
    )
