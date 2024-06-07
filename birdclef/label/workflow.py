from argparse import ArgumentParser

import luigi
import pandas as pd

from birdclef.label.inference import GoogleVocalizationInference
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
            src_path=f"{self.remote_root}/{self.audio_path}/{self.species}",
            dst_path=f"{self.local_root}/{self.audio_path}/{self.species}",
        )

        gvi = GoogleVocalizationInference(
            metadata_path=f"{self.remote_root}/{self.metadata_path}",
            model_path=self.model_path,
        )
        out_path = f"{self.remote_root}/{self.output_path}/{self.species}.parquet"
        df = gvi.predict_species_df(
            f"{self.local_root}/{self.audio_path}",
            self.species,
            out_path,
        )
        print(df.head())


class Workflow(luigi.Task):
    remote_root = luigi.Parameter()
    local_root = luigi.Parameter()

    audio_path = luigi.Parameter()
    metadata_path = luigi.Parameter()
    intermediate_path = luigi.Parameter()
    output_path = luigi.Parameter()

    model_path = luigi.Parameter()
    partitions = luigi.IntParameter(default=64)

    def run(self):
        metadata = pd.read_csv(f"{self.remote_root}/{self.metadata_path}")
        species_list = metadata["primary_label"].unique()

        tasks = []
        for species in species_list:
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
