from argparse import ArgumentParser

import luigi
import pandas as pd

from birdclef.inference.birdnet import BirdNetInference
from birdclef.tasks import RsyncGCSFiles, maybe_gcs_target
from birdclef.utils import spark_resource


class EmbedSpeciesAudio(luigi.Task):
    """Embed all audio files for a species and save to a parquet file."""

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
        return BirdNetInference(
            metadata_path=f"{self.remote_root}/{self.metadata_path}",
        )


class EmbedWorkflow(luigi.Task):
    remote_root = luigi.Parameter()
    local_root = luigi.Parameter()

    audio_path = luigi.Parameter()
    metadata_path = luigi.Parameter()
    intermediate_path = luigi.Parameter()
    output_path = luigi.Parameter()

    partitions = luigi.IntParameter(default=16)

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
            )
            tasks.append(task)
        yield tasks

        with spark_resource(memory="16g") as spark:
            spark.read.parquet(
                f"{self.remote_root}/{self.intermediate_path}/*.parquet"
            ).repartition(self.partitions).write.parquet(
                f"{self.remote_root}/{self.output_path}", mode="overwrite"
            )


class Workflow(luigi.Task):
    remote_root = luigi.Parameter()
    local_root = luigi.Parameter()

    audio_path = luigi.Parameter()
    metadata_path = luigi.Parameter()
    intermediate_path = luigi.Parameter()
    output_path = luigi.Parameter()

    def run(self):
        yield EmbedWorkflow(
            remote_root=self.remote_root,
            local_root=self.local_root,
            audio_path=self.audio_path,
            metadata_path=self.metadata_path,
            intermediate_path=self.intermediate_path,
            output_path=self.output_path,
        )

        # TODO: train workflow


def parse_args():
    parser = ArgumentParser()

    default_out_folder = "birdnet"
    defaults = {
        "remote-root": "gs://dsgt-clef-birdclef-2024/data",
        "local-root": "/mnt/data",
        "audio-path": "raw/birdclef-2024/train_audio",
        "metadata-path": "raw/birdclef-2024/train_metadata.csv",
        "intermediate-path": f"intermediate/{default_out_folder}/v1",
        "output-path": f"processed/{default_out_folder}/v1",
        "scheduler-host": "services.us-central1-a.c.dsgt-clef-2024.internal",
        "workers": 2,
    }

    for arg, default in defaults.items():
        parser.add_argument(f"--{arg}", type=type(default), default=default)

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
