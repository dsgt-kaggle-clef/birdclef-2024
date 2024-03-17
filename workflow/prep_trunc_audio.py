"""Prepare audio for label studio for call/no-call labeling.

Usage:
    python -m workflow.prep_trunc_audio
"""

import os
from pathlib import Path

import ffmpeg
import luigi
import luigi.contrib.gcs
import luigi.local_target
import pandas as pd
import tqdm


class DownloadSpeciesAudio(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    species = luigi.Parameter()

    def output(self):
        # a directory with a _SUCCESS flag file
        return luigi.LocalTarget(Path(self.output_path) / self.species / "_SUCCESS")

    def run(self):
        client = luigi.contrib.gcs.GCSClient()

        # list all the files in the input path
        files = client.listdir(f"{self.input_path}/{self.species}")
        output_root = Path(self.output_path) / self.species
        output_root.mkdir(parents=True, exist_ok=True)
        for file in tqdm.tqdm(files):
            # download the file to a temp file, and write it to the output path
            fp = client.download(file)
            (output_root / Path(file).name).write_bytes(fp.read())

        # create a _SUCCESS file
        self.output().open("w").close()


class TruncateSpeciesAudio(luigi.Task):
    """For a single species, truncate the audio to the first 5 seconds.

    We use the training metadata to find the paths to the audio from each species.
    The paths are relative to a gcs bucket, so they all need to be downloaded to a local directory.
    We use ffmpeg to truncate the audio to the first 5 seconds.
    We also re-encode the audio into mp3 since it's better supported on more devices.
    Then we upload the truncated audio back into the target gcs bucket.
    """

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    species = luigi.Parameter()
    tmp_path = luigi.Parameter(default="/mnt/data/tmp/prep_trunc_audio")

    def requires(self):
        return DownloadSpeciesAudio(
            input_path=f"{self.input_path}/train_audio",
            output_path=self.tmp_path,
            species=self.species,
        )

    def output(self):
        return luigi.contrib.gcs.GCSFlagTarget(f"{self.output_path}/{self.species}/")

    def run(self):
        print(self.output().path)
        # list all the files in the input path
        input_root = Path(self.requires().output().path).parent
        files = list(input_root.glob("*.ogg"))
        client = luigi.contrib.gcs.GCSClient()

        # loop over the files, truncate and re-encode
        for file in tqdm.tqdm(files):
            # use ffmpeg to truncate the audio to the first 5 seconds
            # create a temporary filename for the output
            output_file = Path(self.tmp_path) / "trunc" / f"{file.stem}.mp3"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            (
                ffmpeg.input(file)
                .filter("atrim", start=0, end=5)
                # convert to mp3
                .output(
                    output_file.as_posix(), ar=44100, ac=2, ab="192k", loglevel="quiet"
                )
                .run(overwrite_output=True)
            )

            # upload the file to the output path
            path = f"{self.output_path}/{self.species}/{output_file.name}"
            client.put(output_file.as_posix(), path, mimetype="audio/mpeg")

        # create the _SUCCESS flag
        client.put_string("", f"{self.output_path}/{self.species}/_SUCCESS")


if __name__ == "__main__":
    # get a list of all species
    client = luigi.contrib.gcs.GCSClient()
    fp = client.download(
        "gs://dsgt-clef-birdclef-2024/data/raw/birdclef-2023/train_metadata.csv"
    )
    # now read into pandas
    df = pd.read_csv(fp)
    species = df["primary_label"].unique()
    luigi.build(
        [
            TruncateSpeciesAudio(
                input_path="gs://dsgt-clef-birdclef-2024/data/raw/birdclef-2023",
                output_path="gs://dsgt-clef-birdclef-2024/data/processed/birdclef-2023/truncated_audio",
                species=specie,
            )
            for specie in species
        ],
        scheduler_host="services.us-central1-a.c.dsgt-clef-2024.internal",
        workers=os.cpu_count(),
    )
