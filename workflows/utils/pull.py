import os
from multiprocessing import Pool
from pathlib import Path

import luigi
from google.cloud import storage


class Pull(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    parallelism = luigi.IntParameter(default=os.cpu_count(), significant=False)

    def output(self):
        return luigi.LocalTarget(self.output_path)

    def list_tracks(self):
        storage_client = storage.Client("birdclef-2023")

        bucket = storage_client.get_bucket("birdclef-2023")
        # Note: Client.list_blobs requires at least package version 1.17.0.
        # Use input path "data/raw/birdclef-2022/train_audio/species"
        tracks = bucket.list_blobs(prefix=self.input_path)

        return tracks

    def requires(self):
        return None

    def run(self):
        self.output_path = Path(self.output_path)
        self.input_path = Path(self.input_path)

        tracks = self.list_tracks()

        if not self.output_path.is_dir():
            self.output_path.mkdir(parents=True, exist_ok=True)
        for track_file in tracks:
            name = str(track_file.name).split("/")[-1]

            track_file.download_to_filename(self.output_path / name)
