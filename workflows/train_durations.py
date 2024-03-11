import os
import tempfile
from multiprocessing import Pool
from pathlib import Path

import librosa
import luigi
import pandas as pd
from tqdm import tqdm

from workflows.utils.gcs import single_file_target
from workflows.utils.mixin import DynamicRequiresMixin
from workflows.utils.rsync import GSUtilRsyncTask


def read_path(path: Path) -> dict:
    y, sr = librosa.load(path)
    duration = librosa.get_duration(y=y, sr=sr)
    return dict(filename="/".join(path.parts[-2:]), duration=duration)


class TrainDurations(luigi.Task, DynamicRequiresMixin):
    """Get the duration of each song in the training set into a parquet file."""

    birdclef_root_path = luigi.Parameter()
    output_path = luigi.Parameter()
    parallelism = luigi.IntParameter(default=os.cpu_count())
    capture_output = True

    def output(self):
        return single_file_target(self.output_path)

    def run(self):
        paths = sorted(Path(self.birdclef_root_path).glob("train_audio/**/*.ogg"))
        if not paths:
            raise ValueError("No audio files found", self.birdclef_root_path)
        with Pool(self.parallelism) as pool:
            res = list(tqdm(pool.imap(read_path, paths), total=len(paths)))

        df = pd.DataFrame(res)
        print(df.head())
        # ensure the output directory exists
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.output_path)


class TrainDurationsWorkflow(luigi.Task):
    birdclef_root_path = luigi.Parameter()
    output_path = luigi.Parameter()
    local_data_root = luigi.OptionalParameter(default=None)
    parallelism = luigi.IntParameter(default=os.cpu_count())

    def output(self):
        return single_file_target(self.output_path)

    def run(self):
        # create a temporary path to download the data, make it persistent since
        # we're only wrapping the task here
        if self.local_data_root is None:
            self.local_data_root = tempfile.mkdtemp()

        tmp_path = Path(self.local_data_root)
        tmp_birdclef_root_path = tmp_path / "raw" / Path(self.birdclef_root_path).name
        tmp_birdclef_root_path.mkdir(parents=True, exist_ok=True)
        tmp_output_path = tmp_path / "processed" / "train_durations.parquet"
        tmp_output_path.parent.mkdir(parents=True, exist_ok=True)

        download_task = GSUtilRsyncTask(
            input_path=self.birdclef_root_path,
            output_path=tmp_birdclef_root_path.as_posix(),
            is_dir=True,
        )
        yield download_task

        train_durations = TrainDurations(
            birdclef_root_path=tmp_birdclef_root_path.as_posix(),
            output_path=tmp_output_path.as_posix(),
            parallelism=self.parallelism,
            dynamic_requires=[download_task],
        )
        yield train_durations

        upload_task = GSUtilRsyncTask(
            input_path=tmp_output_path.as_posix(),
            output_path=self.output_path,
            is_dir=False,
            dynamic_requires=[download_task, train_durations],
        )
        yield upload_task


if __name__ == "__main__":
    luigi.build(
        [
            TrainDurations(
                birdclef_root_path="data/raw/birdclef-2022",
                output_path="data/processed/birdclef-2022/train_durations.parquet",
            )
        ],
        log_level="INFO",
    )
