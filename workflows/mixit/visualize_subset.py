import json
import os
from pathlib import Path

import librosa
import luigi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from birdclef import birdnet
from birdclef.data.utils import slice_seconds
from birdclef.utils import spark_resource
from workflows.convert_audio.ogg_to_mp3 import ToMP3Single
from workflows.utils.gcs import single_file_target
from workflows.utils.mixin import DynamicRequiresMixin
from workflows.utils.rsync import GSUtilRsyncTask

from .docker import MixitDockerTask


class GetTrackSubset(luigi.Task):
    durations_path = luigi.Parameter()
    output_path = luigi.Parameter()
    limit = luigi.IntParameter(default=16)
    parallelism = luigi.IntParameter(default=os.cpu_count(), significant=False)

    def output(self):
        return single_file_target(self.output_path)

    def run(self):
        with spark_resource(cores=self.parallelism) as spark:
            df = spark.read.parquet(self.durations_path)
            subset = (
                df.where("duration > 29 and duration < 31")
                .orderBy("duration")
                .limit(self.limit)
            )
            # ensure folder exists
            Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
            subset.toPandas().to_json(self.output().path, orient="records")


class GenerateAssets(luigi.Task, DynamicRequiresMixin):
    output_path = luigi.Parameter()
    track_name = luigi.Parameter()
    birdnet_root = luigi.Parameter()

    def output(self):
        return single_file_target(
            Path(self.output_path)
            / "assets"
            / self.track_name.replace(".ogg", "")
            / "_SUCCESS"
        )

    def run(self):
        # generate spectrograms
        parent = Path(self.output().path).parent
        parent.mkdir(parents=True, exist_ok=True)

        paths = sorted(
            Path(self.output_path).glob(f"audio/**/{Path(self.track_name).stem}*.mp3")
        )
        sr = 48_000
        series = [librosa.load(path, sr=sr)[0] for path in paths]
        # spectogram plots
        for y, path in zip(series, paths):
            plt.figure(figsize=(10, 3))
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel")
            plt.colorbar(format="%+2.0f dB")
            plt.title(f"Mel-frequency spectrogram of {path.parts[-2]}/{path.parts[-1]}")
            plt.tight_layout()
            plt.savefig(f"{parent}/{path.stem}_spectrogram.png")
            plt.close()

        # now lets generate the plots using the labels
        model = birdnet.load_model_from_repo(self.birdnet_root)
        prediction_func = birdnet.prediction_func(model)
        labels = birdnet.load_labels(self.birdnet_root)
        mapped_labels = birdnet.load_mapped_labels(self.birdnet_root)

        # plots for top and bottom 5
        for y, path in zip(series, paths):
            X = slice_seconds(y, sr, seconds=3, step=1)
            pred = prediction_func(X)[0]
            # create a subplot
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            for idx in birdnet.rank_indices(pred, k=5):
                ax[0].plot(pred[:, idx], label=labels[idx])
                ax[0].set_title("top 5 predictions")
                ax[0].legend()
            for idx in birdnet.rank_indices(-pred, k=5):
                ax[1].plot(pred[:, idx], label=labels[idx])
                ax[1].set_title("bottom 5 predictions")
                ax[1].legend()
            plt.savefig(f"{parent}/{path.stem}_top5.png")

            # write out the top 10 predictions and bottom 10 predictions
            top10 = birdnet.rank_indices(pred, k=10)
            bottom10 = birdnet.rank_indices(-pred, k=10)

            # convert these into json
            res = []
            for idx in top10:
                res.append(
                    {
                        "label": labels[idx],
                        "mapped_label": mapped_labels[idx],
                        "y": pred[:, idx].tolist(),
                    }
                )
            Path(f"{parent}/{path.stem}_top10.json").write_text(json.dumps(res))

            res = []
            for idx in bottom10:
                res.append(
                    {
                        "label": labels[idx],
                        "mapped_label": mapped_labels[idx],
                        "y": pred[:, idx].tolist(),
                    }
                )
            Path(f"{parent}/{path.stem}_bottom10.json").write_text(json.dumps(res))

        # touch the success file
        Path(self.output().path).touch()


class MixitWrapperTask(luigi.WrapperTask):
    birdclef_root_path = luigi.Parameter()
    track_subset_path = luigi.Parameter()
    output_path = luigi.Parameter()
    birdnet_root_path = luigi.Parameter()

    def requires(self):
        track_subset_df = pd.read_json(self.track_subset_path)
        for row in track_subset_df.itertuples():
            mixit_task = MixitDockerTask(
                input_path=f"{self.birdclef_root_path}/train_audio",
                output_path=f"{self.output_path}/audio",
                track_name=row.filename,
                num_sources=4,
            )
            yield mixit_task

            # let's also generate the original audio
            mp3_task = ToMP3Single(
                input_path=f"{self.birdclef_root_path}/train_audio",
                output_path=f"{self.output_path}/audio",
                track_name=row.filename,
            )
            yield mp3_task

            generate_task = GenerateAssets(
                output_path=self.output_path,
                track_name=row.filename,
                birdnet_root=self.birdnet_root_path,
                dynamic_requires=[mixit_task, mp3_task],
            )
            yield generate_task


# lets create a task that runs mixit on our subset of data
class VisualizeSubsetWorkflow(luigi.Task):
    birdclef_root_path = luigi.Parameter()
    durations_path = luigi.Parameter()
    output_path = luigi.Parameter()
    birdnet_root_path = luigi.Parameter()
    limit = luigi.IntParameter(default=16)

    def run(self):
        track_subset = GetTrackSubset(
            durations_path=self.durations_path,
            output_path=f"{self.output_path}/subset.json",
            limit=self.limit,
        )
        yield track_subset

        yield MixitWrapperTask(
            birdclef_root_path=self.birdclef_root_path,
            track_subset_path=track_subset.output().path,
            birdnet_root_path=self.birdnet_root_path,
            output_path=self.output_path,
        )


if __name__ == "__main__":
    luigi.build(
        [
            VisualizeSubsetWorkflow(
                birdclef_root_path="data/raw/birdclef-2023",
                durations_path="gs://birdclef-2023/data/processed/birdclef-2023/train_durations_v2.parquet",
                output_path="data/processed/mixit_visualize_subset",
                birdnet_root_path="vendor/BirdNET-Analyzer",
            )
        ],
        workers=max(os.cpu_count() // 2, 1),
        scheduler_host="luigi.us-central1-a.c.birdclef-2023.internal",
        log_level="INFO",
    )
