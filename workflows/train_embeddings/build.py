"""
Module to build the embedding training dataset from the audio files.

We do a number of things here:

- Pad each audio file to be an increment of 3 seconds, with some noise added
- Sound separate the audio files
- Extract embeddings and predictions from the birdnet model on the resulting track
- Consolidate a single parquet dataset with embeddings and predictions

We perform this on a per-track basis.
"""

import itertools
import json
import os
import random
from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import librosa
import luigi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import tqdm
from pyspark.sql import Row

from birdclef import birdnet
from birdclef.data.utils import slice_seconds
from birdclef.utils import spark_resource
from workflows.convert_audio.ogg_to_mp3 import ToMP3Single
from workflows.mixit.docker import MixitDockerTask
from workflows.utils.gcs import single_file_target
from workflows.utils.mixin import DynamicRequiresMixin
from workflows.utils.rsync import GSUtilRsyncTask


def split_duration(duration, max_duration):
    """Split a duration into intervals that is at most max_duration.

    We do this by splitting the duration in half until we get a duration that is
    less than the max_duration.
    """
    if duration <= max_duration:
        return duration
    else:
        return split_duration(duration / 2, max_duration)


def duration_parts(duration, max_duration):
    return np.ceil(duration / split_duration(duration, max_duration)).astype(int)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst.
    https://stackoverflow.com/a/312464
    """
    if n < 1:
        yield lst
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def test_split_duration():
    assert split_duration(10, 3) == 2.5
    assert split_duration(10, 5) == 5
    assert split_duration(10, 10) == 10
    assert split_duration(10, 11) == 10


class PadAudioNoise(luigi.Task, DynamicRequiresMixin):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    track_name = luigi.Parameter()
    noise_alpha = luigi.FloatParameter(default=0.1)
    duration = luigi.FloatParameter(default=-1)
    max_duration = luigi.FloatParameter(default=3 * 60)

    def output(self):
        if self.duration > 0 and self.duration > self.max_duration:
            return [
                single_file_target(
                    Path(self.output_path)
                    / self.track_name.replace(".ogg", f"_part{i:03d}.wav")
                )
                for i in range(duration_parts(self.duration, self.max_duration))
            ]
        else:
            return [
                single_file_target(
                    Path(self.output_path) / self.track_name.replace(".ogg", ".wav")
                )
            ]

    def write_wav(self, y, sr, output_path):
        # lets generate noise using the same distribution as the signal. The shape
        # should be rounded up the the nearest 3rd second.
        noise_shape = (int(y.shape[0] / sr // 3) + 1) * 3 * sr
        noise = np.random.normal(loc=y.mean(), scale=y.std(), size=noise_shape)

        # lets add the noise to the signal via weighted average, making the noise 10% of the signal
        zero_pad_y = np.zeros(noise_shape)
        # now place the signal in the center of the padding
        start = (noise_shape - y.shape[0]) // 2
        end = start + y.shape[0]
        zero_pad_y[start:end] = y

        y_noise = (zero_pad_y * 0.9) + (noise * 0.1)

        # write this as a temporary wav file
        sf.write(output_path, y_noise, sr)

        # wait for the file to be written up to 10 seconds
        for i in range(20):
            if output_path.exists():
                break
            time.sleep(0.5)

    def run(self):
        path = Path(self.input_path) / self.track_name
        y, sr = librosa.load(path.as_posix(), sr=48_000, mono=True)

        # now lets write the files
        output_path = Path(self.output()[0].path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.duration <= 0:
            self.write_wav(y, sr, output_path)
        else:
            duration = split_duration(self.duration, self.max_duration)
            slices = slice_seconds(y, sr, seconds=duration)
            paths = [Path(x.path) for x in self.output()]
            for path, X in zip(paths, slices):
                self.write_wav(X, sr, path)


class ExtractEmbedding(luigi.Task, DynamicRequiresMixin):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    track_name = luigi.Parameter()
    birdnet_root_path = luigi.Parameter()
    parallelism = luigi.IntParameter(default=2)

    @property
    def track_stem(self):
        return self.track_name.replace(".ogg", "")

    def output(self):
        return single_file_target(Path(self.output_path) / self.track_stem / "_SUCCESS")

    def process_audio_files(
        self,
        paths,
        prediction_func,
        embedding_func,
        labels,
        mapped_labels,
        sr=48_000,
    ):
        for path in tqdm.tqdm(paths):
            y, sr = librosa.load(path.as_posix(), sr=sr, mono=True)

            # we add in functionality from our notebook, so that we can process
            # audio in 5 second chunks
            X = slice_seconds(y, sr, seconds=3, step=1)

            pred = prediction_func(X)[0]
            emb = embedding_func(X)[0]

            pred_sigmoid = 1 / (1 + np.exp(-pred))
            indices = birdnet.rank_indices(pred_sigmoid, k=3)

            for i in range(pred_sigmoid.shape[0]):
                predictions = [
                    {
                        "index": int(j),
                        "label": labels[j],
                        "mapped_label": mapped_labels[j],
                        "probability": float(pred_sigmoid[i][j]),
                    }
                    for rank, j in enumerate(indices)
                ]
                # sort and add an index for the rank, because each segment
                # is ranked by the most common species across the entire
                # segment
                predictions = [
                    Row(rank=rank, **row)
                    for rank, row in enumerate(
                        sorted(predictions, key=lambda row: -row["probability"])
                    )
                ]
                yield Row(
                    **{
                        "species": path.parts[-2],
                        "track_stem": path.stem.split("_source")[0],
                        "track_type": (
                            path.stem.split("_")[-1]
                            if "source" in path.stem
                            else "original"
                        ),
                        "track_name": "/".join(path.parts[-2:]),
                        "embedding": emb[i].astype(float).tolist(),
                        "prediction_vec": pred[i].astype(float).tolist(),
                        "predictions": predictions,
                        "start_time": i,
                        "energy": float(np.sum(X[i] ** 2)),
                    }
                )

    def run(self):
        # load the model
        repo_path = self.birdnet_root_path
        model = birdnet.load_model_from_repo(repo_path)

        # find all the audio files to process

        # load the audio file
        paths = sorted(Path(self.input_path).glob(f"{self.track_stem}*.mp3"))
        rows = list(
            self.process_audio_files(
                paths,
                birdnet.prediction_func(model),
                birdnet.embedding_func(model),
                birdnet.load_labels(repo_path),
                birdnet.load_mapped_labels(repo_path),
            )
        )
        # now with spark, lets write the parquet file
        with spark_resource(cores=self.parallelism) as spark:
            df = spark.createDataFrame(rows)
            assert df.count() > 0, "No rows found in dataframe"
            df.coalesce(2).write.parquet(
                # the output is the success file, but we want the parent directory
                Path(self.output().path).parent.as_posix(),
                mode="overwrite",
            )


class TrackWorkflow(luigi.Task):
    birdclef_root_path = luigi.Parameter()
    output_path = luigi.Parameter()
    birdnet_root_path = luigi.Parameter()
    track_name = luigi.Parameter()
    duration = luigi.FloatParameter(default=-1)
    max_duration = luigi.FloatParameter(default=3 * 60)

    def output(self):
        outputs = [
            single_file_target(
                Path(self.output_path)
                / "embeddings"
                / self.track_name.replace(".ogg", "")
                / "_SUCCESS"
            )
        ]
        return outputs

    def run(self):
        pad_noise = yield PadAudioNoise(
            input_path=f"{self.birdclef_root_path}/train_audio",
            output_path=f"{self.output_path}/audio",
            track_name=self.track_name,
            duration=self.duration,
            max_duration=self.max_duration,
        )
        deps = []
        for output in pad_noise:
            p = Path(output.path)
            wav_track_name = "/".join(p.parts[-2:])
            convert_mp3 = ToMP3Single(
                input_path=f"{self.output_path}/audio",
                output_path=f"{self.output_path}/audio",
                track_name=wav_track_name,
                input_ext=".wav",
                dynamic_requires=pad_noise,
            )
            mixit = MixitDockerTask(
                input_path=f"{self.output_path}/audio",
                output_path=f"{self.output_path}/audio",
                track_name=wav_track_name,
                num_sources=4,
                dynamic_requires=pad_noise,
            )
            deps.append(convert_mp3)
            deps.append(mixit)
        dep_output = yield deps
        dep_output = [(d if isinstance(d, list) else [d]) for d in dep_output]
        dep_output = [item for sublist in dep_output for item in sublist]

        yield ExtractEmbedding(
            input_path=f"{self.output_path}/audio",
            output_path=f"{self.output_path}/embeddings",
            track_name=self.track_name,
            birdnet_root_path=self.birdnet_root_path,
            dynamic_requires=[pad_noise] + dep_output,
        )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--birdclef-root-path", default="data/raw/birdclef-2023")
    parser.add_argument(
        "--output-path", default="data/processed/birdclef-2023/train_embeddings"
    )
    parser.add_argument(
        "--birdnet-root-path", default="data/models/birdnet-analyzer-pruned"
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", default=100, type=int)
    parser.add_argument(
        "--workers", default=max(int(os.cpu_count() * 3 / 8), 1), type=int
    )
    parser.add_argument(
        "--scheduler-host",
        type=str,
        default="luigi.us-central1-a.c.birdclef-2023.internal",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # let's disable cuda for now
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    train_audio_root = Path(args.birdclef_root_path) / "train_audio"
    species = sorted([p.name for p in train_audio_root.glob("*")])

    # this doesn't work super well for a couple different reasonqs:
    # 1. there are a lot of small files, so the overhead of running luigi is high
    # 2. the species are not evenly distributed, so some species will take a long time
    # 3. we can get stuck on super long audio files...

    # for s in species:
    #     luigi.build(
    #         [
    #             SpeciesWorkflow(
    #                 birdclef_root_path=birdclef_root_path,
    #                 output_path=output_path,
    #                 birdnet_root_path=birdnet_root_path,
    #                 species=s,
    #                 # limit=4,
    #             )
    #         ],
    #         workers=max(int(os.cpu_count() / 2), 1),
    #         # scheduler_host="luigi.us-central1-a.c.birdclef-2023.internal",
    #         log_level="INFO",
    #     )
    # gs://birdclef-2023/data/processed/birdclef-2023/train_durations_v2.parquet
    df = pd.read_parquet("data/processed/birdclef-2023/train_durations_v2.parquet")

    track_names = [(r.filename, r.duration) for r in df.itertuples()]
    random.shuffle(track_names)
    if args.skip_existing:
        # check if the track embedding has already been computed
        track_names = [
            t
            for t in track_names
            if not (
                Path(args.output_path)
                / "embeddings"
                / t[0].replace(".ogg", "")
                / "_SUCCESS"
            ).exists()
        ]
    # track_names = [t for t in track_names if "bltapa1/XC621907" in t[0]]
    print(f"Found {len(track_names)} tracks to process")
    if args.dry_run:
        exit(0)

    # Actually, let's shuffle the tracks first so we don't get stuck on the same
    # tracks over and over. We can decrease the likelihood of getting stuck on
    # a single track by increasing the number luigi processes.
    for batch in chunks(track_names, args.batch_size):
        res = luigi.build(
            [
                TrackWorkflow(
                    birdclef_root_path=args.birdclef_root_path,
                    output_path=args.output_path,
                    birdnet_root_path=args.birdnet_root_path,
                    track_name=t,
                    duration=d,
                )
                for (t, d) in batch
            ],
            workers=args.workers,
            scheduler_host=args.scheduler_host,
            log_level="INFO",
            # get the full response so we can check for errors
            detailed_summary=True,
        )
