from pathlib import Path

import lightning as pl
import pandas as pd
from lightning.pytorch.profilers import AdvancedProfiler
from tqdm import tqdm

from birdclef.config import DEFAULT_VOCALIZATION_MODEL_PATH, SPECIES
from birdclef.experiment.label.data import GoogleVocalizationSoundscapeDataModule
from birdclef.experiment.model import LinearClassifier, TwoLayerClassifier


def make_submission(
    soundscape_path: str,
    metadata_path: str,
    output_csv_path: str,
    model_path: str,
    model_type: str = "linear",
    vocalization_model_path: str = DEFAULT_VOCALIZATION_MODEL_PATH,
    batch_size: int = 32,
    num_workers: int = 0,
    use_compiled: bool = True,
    limit=None,
    should_profile=False,
    profile_path="perf_logs",
):
    Path(output_csv_path).parent.mkdir(exist_ok=True, parents=True)
    dm = GoogleVocalizationSoundscapeDataModule(
        soundscape_path=soundscape_path,
        metadata_path=metadata_path,
        model_path=vocalization_model_path,
        batch_size=batch_size,
        use_compiled=use_compiled,
        num_workers=num_workers,
        limit=limit,
    )
    kwargs = dict()
    if should_profile:
        Path(profile_path).mkdir(exist_ok=True, parents=True)
        profiler = AdvancedProfiler(dirpath="logs", filename=profile_path)
        kwargs["profiler"] = profiler
    trainer = pl.Trainer(**kwargs)

    if model_type == "linear":
        model_class = LinearClassifier
    elif model_type == "two_layer":
        model_class = TwoLayerClassifier
    else:
        raise ValueError(f"invalid class: {model_type}")

    model = model_class.load_from_checkpoint(model_path)
    predictions = trainer.predict(model, dm)

    rows = []
    for batch in tqdm(predictions):
        for row_id, prediction in zip(batch["row_id"], batch["prediction"]):
            predictions = zip(SPECIES, prediction.numpy().tolist())
            row = {"row_id": row_id, **dict(predictions)}
            rows.append(row)
    submission_df = pd.DataFrame(rows)[["row_id", *SPECIES]]
    submission_df.to_csv(output_csv_path, index=False)
    return submission_df


if __name__ == "__main__":
    # this is for testing the performance against soundscape data
    import luigi

    from birdclef.tasks import RsyncGCSFiles

    luigi.build(
        [
            RsyncGCSFiles(
                src_path="gs://dsgt-clef-birdclef-2024/data/raw/birdclef-2024/unlabeled_soundscapes",
                dst_path="/mnt/data/raw/birdclef-2024/unlabeled_soundscapes",
            ),
            RsyncGCSFiles(
                src_path="gs://dsgt-clef-birdclef-2024/models",
                dst_path="/mnt/data/models",
            ),
        ],
        scheduler_host="services.us-central1-a.c.dsgt-clef-2024.internal",
    )

    # 10 samples in 570 seconds
    # 20 soundscapes compiled in 2:48
    model_name = "torch-v1-google-linear-asl-gamma_neg4-gamma_pos1"
    model_type = "linear"
    make_submission(
        "/mnt/data/raw/birdclef-2024/unlabeled_soundscapes",
        "gs://dsgt-clef-birdclef-2024/data/raw/birdclef-2024/train_metadata.csv",
        "/mnt/data/tmp/submission.csv",
        f"/mnt/data/models/{model_name}/checkpoints/last.ckpt",
        model_type,
        num_workers=0,
        limit=5,
        should_profile=True,
        profile_path="vocalization_linear_compiled",
    )
