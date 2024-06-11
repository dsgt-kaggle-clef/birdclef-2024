from pathlib import Path

import lightning as pl
import pandas as pd
import torch
from tqdm import tqdm

from birdclef.config import SPECIES

from .data import SoundscapeDataModule


class PassthroughModel(pl.LightningModule):
    def forward(self, x):
        return x

    def predict_step(self, batch, batch_idx):
        batch["prediction"] = torch.ones((batch["row_id"].shape[0], len(SPECIES))) * 0.5
        return batch


def make_submission(
    soundscape_path: str,
    metadata_path: str,
    output_csv_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
    limit=None,
    **kwargs,
):
    Path(output_csv_path).parent.mkdir(exist_ok=True, parents=True)
    dm = SoundscapeDataModule(
        soundscape_path=soundscape_path,
        metadata_path=metadata_path,
        batch_size=batch_size,
        num_workers=num_workers,
        limit=limit,
    )
    kwargs = dict()
    trainer = pl.Trainer(**kwargs)
    predictions = trainer.predict(PassthroughModel(), dm)

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
        ],
        scheduler_host="services.us-central1-a.c.dsgt-clef-2024.internal",
    )

    # 10 samples in 570 seconds
    make_submission(
        "/mnt/data/raw/birdclef-2024/unlabeled_soundscapes",
        "gs://dsgt-clef-birdclef-2024/data/raw/birdclef-2024/train_metadata.csv",
        "/mnt/data/tmp/submission.csv",
        num_workers=2,
        limit=20,
        should_profile=False,
    )
