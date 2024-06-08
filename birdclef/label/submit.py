import pandas as pd
from tqdm import tqdm

from birdclef.config import DEFAULT_VOCALIZATION_MODEL_PATH, SPECIES

from .data import GoogleVocalizationSoundscapeDataModule


def make_submission(
    soundscape_path: str,
    metadata_path: str,
    output_csv_path: str,
    model_path: str = DEFAULT_VOCALIZATION_MODEL_PATH,
):
    dm = GoogleVocalizationSoundscapeDataModule(
        soundscape_path=soundscape_path,
        metadata_path=metadata_path,
        model_path=model_path,
    )
    dm.setup()
    predictions = dm.predict_dataloader()

    rows = []
    for batch in tqdm(predictions):
        for row_id, prediction in zip(batch["row_id"], batch["prediction"]):
            predictions = zip(SPECIES, prediction.numpy().tolist())
            row = {"row_id": row_id, **dict(predictions)}
            rows.append(row)
    submission_df = pd.DataFrame(rows)[["row_id", *SPECIES]]
    submission_df.to_csv(output_csv_path, index=False)
    return submission_df