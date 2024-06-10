from birdclef.config import SPECIES
from birdclef.experiment.label.submit import make_submission


def test_make_submission(soundscape_path, metadata_full_path, tmp_path):
    submission_path = tmp_path / "submission.csv"
    df = make_submission(soundscape_path, metadata_full_path, submission_path)
    assert submission_path.exists()
    assert len(df) == 3 * 2
    assert set(df.columns) == {"row_id", *SPECIES}
    # assert values are between 0 and 1
    assert df[SPECIES].min().min() >= 0
    assert df[SPECIES].max().max() <= 1
