from luigi import LocalTarget
from luigi.contrib.gcs import GCSTarget


def single_file_target(path):
    path = str(path)
    return GCSTarget(path) if path.startswith("gs://") else LocalTarget(path)
