import json
import os
from pathlib import Path

import luigi
from google.cloud import storage
from pyspark.sql import functions as F

from workflows.utils.pull import Pull
from workflows.utils.push import Push

from .plotting import ClusterPlotAllTasks


def get_species_list(path):
    agreement_path = Path(
        f"{path}/birdnet-embeddings-with-neighbors-agreement-static/v1/agreement.json"
    )
    return [t["ego_primary_label"] for t in json.loads(agreement_path.read_text())]


if __name__ == "__main__":
    data_path = (
        Path(__file__).parent.parent.parent.parent / "data/processed/birdclef-2022"
    )
    os.makedirs(data_path)
    os.makedirs(data_path / "birdnet-embeddings-with-neighbors-static/v1")
    luigi.build(
        [
            Pull(
                input_path="data/processed/birdclef-2022/birdnet-embeddings-with-neighbors/v1",
                output_path=f"{str(data_path)}/birdnet-embeddings-with-neighbors/v1",
                parallelism=os.cpu_count(),
            ),
            Pull(
                input_path="data/processed/birdclef-2022/birdnet-embeddings-with-neighbors-agreement-static/v1",
                output_path=f"{str(data_path)}/birdnet-embeddings-with-neighbors-agreement-static/v1",
                parallelism=os.cpu_count(),
            ),
        ],
        scheduler_host="luigi.us-central1-a.c.birdclef-2023.internal",
        workers=2,
    )
    species_list = get_species_list(str(data_path))
    storage_client = storage.Client("birdclef-2023")
    bucket = storage_client.get_bucket("birdclef-2023")
    for species in species_list:
        blobs = list(
            bucket.list_blobs(
                prefix=f"data/processed/birdclef-2022/birdnet-embeddings-with-neighbors-static/v1/{species}"
            )
        )
        if len(blobs) == 4:
            luigi.build(
                [
                    Pull(
                        input_path=f"data/processed/birdclef-2022/birdnet-embeddings-with-neighbors-static/v1/{species}",
                        output_path=f"{data_path}/birdnet-embeddings-with-neighbors-static/v1/{species}",
                        parallelism=os.cpu_count(),
                    )
                ],
                scheduler_host="luigi.us-central1-a.c.birdclef-2023.internal",
                workers=2,
            )
    luigi.build(
        [
            ClusterPlotAllTasks(
                spark_path=str(data_path / "birdnet-embeddings-with-neighbors/v1"),
                local_path=str(
                    data_path / "birdnet-embeddings-with-neighbors-static/v1"
                ),
                total_cnt=len(species_list),
            ),
        ],
        scheduler_host="luigi.us-central1-a.c.birdclef-2023.internal",
        workers=2,
    )
    for species in species_list:
        luigi.build(
            [
                Push(
                    input_path=str(
                        data_path
                        / "birdnet-embeddings-with-neighbors-static/v1"
                        / species
                    ),
                    output_path=f"data/processed/birdclef-2022/birdnet-embeddings-with-neighbors-static/v1/{species}",
                    parallelism=os.cpu_count(),
                    dynamic_requires=[],
                )
            ],
            scheduler_host="luigi.us-central1-a.c.birdclef-2023.internal",
            workers=2,
        )
