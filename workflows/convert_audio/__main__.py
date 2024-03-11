import os
from pathlib import Path

import luigi
from google.cloud import storage

from workflows.convert_audio.ogg_to_mp3 import OggToMP3
from workflows.utils.pull import Pull
from workflows.utils.push import Push


class ConvertAudioTask(luigi.WrapperTask):
    """Wrapper around the entire DAG to convert .ogg files to .mp3"""

    birdclef_root_path = luigi.Parameter()
    intermediate_path = luigi.Parameter()
    output_path = luigi.Parameter()
    species = luigi.Parameter()

    n_threads = luigi.IntParameter(default=4)
    parallelism = luigi.IntParameter(default=os.cpu_count())

    def requires(self):
        pull_data = Pull(
            input_path=os.path.join(self.birdclef_root_path, self.species),
            output_path=os.path.join(self.intermediate_path, "ogg", self.species),
            parallelism=self.parallelism,
        )
        yield pull_data

        ogg_to_mp3 = OggToMP3(
            input_path=os.path.join(self.intermediate_path, "ogg", self.species),
            output_path=os.path.join(self.intermediate_path, "mp3", self.species),
            parallelism=self.parallelism,
            dynamic_requires=[pull_data],
        )
        yield ogg_to_mp3

        push_data = Push(
            input_path=os.path.join(self.intermediate_path, "mp3", self.species),
            output_path=os.path.join(self.output_path, self.species),
            parallelism=self.parallelism,
            dynamic_requires=[pull_data, ogg_to_mp3],
        )
        yield push_data


def list_species(birdclef_root_path):
    storage_client = storage.Client("birdclef-2023")

    bucket = storage_client.get_bucket("birdclef-2023")

    species = bucket.list_blobs(prefix=birdclef_root_path)

    species_list = []
    for blob in species:
        name = blob.name
        name = name.replace(birdclef_root_path + "/", "")
        name = name.split("/")[0]
        species_list.append(name.strip())

    return list(set(species_list))


if __name__ == "__main__":
    species_list = list_species("data/raw/birdclef-2022/train_audio")
    for species in species_list:
        n_threads = os.cpu_count()
        luigi.build(
            [
                ConvertAudioTask(
                    birdclef_root_path="data/raw/birdclef-2022/train_audio",
                    intermediate_path="data/raw/birdclef-2022/train_audio",
                    output_path="data/processed/mp3",
                    n_threads=n_threads,
                    parallelism=os.cpu_count(),
                    species=species,
                ),
            ],
            scheduler_host="luigi.us-central1-a.c.birdclef-2023.internal",
            workers=os.cpu_count() // n_threads,
        )
