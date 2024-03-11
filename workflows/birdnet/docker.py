import os
import tempfile
from pathlib import Path

import luigi
from luigi.contrib.docker_runner import DockerTask


class BirdNetAnalyzeTask(DockerTask):
    """Run BirdNet on the training set."""

    birdclef_root_path = luigi.Parameter()
    output_path = luigi.Parameter()
    species = luigi.Parameter()
    n_threads = luigi.IntParameter(default=4, significant=False)

    image = "us-central1-docker.pkg.dev/birdclef-2023/birdclef-2023/birdnet:latest"
    environment = {"NUMBA_CACHE_DIR": "/tmp"}
    container_options = {
        "user": f"{os.getuid()}:{os.getgid()}",
    }

    @property
    def staging_path(self):
        # a singleton variable that's set on the first call, hacky solution
        if not hasattr(self, "_staging_path"):
            self._staging_path = Path(tempfile.mkdtemp())
        return self._staging_path

    @property
    def binds(self):
        root = Path("./data").absolute()

        # create a temporary file to mount as the error log
        _, error_log_path = tempfile.mkstemp()

        return [
            f"{root}:/mnt/data",
            f"{self.staging_path}:/mnt/staging_output",
            # we also need to somehow allow the errors to be written
            f"{error_log_path}:/error_log.txt",
        ]

    def requires(self):
        return []

    def output(self):
        return luigi.LocalTarget(f"{self.output_path}/{self.species}")

    @property
    def command(self):
        return " ".join(
            [
                "analyze.py",
                f"--i /mnt/{self.birdclef_root_path}/train_audio/{self.species}",
                f"--o /mnt/staging_output",
                f"--threads {self.n_threads}",
                "--rtype csv",
            ]
        )

    def run(self):
        super().run()
        # move the intermediate results to the output, ensuring parents exist
        self.output().makedirs()
        print(f"Moving {self.staging_path} to {self.output().path}")
        Path(self.staging_path).rename(self.output().path)


class BirdNetEmbeddingsTask(BirdNetAnalyzeTask):
    @property
    def command(self):
        return " ".join(
            [
                "embeddings.py",
                f"--i /mnt/{self.birdclef_root_path}/train_audio/{self.species}",
                f"--o /mnt/staging_output",
                f"--threads {self.n_threads}",
                "--batchsize 16",
            ]
        )


# wrapper task to run BirdNet on all species
class BirdNetAnalyzeTaskAllTask(luigi.WrapperTask):
    birdclef_root_path = luigi.Parameter()
    output_path = luigi.Parameter()
    n_threads = luigi.IntParameter(default=4, significant=False)
    limit = luigi.IntParameter(default=-1)

    upstream_task_type = BirdNetAnalyzeTask

    def requires(self):
        species = sorted(
            [p.name for p in Path(f"{self.birdclef_root_path}/train_audio").glob("*")]
        )
        if self.limit > -1:
            species = species[: self.limit]
        for specie in species:
            yield self.upstream_task_type(
                birdclef_root_path=self.birdclef_root_path,
                output_path=self.output_path,
                species=specie,
                n_threads=self.n_threads,
            )


class BirdNetEmbeddingsTaskAllTask(BirdNetAnalyzeTaskAllTask):
    upstream_task_type = BirdNetEmbeddingsTask
