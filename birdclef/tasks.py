import tempfile
from textwrap import dedent

import luigi.contrib.gcs
from luigi.contrib.external_program import ExternalProgramTask


def maybe_gcs_target(path: str) -> luigi.Target:
    """Return a GCS target if the path starts with gs://, otherwise a LocalTarget."""
    if path.startswith("gs://"):
        return luigi.contrib.gcs.GCSTarget(path)
    else:
        return luigi.LocalTarget(path)


class BashScriptTask(ExternalProgramTask):
    def script_text(self) -> str:
        """The contents of to write to a bash script for running."""
        return dedent(
            """
            #!/bin/bash
            echo 'hello world'
            exit 1
            """
        )

    def program_args(self):
        """Execute the script."""
        script_text = self.script_text().strip()
        script_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        script_file.write(script_text)
        script_file.close()
        print(f"Script file: {script_file.name}")
        print(script_text)
        return ["/bin/bash", script_file.name]


class RsyncGCSFiles(BashScriptTask):
    """Download using the gcloud command-line tool."""

    src_path = luigi.Parameter()
    dst_path = luigi.Parameter()

    def output(self):
        path = f"{self.dst_path}/_SUCCESS"
        if path.startswith("gs://"):
            return luigi.contrib.gcs.GCSTarget(path)
        else:
            return luigi.LocalTarget(path)

    def script_text(self) -> str:
        return dedent(
            f"""
            #!/bin/bash
            set -eux -o pipefail
            gcloud storage rsync -r {self.src_path} {self.dst_path}
            """
        )

    def run(self):
        super().run()
        with self.output().open("w") as f:
            f.write("")
