from pathlib import Path

import luigi
from luigi.contrib.external_program import ExternalProgramTask
from luigi.contrib.gcs import GCSTarget

from workflows.utils.mixin import DynamicRequiresMixin

from .gcs import single_file_target


class GSUtilRsyncTask(ExternalProgramTask, DynamicRequiresMixin):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    is_dir = luigi.BoolParameter(default=True)
    touch_success = luigi.BoolParameter(default=True)
    capture_output = True

    def program_args(self):
        in_path = self.input_path.rstrip("/")
        out_path = self.output_path.rstrip("/")

        if self.is_dir:
            Path(out_path).mkdir(parents=True, exist_ok=True)
            return [
                "bash",
                "-ce",
                (
                    f"gsutil -m rsync -r {in_path}/ {out_path}/"
                    + (f" && touch {out_path}/_SUCCESS" if self.touch_success else "")
                ),
            ]
        else:
            return ["bash", "-ce", f"gsutil -m cp {in_path} {out_path}"]

    def output(self):
        out_path = self.output_path.rstrip("/")
        target = f"{out_path}/_SUCCESS" if self.is_dir else out_path
        return single_file_target(target)
