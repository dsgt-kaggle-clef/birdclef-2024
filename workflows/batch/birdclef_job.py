from argparse import REMAINDER, ArgumentParser
from datetime import datetime

from google.cloud import batch_v1


def ds_nodash():
    return datetime.now().strftime("%Y%m%d%H%M%S")


def create_script_job(
    project_id: str,
    region: str,
    job_name: str,
    template_link: str,
    branch: str,
    cmd: str,
) -> batch_v1.Job:
    """
    This method shows how to create a sample Batch Job that will run
    a simple command on Cloud Compute instances.

    https://cloud.google.com/batch/docs/create-run-job#create-basic-job
    """
    client = batch_v1.BatchServiceClient()

    # Define what will be done as part of the job.
    task = batch_v1.TaskSpec(
        runnables=[
            batch_v1.Runnable(
                script=batch_v1.Runnable.Script(text=f"run-birdclef {branch} {cmd}")
            )
        ],
        compute_resource=batch_v1.ComputeResource(cpu_milli=8000, memory_mib=30_000),
        max_retry_count=0,
        max_run_duration="3600s",
    )

    job = batch_v1.Job(
        task_groups=[
            batch_v1.TaskGroup(
                task_count=1,
                task_spec=task,
            )
        ],
        allocation_policy=batch_v1.AllocationPolicy(
            instances=[
                batch_v1.AllocationPolicy.InstancePolicyOrTemplate(
                    instance_template=template_link
                )
            ]
        ),
        labels={"env": "prod", "type": "script"},
        logs_policy=batch_v1.LogsPolicy(
            destination=batch_v1.LogsPolicy.Destination.CLOUD_LOGGING
        ),
    )

    return client.create_job(
        batch_v1.CreateJobRequest(
            job=job,
            job_id=job_name,
            parent=f"projects/{project_id}/locations/{region}",
        )
    )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--project_id", type=str, default="birdclef-2023")
    parser.add_argument("--region", type=str, default="us-central1")
    parser.add_argument("--template_link", type=str, default="batch-cpu-template")
    parser.add_argument("--branch", type=str, required=True)
    parser.add_argument("--cmd", nargs=REMAINDER, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    job_name = f"birdclef-job-{ds_nodash()}"
    create_script_job(
        args.project_id,
        args.region,
        job_name,
        args.template_link,
        args.branch,
        " ".join(args.cmd),
    )
    print(f"Created job {job_name}.")
    print(
        "See https://console.cloud.google.com/batch/jobsDetail/regions/"
        f"{args.region}/jobs/{job_name}?project={args.project_id}"
    )


if __name__ == "__main__":
    main()
