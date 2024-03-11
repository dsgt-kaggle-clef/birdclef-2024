from datetime import datetime

from google.cloud import batch_v1


def ds_nodash():
    return datetime.now().strftime("%Y%m%d%H%M%S")


def create_script_job(project_id: str, region: str, job_name: str) -> batch_v1.Job:
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
                script=batch_v1.Runnable.Script(
                    text=(
                        "echo Hello world! This is task ${BATCH_TASK_INDEX}. "
                        "This job has a total of ${BATCH_TASK_COUNT} tasks."
                    )
                )
            )
        ],
        compute_resource=batch_v1.ComputeResource(cpu_milli=2000, memory_mib=16),
        max_retry_count=2,
        max_run_duration="3600s",
    )

    job = batch_v1.Job(
        task_groups=[
            batch_v1.TaskGroup(
                task_count=4,
                task_spec=task,
            )
        ],
        allocation_policy=batch_v1.AllocationPolicy(
            instances=[
                batch_v1.AllocationPolicy.InstancePolicyOrTemplate(
                    policy=batch_v1.AllocationPolicy.InstancePolicy(
                        machine_type="e2-standard-4"
                    )
                )
            ]
        ),
        labels={"env": "testing", "type": "script"},
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


if __name__ == "__main__":
    job_name = f"test-job-{ds_nodash()}"
    create_script_job("birdclef-2023", "us-central1", job_name)
    print(f"Created job {job_name}.")
