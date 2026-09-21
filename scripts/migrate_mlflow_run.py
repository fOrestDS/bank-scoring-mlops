"""Copy an existing run to HTTP artifact storage without retraining or deleting it."""
import argparse
from pathlib import Path

from mlflow import MlflowClient

from ml.mlflow_setup import setup_mlflow


def migrate_run(client, source_run_id, source_artifacts, experiment_id):
    source_artifacts = Path(source_artifacts)
    if not source_artifacts.is_dir() or not any(source_artifacts.iterdir()):
        raise ValueError("Source artifact directory is missing or empty")
    source = client.get_run(source_run_id)
    if source.info.status != "FINISHED":
        raise ValueError("Only completed runs can be migrated")
    # Reuse a successfully migrated copy if this command is repeated.
    for existing in client.search_runs([experiment_id]):
        if (existing.data.tags.get("migration.source_run_id") == source_run_id
                and existing.info.status == "FINISHED"):
            return existing.info.run_id
    tags = {key: value for key, value in source.data.tags.items()
            if not key.startswith("mlflow.")}
    tags["migration.source_run_id"] = source_run_id
    tags["migration.source_artifact_uri"] = source.info.artifact_uri
    target = client.create_run(
        experiment_id, tags=tags, run_name=source.info.run_name,
        start_time=source.info.start_time,
    )
    run_id = target.info.run_id
    try:
        for key, value in source.data.params.items():
            client.log_param(run_id, key, value)
        for key in source.data.metrics:
            for metric in client.get_metric_history(source_run_id, key):
                client.log_metric(run_id, key, metric.value,
                                  timestamp=metric.timestamp, step=metric.step)
        client.log_artifacts(run_id, str(source_artifacts))
        client.set_terminated(run_id, status="FINISHED", end_time=source.info.end_time)
    except Exception:
        client.set_terminated(run_id, status="FAILED")
        raise
    return run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifacts", required=True, type=Path,
                        help="Local root of the source run's artifacts (including calibration/)")
    args = parser.parse_args()
    experiment = setup_mlflow()
    client = MlflowClient()
    run_id = migrate_run(client, args.run_id, args.artifacts, experiment.experiment_id)
    print(f"experiment_id={experiment.experiment_id}; run_id={run_id}")
    print(f"artifact_uri={client.get_run(run_id).info.artifact_uri}")


if __name__ == "__main__":
    main()
