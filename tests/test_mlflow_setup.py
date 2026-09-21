from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ml import mlflow_setup
from scripts.migrate_mlflow_run import migrate_run


@pytest.mark.parametrize("uri", [None, "http://mlflow:5000"])
def test_setup_uses_http_and_proxy_storage(monkeypatch, uri):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    if uri:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    tracking = Mock()
    experiment = SimpleNamespace(artifact_location="mlflow-artifacts:/2")
    select = Mock(return_value=experiment)
    monkeypatch.setattr(mlflow_setup.mlflow, "set_tracking_uri", tracking)
    monkeypatch.setattr(mlflow_setup.mlflow, "set_experiment", select)
    assert mlflow_setup.setup_mlflow() is experiment
    tracking.assert_called_once_with(uri or "http://localhost:5000")
    select.assert_called_once_with("bank-scoring-http")


def test_setup_rejects_direct_database(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    with pytest.raises(ValueError, match="HTTP"):
        mlflow_setup.setup_mlflow()


def test_setup_rejects_existing_local_experiment(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setattr(mlflow_setup.mlflow, "set_tracking_uri", Mock())
    monkeypatch.setattr(mlflow_setup.mlflow, "set_experiment",
                        Mock(return_value=SimpleNamespace(artifact_location="file:D:/mlruns")))
    with pytest.raises(ValueError, match="local artifact"):
        mlflow_setup.setup_mlflow()


def test_migration_preserves_metric_steps_and_is_repeatable(tmp_path):
    (tmp_path / "curve.png").write_bytes(b"example")
    client = Mock()
    source = SimpleNamespace(
        info=SimpleNamespace(status="FINISHED", run_name="Calibration", start_time=100,
                             end_time=200, artifact_uri="file:D:/mlruns/old"),
        data=SimpleNamespace(tags={"note": "original"}, params={"method": "sigmoid"}, metrics={"loss": 0.2}),
    )
    client.get_run.return_value = source
    client.search_runs.return_value = []
    client.create_run.return_value = SimpleNamespace(info=SimpleNamespace(run_id="new"))
    client.get_metric_history.return_value = [SimpleNamespace(value=0.2, timestamp=150, step=3)]
    assert migrate_run(client, "old", tmp_path, "2") == "new"
    client.log_metric.assert_called_once_with("new", "loss", 0.2, timestamp=150, step=3)
    client.log_artifacts.assert_called_once_with("new", str(tmp_path))
    client.set_terminated.assert_called_once_with("new", status="FINISHED", end_time=200)
    client.search_runs.return_value = [SimpleNamespace(
        info=SimpleNamespace(status="FINISHED", run_id="new"),
        data=SimpleNamespace(tags={"migration.source_run_id": "old"}),
    )]
    assert migrate_run(client, "old", tmp_path, "2") == "new"
    client.create_run.assert_called_once()


def test_migration_marks_failed_upload(tmp_path):
    (tmp_path / "curve.png").write_bytes(b"example")
    client = Mock()
    client.get_run.return_value = SimpleNamespace(
        info=SimpleNamespace(status="FINISHED", run_name="Calibration", start_time=100, artifact_uri="file:old"),
        data=SimpleNamespace(tags={}, params={}, metrics={}),
    )
    client.search_runs.return_value = []
    client.create_run.return_value = SimpleNamespace(info=SimpleNamespace(run_id="new"))
    client.log_artifacts.side_effect = RuntimeError("upload failed")
    with pytest.raises(RuntimeError, match="upload failed"):
        migrate_run(client, "old", tmp_path, "2")
    client.set_terminated.assert_called_once_with("new", status="FAILED")
