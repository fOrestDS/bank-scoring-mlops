from unittest.mock import MagicMock, Mock

import mlflow
import pytest

from ml import mlflow_setup
from scripts import run_calibration


@pytest.mark.parametrize("fails", [False, True])
def test_run_starts_before_work_and_records_airflow_context(monkeypatch, fails):
    monkeypatch.setattr("sys.argv", ["run_calibration", "--extract"])
    monkeypatch.setenv("AIRFLOW_CTX_DAG_ID", "weekly_retrain")
    monkeypatch.setenv("AIRFLOW_CTX_TASK_ID", "calibrate_catboost")
    monkeypatch.setenv("AIRFLOW_CTX_DAG_RUN_ID", "manual_example")
    monkeypatch.setattr(mlflow_setup, "setup_mlflow", Mock())
    context = MagicMock()
    start = Mock(return_value=context)
    monkeypatch.setattr(mlflow, "start_run", start)
    tags = Mock()
    monkeypatch.setattr(mlflow, "set_tags", tags)
    def work(args):
        context.__enter__.assert_called_once()
        assert args.extract
        if fails:
            raise RuntimeError("training failed")
    monkeypatch.setattr(run_calibration, "run_experiment", work)
    if fails:
        with pytest.raises(RuntimeError, match="training failed"):
            run_calibration.main()
        assert context.__exit__.call_args.args[0] is RuntimeError
    else:
        run_calibration.main()
        assert context.__exit__.call_args.args[0] is None
    start.assert_called_once_with(run_name="CatBoostTemporalCalibration")
    tags.assert_called_once_with({"airflow.dag_id": "weekly_retrain",
                                  "airflow.task_id": "calibrate_catboost",
                                  "airflow.run_id": "manual_example"})


def test_no_mlflow_still_runs_experiment(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_calibration", "--no-mlflow"])
    setup = Mock()
    monkeypatch.setattr(mlflow_setup, "setup_mlflow", setup)
    work = Mock()
    monkeypatch.setattr(run_calibration, "run_experiment", work)
    run_calibration.main()
    work.assert_called_once()
    setup.assert_not_called()
