"""Run the calibration experiment without modifying serving or promotion artifacts."""
import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from ml.calibration import CalibrationPeriods, train_calibrated_catboost


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true", help="Fetch June through October 2025 from ClickHouse")
    parser.add_argument("--dataset", type=Path, default=Path("data/calibration/dataset.parquet"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("ml/artifacts/calibration") / datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--as-of", help="Date through which outcomes are observed (YYYY-MM-DD)")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    if args.no_mlflow:
        run_experiment(args)
        return

    import mlflow
    from ml.mlflow_setup import setup_mlflow
    setup_mlflow()
    # Create the run before extraction/training so progress and failures are visible.
    with mlflow.start_run(run_name="CatBoostTemporalCalibration") as run:
        context = {
            "airflow.dag_id": os.getenv("AIRFLOW_CTX_DAG_ID"),
            "airflow.task_id": os.getenv("AIRFLOW_CTX_TASK_ID"),
            "airflow.run_id": os.getenv("AIRFLOW_CTX_DAG_RUN_ID"),
        }
        mlflow.set_tags({key: value for key, value in context.items() if value})
        print(f"MLflow run_id={run.info.run_id}", flush=True)
        run_experiment(args)


def run_experiment(args):
    periods = CalibrationPeriods()
    if args.extract:
        from ml.extract import load_client_features, load_equifax_features
        print("Extracting application features and outcomes...", flush=True)
        clients = load_client_features(periods.train_start, periods.test_end)
        print(f"Client rows: {len(clients)}; extracting credit history aggregates...", flush=True)
        equifax = load_equifax_features(periods.train_start, periods.test_end)
        dataset = clients.merge(equifax, how="inner", left_on="ApplicationId",
                                right_on="ec.ApplicationId", validate="one_to_one")
        print(f"Joined rows: {len(dataset)}; clients without history: {len(clients) - len(dataset)}", flush=True)
        args.dataset.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(args.dataset, index=False)
    else:
        if not args.dataset.exists():
            parser.error("Calibration dataset not found. Run with --extract first.")
        dataset = pd.read_parquet(args.dataset)
    train_calibrated_catboost(dataset, args.output_dir, periods=periods, as_of=args.as_of,
                             iterations=args.iterations, log_mlflow=not args.no_mlflow)
    print(f"Artifacts: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
