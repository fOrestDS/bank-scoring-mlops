# Bank Scoring MLOps

Bank credit scoring system with ClickHouse, FastAPI, CatBoost, Docker and reproducible ML pipeline.

## What it does

- pulls data from ClickHouse
- builds a training dataset from SQL extracts
- trains a baseline Logistic Regression model
- trains a CatBoost model
- compares models by AUC, Gini and KS
- serves scoring via FastAPI
- supports model reload without restart
- runs in Docker

## Project structure

- `app/` — FastAPI app
- `config/` — ClickHouse connection settings
- `ml/` — data loading, preprocessing, training, inference
- `scripts/` — runnable entry points
- `sql/` — SQL feature extracts
- `tests/` — smoke tests

## Main commands

Run the baseline model:
```powershell
python -m scripts.run_train_parquet
```

## CatBoost probability calibration

### MLflow storage

Start MLflow with `docker compose up -d mlflow`. Training clients use
`MLFLOW_TRACKING_URI` (default `http://localhost:5000` on Windows;
`http://mlflow:5000` in Compose) and experiment `bank-scoring-http`.
Override the experiment name with `MLFLOW_EXPERIMENT_NAME` if needed.
The server receives artifact uploads over HTTP and stores them in the mounted
`./mlartifacts` directory. Clients do not write directly to SQLite or depend on
Windows/Linux filesystem paths. The UI is at http://localhost:5000.

Old experiments keep their original artifact locations. Copy a completed run
to the HTTP-backed experiment without retraining:

```powershell
python -m scripts.migrate_mlflow_run --run-id SOURCE_RUN_ID --artifacts PATH_TO_RUN_ARTIFACTS
```

Pass the artifact root containing `calibration/`, not just that subdirectory.
The copy preserves parameters, metric history, run times and user tags, and
records its source run ID. The source is retained. Repeating a successful
migration reuses the completed copy. In the UI select `bank-scoring-http`, open
`CatBoostTemporalCalibration`, then **Artifacts → calibration** for PNG and CSV
reports. Scalar results appear in **Metrics**.

### Run the experiment

Run an isolated experiment using the configured ClickHouse connection:

```powershell
python -m scripts.run_calibration --extract --as-of 2026-09-20
```

The command saves a separate dataset in `data/calibration/dataset.parquet` and
artifacts in a timestamped directory under `ml/artifacts/calibration/`.
To repeat training on the saved snapshot, omit `--extract`.
Use `--output-dir PATH` to specify the artifact directory, and `--no-mlflow`
to disable experiment tracking.

| Period (2025) | Purpose |
| --- | --- |
| June–July | Fit CatBoost |
| August | Select the best boosting iteration |
| September | Fit sigmoid calibration on raw CatBoost margins |
| October | Evaluate raw and calibrated probabilities once |

Application IDs, dates and outcome columns are excluded from model inputs.
Rows with missing outcomes or less than 90 days of observation as of `--as-of`
are excluded; each period must contain both target classes. The observation
cutoff is supplied by the operator and must reflect the source data's coverage.
The report includes excluded counts, event counts and event rates. Applications
without credit history are excluded by the same inner join used for training.

Artifacts:

- `split_summary.csv`: sample counts and event rates by period.
- `calibration_metrics.csv`: October AUC, Gini, KS, Brier score, LogLoss,
  mean prediction and observed event rate for both variants.
- `calibration_curve.csv` and `.png`: reliability diagram with bin counts in CSV.
- `raw_bundle.pkl`: the fitted CatBoost without calibration.
- `catboost_bundle.pkl`: the same CatBoost plus its fitted sigmoid calibrator.
- `experiment.json`: period boundaries and experiment metadata.

`CatBoostPredictor(bundle_path=...)` supports both calibrated and legacy bundles.
The experiment does not overwrite the API's model or the existing promotion
reports. The weekly DAG runs `calibrate_catboost` after `benchmark_models` as a
separate reporting branch. Each attempt uses its own dataset and artifact path.
The calibration run appears in MLflow before extraction/training starts and
includes `airflow.dag_id`, `airflow.task_id` and `airflow.run_id` tags. A failed
calibration task fails that DAG run; it does not automatically promote a model.
The experiment still uses the fixed June–October 2025 periods above, rather than
a moving window. Existing completed Airflow runs are not automatically rerun.
Historical random-split metrics must not be compared directly with this October
holdout. Calibration is not selected or refitted using October labels.

This is a retrospective period comparison, not a simulation of when a model
could have been deployed: 90-day labels become available after application dates.
The existing SQL also computes `CH_length` relative to `today()`; retain the
saved dataset for reproducibility. A production historical backtest needs a
separate audit of feature availability at the application time.

Run the calibration and inference tests:

```powershell
python -m pytest tests/test_calibration.py tests/test_extract.py tests/test_scoring_features.py tests/test_api.py -q
```
