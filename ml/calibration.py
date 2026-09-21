"""An isolated, chronological CatBoost calibration experiment."""
from dataclasses import asdict, dataclass
from contextlib import nullcontext
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve

from ml.preprocess import preprocess_dataset


@dataclass(frozen=True)
class CalibrationPeriods:
    train_start: str = "2025-06-01"
    validation_start: str = "2025-08-01"
    calibration_start: str = "2025-09-01"
    test_start: str = "2025-10-01"
    test_end: str = "2025-11-01"

    def boundaries(self):
        boundaries = pd.to_datetime(list(asdict(self).values()))
        if not all(left < right for left, right in zip(boundaries[:-1], boundaries[1:])):
            raise ValueError("Calibration period boundaries must be strictly increasing")
        return boundaries


class SigmoidCalibrator:
    """Platt-style sigmoid fitted to CatBoost raw margins, without class weights."""
    def fit(self, raw_scores, target):
        self.model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        self.model.fit(np.asarray(raw_scores).reshape(-1, 1), target)
        if self.model.coef_[0, 0] <= 0:
            raise ValueError("Calibration would reverse the risk ranking; inspect the data")
        return self

    def predict(self, raw_scores):
        return self.model.predict_proba(np.asarray(raw_scores).reshape(-1, 1))[:, 1]


def split_by_time(dataset, periods=CalibrationPeriods(), *, as_of=None):
    required = {"ApplicationId", "ApplicationDate", "MaxOverdueDays90"}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"Re-extract calibration data; missing columns: {sorted(missing)}")
    dates = pd.to_datetime(dataset["ApplicationDate"], errors="raise")
    if dates.isna().any():
        raise ValueError("ApplicationDate contains missing values")
    boundaries = periods.boundaries()
    selected = dataset.loc[(dates >= boundaries[0]) & (dates < boundaries[-1])].copy()
    if selected["ApplicationId"].isna().any() or selected["ApplicationId"].duplicated().any():
        raise ValueError("Expected exactly one row per ApplicationId")
    selected["ApplicationDate"] = dates.loc[selected.index]
    as_of = pd.Timestamp(as_of or pd.Timestamp.today().normalize())
    mature = selected["ApplicationDate"] + pd.Timedelta(days=90) <= as_of
    observed = selected["MaxOverdueDays90"].notna()
    usable = selected.loc[mature & observed].copy()
    outcome = pd.to_numeric(usable["MaxOverdueDays90"], errors="raise")
    if not np.isfinite(outcome).all() or (outcome < 0).any():
        raise ValueError("Invalid MaxOverdueDays90 values")
    usable["Target"] = (outcome >= 25).astype(int)

    splits, rows = {}, []
    for name, start, end in zip(
        ("train", "validation", "calibration", "test"), boundaries[:-1], boundaries[1:]
    ):
        in_period = (selected["ApplicationDate"] >= start) & (selected["ApplicationDate"] < end)
        frame = usable.loc[(usable["ApplicationDate"] >= start) & (usable["ApplicationDate"] < end)]
        if frame.empty or frame["Target"].nunique() != 2:
            raise ValueError(f"{name}: need nonempty data with both target classes")
        X = preprocess_dataset(frame.drop(columns=["Target", "MaxOverdueDays90"]))
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna("Unknown").astype(str)
        splits[name] = (X, frame["Target"])
        rows.append({
            "split": name, "start": start.date().isoformat(), "end_exclusive": end.date().isoformat(),
            "rows": len(frame), "defaults": int(frame["Target"].sum()),
            "default_rate": float(frame["Target"].mean()),
            "excluded_missing_outcome": int((in_period & ~observed).sum()),
            "excluded_immature": int((in_period & ~mature).sum()),
        })
    return splits, pd.DataFrame(rows)


def probability_metrics(y, proba):
    auc = float(roc_auc_score(y, proba))
    fpr, tpr, _ = roc_curve(y, proba)
    return {
        "auc": auc, "gini": 2 * auc - 1, "ks": float(np.max(tpr - fpr)),
        "brier": float(brier_score_loss(y, proba)), "log_loss": float(log_loss(y, proba)),
        "mean_probability": float(np.mean(proba)), "default_rate": float(np.mean(y)),
    }


def reliability_table(y, proba, variant):
    frame = pd.DataFrame({"probability": np.asarray(proba), "target": np.asarray(y)})
    frame["bin"] = pd.cut(frame["probability"], np.linspace(0, 1, 11), include_lowest=True)
    result = frame.groupby("bin", observed=True).agg(
        count=("target", "size"), mean_probability=("probability", "mean"),
        observed_default_rate=("target", "mean"),
    ).reset_index()
    result["bin"] = result["bin"].astype(str)
    result.insert(0, "variant", variant)
    return result


def train_calibrated_catboost(dataset, output_dir, *, periods=CalibrationPeriods(),
                             as_of=None, iterations=1000, log_mlflow=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits, summary = split_by_time(dataset, periods, as_of=as_of)
    summary.to_csv(output_dir / "split_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    X_train, y_train = splits["train"]
    categorical = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    model = CatBoostClassifier(
        iterations=iterations, learning_rate=0.03, depth=6, loss_function="Logloss",
        eval_metric="AUC", random_seed=42, verbose=100, thread_count=4, allow_writing_files=False,
    )
    model.fit(X_train, y_train, cat_features=categorical,
              eval_set=splits["validation"], use_best_model=True)
    X_cal, y_cal = splits["calibration"]
    calibrator = SigmoidCalibrator().fit(model.predict(X_cal, prediction_type="RawFormulaVal"), y_cal)
    # The test set is used only after the model and sigmoid have been fitted.
    X_test, y_test = splits["test"]
    predictions = {
        "raw": model.predict_proba(X_test)[:, 1],
        "sigmoid": calibrator.predict(model.predict(X_test, prediction_type="RawFormulaVal")),
    }
    metrics = pd.DataFrame([
        {"variant": variant, **probability_metrics(y_test, proba)}
        for variant, proba in predictions.items()
    ])
    metrics.to_csv(output_dir / "calibration_metrics.csv", index=False)
    reliability = pd.concat([
        reliability_table(y_test, proba, variant) for variant, proba in predictions.items()
    ], ignore_index=True)
    reliability.to_csv(output_dir / "calibration_curve.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    for variant, table in reliability.groupby("variant"):
        ax.plot(table["mean_probability"], table["observed_default_rate"], "o-", label=variant)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed default rate",
           title="CatBoost: October 2025 holdout", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_curve.png", dpi=150)
    plt.close(fig)

    metadata = {
        "periods": asdict(periods), "as_of": str(as_of or pd.Timestamp.today().date()),
        "target": "MaxOverdueDays90 >= 25", "best_iteration": model.get_best_iteration(),
        "calibration": "sigmoid_on_raw_margin", "auto_promoted": False,
    }
    base_bundle = {"model": model, "feature_names": X_train.columns.tolist(),
                   "categorical_features": categorical, "metadata": metadata}
    model.save_model(str(output_dir / "catboost.cbm"))
    joblib.dump({**base_bundle, "calibrator": None}, output_dir / "raw_bundle.pkl")
    joblib.dump({**base_bundle, "calibrator": calibrator}, output_dir / "catboost_bundle.pkl")
    (output_dir / "experiment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if log_mlflow:
        import mlflow
        from ml.mlflow_setup import setup_mlflow
        setup_mlflow()
        with (nullcontext() if mlflow.active_run() else mlflow.start_run(run_name="CatBoostTemporalCalibration")):
            mlflow.log_params({**asdict(periods), "iterations": iterations,
                               "best_iteration": model.get_best_iteration(), "calibration": "sigmoid"})
            for row in metrics.to_dict("records"):
                variant = row.pop("variant")
                mlflow.log_metrics({f"test_{variant}_{key}": value for key, value in row.items()})
            mlflow.log_artifacts(str(output_dir), artifact_path="calibration")
    print(metrics.to_string(index=False), flush=True)
    return metrics
