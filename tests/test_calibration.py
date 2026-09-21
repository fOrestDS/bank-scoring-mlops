import joblib
import numpy as np
import pandas as pd
import pytest

from ml.calibration import CalibrationPeriods, SigmoidCalibrator, split_by_time, train_calibrated_catboost
from ml.inference import CatBoostPredictor


def sample_data():
    rng = np.random.default_rng(42)
    rows = []
    for month in (6, 7, 8, 9, 10):
        for i in range(40):
            rows.append({
                "ApplicationId": month * 100 + i,
                "ApplicationDate": pd.Timestamp(2025, month, 1) + pd.Timedelta(days=i % 28),
                "MaxOverdueDays90": 30 if i % 2 else 0,
                "income": float(rng.normal(100 + 20 * (i % 2), 30)),
                "OS": "Android" if i % 3 else "iOS",
            })
    return pd.DataFrame(rows)


def test_splits_are_disjoint_and_exclude_metadata():
    data = sample_data()
    splits, summary = split_by_time(data, as_of="2026-06-20")
    assert summary["rows"].tolist() == [80, 40, 40, 40]
    indices = []
    for X, y in splits.values():
        assert set(X.columns) == {"income", "OS"}
        assert X.index.equals(y.index)
        indices.extend(X.index)
    assert len(indices) == len(set(indices)) == len(data)
    assert data.loc[splits["calibration"][0].index, "ApplicationDate"].dt.month.eq(9).all()


def test_unknown_outcomes_are_excluded_not_labeled_good():
    data = sample_data()
    data.loc[0, "MaxOverdueDays90"] = np.nan
    splits, summary = split_by_time(data, as_of="2026-06-20")
    assert 0 not in splits["train"][0].index
    assert summary.iloc[0]["excluded_missing_outcome"] == 1


def test_immature_data_is_not_used():
    with pytest.raises(ValueError, match="test"):
        split_by_time(sample_data(), as_of="2025-12-29")


@pytest.mark.parametrize("defect", ["date", "duplicate", "one_class", "missing_metadata"])
def test_invalid_data_is_rejected(defect):
    data = sample_data()
    if defect == "date":
        data.loc[0, "ApplicationDate"] = pd.NaT
    elif defect == "duplicate":
        data = pd.concat([data, data.iloc[:1]], ignore_index=True)
    elif defect == "one_class":
        data["MaxOverdueDays90"] = 0
    else:
        data = data.drop(columns="ApplicationDate")
    with pytest.raises(ValueError):
        split_by_time(data, as_of="2026-06-20")


def test_invalid_periods_are_rejected():
    with pytest.raises(ValueError, match="increasing"):
        split_by_time(sample_data(), CalibrationPeriods(test_start="2025-08-01"))


def test_training_roundtrip_and_calibration_uses_september_only(tmp_path, monkeypatch):
    fitted_labels = []
    original = SigmoidCalibrator.fit
    def capture_fit(self, raw, y):
        fitted_labels.extend(y.index)
        return original(self, raw, y)
    monkeypatch.setattr(SigmoidCalibrator, "fit", capture_fit)
    data = sample_data()
    metrics = train_calibrated_catboost(data, tmp_path, iterations=20,
                                       as_of="2026-06-20", log_mlflow=False)
    assert data.loc[fitted_labels, "ApplicationDate"].dt.month.eq(9).all()
    assert set(metrics.variant) == {"raw", "sigmoid"}
    assert np.isfinite(metrics.drop(columns="variant").to_numpy()).all()
    assert metrics.iloc[0].auc == pytest.approx(metrics.iloc[1].auc)
    row = data.iloc[[0]][["income", "OS"]]
    bundle = joblib.load(tmp_path / "catboost_bundle.pkl")
    expected = bundle["calibrator"].predict(bundle["model"].predict(row, prediction_type="RawFormulaVal"))[0]
    predictor = CatBoostPredictor(tmp_path / "catboost_bundle.pkl")
    assert predictor.predict_proba(row) == pytest.approx(expected)
    # Existing uncalibrated bundles remain supported, including reload from calibrated to raw.
    raw_bundle = joblib.load(tmp_path / "raw_bundle.pkl")
    raw_bundle.pop("calibrator")
    joblib.dump(raw_bundle, tmp_path / "legacy.pkl")
    predictor.bundle_path = tmp_path / "legacy.pkl"
    predictor.reload()
    assert predictor.calibrator is None
    assert predictor.predict_proba(row) == pytest.approx(bundle["model"].predict_proba(row)[0, 1])
    assert (tmp_path / "calibration_curve.png").exists()
