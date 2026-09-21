from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class CatBoostPredictor:
    def __init__(self, bundle_path: str | Path = "ml/artifacts/catboost_bundle.pkl") -> None:
        self.bundle_path = Path(bundle_path)
        self.load()

    def load(self) -> None:
        bundle = joblib.load(self.bundle_path)
        self.model = bundle["model"]
        self.feature_names = bundle["feature_names"]
        self.categorical_features = bundle.get("categorical_features", [])
        self.calibrator = bundle.get("calibrator")

    def reload(self) -> None:
        self.load()

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()

        X = X.reindex(columns=self.feature_names)

        for col in self.feature_names:
            if col in self.categorical_features:
                X[col] = X[col].fillna("Unknown").astype(str)
            else:
                X[col] = (
                    pd.to_numeric(X[col], errors="coerce")
                    .replace([float("inf"), float("-inf")], float("nan"))
                    .fillna(0)
                )

        return X[self.feature_names]

    def predict_proba(self, df: pd.DataFrame) -> float:
        X = self._prepare_features(df)
        if self.calibrator is not None:
            raw_scores = self.model.predict(X, prediction_type="RawFormulaVal")
            return float(self.calibrator.predict(raw_scores)[0])
        return float(self.model.predict_proba(X)[:, 1][0])
