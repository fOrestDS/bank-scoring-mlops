from typing import Any

import pandas as pd

from ml.extract import load_application_features
from ml.inference import CatBoostPredictor
from ml.preprocess import preprocess_dataset


class ScoringService:
    def __init__(self) -> None:
        self.predictor = CatBoostPredictor()

    def reload_model(self) -> None:
        self.predictor.reload()

    @staticmethod
    def _risk_level(proba: float) -> str:
        if proba >= 0.5:
            return "High"
        if proba >= 0.3:
            return "Medium"
        return "Low"

    def _score_dataframe(self, row: pd.DataFrame, application_id: int | None = None) -> dict:
        row = preprocess_dataset(row)
        row = row.drop(columns=["Target"], errors="ignore")

        proba = self.predictor.predict_proba(row)

        return {
            "application_id": application_id,
            "default_probability": round(proba, 4),
            "risk_level": self._risk_level(proba),
        }

    def score_by_id(self, application_id: int) -> dict:
        row = load_application_features(application_id)

        if row.empty:
            raise ValueError(f"ApplicationId {application_id} not found")

        missing = set(self.predictor.feature_names) - set(row.columns)
        if missing:
            raise ValueError(f"Missing model features: {', '.join(sorted(missing))}")

        return self._score_dataframe(row, application_id=application_id)

    def score_by_features(self, features: dict[str, Any]) -> dict:
        row = pd.DataFrame([features])
        return self._score_dataframe(row, application_id=None)


scoring_service = ScoringService()
