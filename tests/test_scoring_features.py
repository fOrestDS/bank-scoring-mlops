from unittest.mock import Mock

import pandas as pd
import pytest

from app.services import scoring_service as service_module


def test_by_id_passes_aggregates_to_predictor(monkeypatch):
    predictor = Mock(feature_names=["active_count", "active_debt"])
    predictor.predict_proba.return_value = 0.2
    monkeypatch.setattr(service_module, "CatBoostPredictor", lambda: predictor)
    monkeypatch.setattr(service_module, "load_application_features", lambda _: pd.DataFrame({
        "ApplicationId": [123], "ec.ApplicationId": [123],
        "active_count": [3], "active_debt": [45000],
    }))
    result = service_module.ScoringService().score_by_id(123)
    row = predictor.predict_proba.call_args.args[0]
    assert row.to_dict("records") == [{"active_count": 3, "active_debt": 45000}]
    assert result == {"application_id": 123, "default_probability": 0.2, "risk_level": "Low"}


def test_by_id_rejects_missing_model_columns(monkeypatch):
    predictor = Mock(feature_names=["active_count", "active_debt"])
    monkeypatch.setattr(service_module, "CatBoostPredictor", lambda: predictor)
    monkeypatch.setattr(service_module, "load_application_features", lambda _: pd.DataFrame({
        "ApplicationId": [123], "active_count": [3],
    }))
    with pytest.raises(ValueError, match="Missing model features: active_debt"):
        service_module.ScoringService().score_by_id(123)
    predictor.predict_proba.assert_not_called()
