from typing import Any

from fastapi import APIRouter, HTTPException

from app.schemas.prediction import PredictRequest, PredictResponse
from app.services.scoring_service import scoring_service

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("/by-id", response_model=PredictResponse)
def predict_by_id(payload: PredictRequest):
    try:
        return scoring_service.score_by_id(payload.application_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/by-features", response_model=PredictResponse)
def predict_by_features(payload: dict[str, Any]):
    try:
        return scoring_service.score_by_features(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))