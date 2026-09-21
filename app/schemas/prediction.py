from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    application_id: int = Field(..., ge=1)


class PredictResponse(BaseModel):
    application_id: int | None = None
    default_probability: float
    risk_level: str