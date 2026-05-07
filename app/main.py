import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException

from app.model import feature_cols, predict_rul
from app.schemas import BearingFeatures, RULPrediction


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(api_app: FastAPI) -> AsyncIterator[None]:
    logger.info(
        "model_loaded",
        extra={"model": "xgb_rul_femto", "n_features": len(feature_cols)},
    )
    yield


app = FastAPI(
    title="RUL Prediction API",
    version="1.0.0",
    description="XGBoost bearing RUL predictor trained on FEMTO dataset.",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": "xgb_rul_femto", "n_features": 44}


@app.post("/predict", response_model=RULPrediction)
async def predict(features: BearingFeatures) -> dict:
    try:
        return predict_rul(features.model_dump())
    except Exception as prediction_error:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from prediction_error
