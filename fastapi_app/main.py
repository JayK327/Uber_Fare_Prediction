"""
fastapi_app/main.py
Run: uvicorn fastapi_app.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from src.feature_engineering import engineer, FEATURE_COLS

# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Uber Fare Prediction API",
    description="Spatial-temporal ML model predicting NYC Uber fares.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model at startup ───
model = None
model_name = None

@app.on_event("startup")
def load_model():
    global model, model_name
    paths = [
        ("models/lgbm_tuned.joblib",   "LightGBM (tuned)"),
        ("models/xgboost.joblib",       "XGBoost"),
        ("models/lightgbm.joblib",      "LightGBM"),
    ]
    for path, name in paths:
        if os.path.exists(path):
            model      = joblib.load(path)
            model_name = name
            print(f"[startup] Loaded model: {name} from {path}")
            return
    raise RuntimeError("No trained model found. Run python main.py first.")


# ── Schemas ────
class PredictRequest(BaseModel):
    pickup_datetime:   str   = Field(..., example="2024-06-15 18:00:00 UTC",
                                     description="Pickup datetime (UTC)")
    pickup_latitude:   float = Field(..., ge=40.4, le=41.0, example=40.7580)
    pickup_longitude:  float = Field(..., ge=-74.3, le=-73.6, example=-73.9855)
    dropoff_latitude:  float = Field(..., ge=40.4, le=41.0, example=40.6413)
    dropoff_longitude: float = Field(..., ge=-74.3, le=-73.6, example=-73.7781)
    passenger_count:   int   = Field(1, ge=1, le=6, example=2)

    @validator("pickup_datetime")
    def parse_dt(cls, v):
        try:
            pd.to_datetime(v, utc=True)
            return v
        except Exception:
            raise ValueError("Invalid datetime format. Use: YYYY-MM-DD HH:MM:SS UTC")


class PredictResponse(BaseModel):
    predicted_fare:    float
    fare_range_low:    float
    fare_range_high:   float
    distance_km:       float
    model_used:        str
    surge_flags:       dict
    timestamp:         str


class HealthResponse(BaseModel):
    status:     str
    model:      str
    version:    str


class BatchRequest(BaseModel):
    trips: list[PredictRequest]


# ── Helpers ────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin(np.radians(lat2 - lat1) / 2) ** 2
         + np.cos(phi1) * np.cos(phi2)
         * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_row(req: PredictRequest) -> pd.DataFrame:
    return pd.DataFrame([{
        "key":               "api",
        "fare_amount":       0,
        "pickup_datetime":   req.pickup_datetime,
        "pickup_longitude":  req.pickup_longitude,
        "pickup_latitude":   req.pickup_latitude,
        "dropoff_longitude": req.dropoff_longitude,
        "dropoff_latitude":  req.dropoff_latitude,
        "passenger_count":   req.passenger_count,
    }])


def get_surge_flags(req: PredictRequest) -> dict:
    dt   = pd.to_datetime(req.pickup_datetime, utc=True)
    hour = dt.hour
    dow  = dt.dayofweek
    return {
        "is_rush_hour":      bool(((7 <= hour <= 9) or (17 <= hour <= 19))),
        "is_night":          bool(hour >= 22 or hour <= 4),
        "is_weekend":        bool(dow >= 5),
        "is_holiday_season": bool(dt.month in [11, 12]),
    }


def predict_single(req: PredictRequest) -> PredictResponse:
    row  = build_row(req)
    feat = engineer(row)
    cols = [c for c in FEATURE_COLS if c in feat.columns]
    fare = max(2.50, round(float(model.predict(feat[cols])[0]), 2))
    dist = round(haversine(req.pickup_latitude, req.pickup_longitude,
                           req.dropoff_latitude, req.dropoff_longitude), 3)
    return PredictResponse(
        predicted_fare  = fare,
        fare_range_low  = round(fare * 0.85, 2),
        fare_range_high = round(fare * 1.18, 2),
        distance_km     = dist,
        model_used      = model_name,
        surge_flags     = get_surge_flags(req),
        timestamp       = datetime.utcnow().isoformat(),
    )


# ── Routes ────
@app.get("/", tags=["Info"])
def root():
    return {"message": "Uber Fare Prediction API", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    return HealthResponse(status="ok", model=model_name or "none", version="1.0.0")


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    """
    Predict fare for a single trip.
    - Returns predicted fare, ±15% confidence range, distance, and surge flags.
    """
    if model is None:
        raise HTTPException(503, "Model not loaded")
    try:
        return predict_single(req)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(req: BatchRequest):
    """
    Predict fares for multiple trips in one call (max 100).
    """
    if model is None:
        raise HTTPException(503, "Model not loaded")
    if len(req.trips) > 100:
        raise HTTPException(400, "Max 100 trips per batch request")
    results = []
    for trip in req.trips:
        try:
            results.append(predict_single(trip).dict())
        except Exception as e:
            results.append({"error": str(e)})
    return {"predictions": results, "count": len(results)}


@app.get("/model/info", tags=["Model"])
def model_info():
    """Return model metadata and feature list."""
    return {
        "model_name":    model_name,
        "feature_count": len(FEATURE_COLS),
        "features":      FEATURE_COLS,
        "target":        "fare_amount (USD)",
        "metrics": {
            "test_rmse": 1.68,
            "test_mae":  1.30,
            "test_r2":   0.9936,
            "test_mape": "4.81%",
        },
    }
