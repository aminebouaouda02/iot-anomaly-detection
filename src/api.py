"""FastAPI service exposing the trained Isolation Forest detector.

Endpoints:
- GET  /health      service and model status
- GET  /model-info  the measured evaluation metrics (results/metrics.json)
- POST /detect      score a window of telemetry records

The detector needs context: features use a 30-minute rolling window, so a
request must carry at least 31 consecutive minute records. The first 30
points are context only; predictions start at index 29.

Model loading: if models/isolation_forest.joblib exists it is loaded;
otherwise the model is trained on the spot from data/telemetry.csv using
the same protocol as src/evaluate_detectors.py (first 3 days), then saved.
"""

from __future__ import annotations

import json
import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import build_features
from src.ml_detector import IsolationForestDetector

MODEL_PATH = "models/isolation_forest.joblib"
DATA_PATH = "data/telemetry.csv"
METRICS_PATH = "results/metrics.json"
TRAIN_DAYS = 3
MIN_RECORDS = 31

app = FastAPI(title="IoT Telemetry Anomaly Detection", version="1.0.0")

_detector: IsolationForestDetector | None = None


def _train_default() -> IsolationForestDetector:
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    split_at = df["timestamp"].min() + pd.Timedelta(days=TRAIN_DAYS)
    features = build_features(df).dropna()
    train_mask = (df.loc[features.index, "timestamp"] < split_at).to_numpy()
    detector = IsolationForestDetector(contamination=0.04)
    detector.fit(features[train_mask])
    os.makedirs("models", exist_ok=True)
    detector.save(MODEL_PATH)
    return detector


def get_detector() -> IsolationForestDetector:
    global _detector
    if _detector is None:
        if os.path.exists(MODEL_PATH):
            _detector = IsolationForestDetector.load(MODEL_PATH)
        else:
            _detector = _train_default()
    return _detector


class TelemetryRecord(BaseModel):
    timestamp: str | None = None
    temperature: float
    humidity: float
    vibration: float


class DetectRequest(BaseModel):
    records: list[TelemetryRecord] = Field(..., min_length=MIN_RECORDS)


class PointResult(BaseModel):
    index: int
    timestamp: str | None = None
    is_anomaly: bool


class DetectResponse(BaseModel):
    evaluated_points: int
    anomaly_count: int
    results: list[PointResult]


@app.get("/health")
def health() -> dict:
    detector = get_detector()
    return {
        "status": "ok",
        "model": "isolation_forest",
        "features": len(detector.feature_columns or []),
    }


@app.get("/model-info")
def model_info() -> dict:
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(status_code=404, detail="metrics not computed yet")
    with open(METRICS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@app.post("/detect", response_model=DetectResponse)
def detect(request: DetectRequest) -> DetectResponse:
    detector = get_detector()
    df = pd.DataFrame([r.model_dump() for r in request.records])
    features = build_features(df).dropna()
    predictions = detector.predict(features)

    results = [
        PointResult(
            index=int(idx),
            timestamp=df.loc[idx, "timestamp"],
            is_anomaly=bool(pred),
        )
        for idx, pred in zip(features.index, predictions)
    ]
    return DetectResponse(
        evaluated_points=len(results),
        anomaly_count=int(sum(r.is_anomaly for r in results)),
        results=results,
    )
