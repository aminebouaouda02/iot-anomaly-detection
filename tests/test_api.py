from fastapi.testclient import TestClient

from src.api import app
from src.telemetry_generator import generate_telemetry

client = TestClient(app)


def _records(n=180, seed=21):
    df = generate_telemetry(days=1, seed=seed).head(n)
    return [
        {
            "timestamp": str(row.timestamp),
            "temperature": float(row.temperature),
            "humidity": float(row.humidity),
            "vibration": float(row.vibration),
        }
        for row in df.itertuples()
    ]


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "isolation_forest"
    assert body["features"] == 12


def test_model_info_serves_measured_metrics():
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert "isolation_forest" in body and "baseline_zscore" in body
    assert body["test_anomalies"] > 0


def test_detect_rejects_short_window():
    response = client.post("/detect", json={"records": _records(10)})
    assert response.status_code == 422


def test_detect_flags_an_obvious_spike():
    records = _records(180)
    records[-1]["temperature"] += 20.0  # unmistakable spike on the last point
    response = client.post("/detect", json={"records": records})
    assert response.status_code == 200
    body = response.json()
    assert body["evaluated_points"] == 180 - 29
    assert body["anomaly_count"] >= 1
    assert body["results"][-1]["is_anomaly"] is True
