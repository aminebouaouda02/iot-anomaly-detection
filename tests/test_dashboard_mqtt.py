import json

import pandas as pd

from src.api import get_detector
from src.dashboard_data import available_days, day_slice, score_frame
from src.mqtt_live import TelemetryBuffer, parse_payload
from src.telemetry_generator import generate_telemetry


def test_day_slice_returns_one_full_day():
    df = generate_telemetry(days=3, seed=4)
    days = available_days(df)
    assert len(days) == 3
    one = day_slice(df, days[1])
    assert len(one) == 24 * 60
    assert one["timestamp"].dt.normalize().nunique() == 1


def test_score_frame_adds_predictions():
    df = generate_telemetry(days=1, seed=6)
    scored = score_frame(df, get_detector())
    assert "prediction" in scored.columns
    assert len(scored) == len(df) - 29
    assert set(scored["prediction"].unique()) <= {0, 1}


def test_parse_payload_validates_and_converts():
    record = parse_payload(
        json.dumps({"temperature": "22.5", "humidity": 55, "vibration": 0.5}).encode()
    )
    assert record["temperature"] == 22.5
    assert isinstance(record["humidity"], float)


def test_buffer_waits_for_context_then_scores():
    df = generate_telemetry(days=1, seed=8).head(60)
    buffer = TelemetryBuffer(get_detector())
    results = []
    for row in df.itertuples():
        out = buffer.add(
            {
                "timestamp": str(row.timestamp),
                "temperature": float(row.temperature),
                "humidity": float(row.humidity),
                "vibration": float(row.vibration),
            }
        )
        if out is not None:
            results.append(out)
    assert results  # scoring started once 31 records were buffered
    assert results[0]["index"] == 30


def test_buffer_flags_an_obvious_spike():
    df = generate_telemetry(days=1, seed=9).head(80)
    buffer = TelemetryBuffer(get_detector())
    last = None
    for row in df.itertuples():
        last = buffer.add(
            {
                "timestamp": str(row.timestamp),
                "temperature": float(row.temperature),
                "humidity": float(row.humidity),
                "vibration": float(row.vibration),
            }
        )
    spiked = buffer.add(
        {"timestamp": "2026-10-01T01:20:00", "temperature": 45.0, "humidity": 55.0, "vibration": 0.5}
    )
    assert spiked is not None and spiked["is_anomaly"] is True
