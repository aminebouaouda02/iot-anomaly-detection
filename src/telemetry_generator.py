"""Synthetic SCADA/IoT telemetry generator with labeled anomalies.

Generates realistic minute-level sensor series (temperature, humidity,
vibration) with daily cycles, noise and a slow drift, then injects known
anomalies and labels them. The labels are the ground truth used later to
evaluate detectors honestly (precision / recall against known anomalies).

Anomaly types:
- spike: short sharp rise in temperature (1-3 minutes)
- drop:  short sharp fall in humidity (5-15 minutes)
- stuck: vibration sensor frozen at a constant value (30-60 minutes)
- drift: temperature slowly rising beyond the normal drift (about 2 hours)
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

SENSORS = ["temperature", "humidity", "vibration"]
ANOMALY_TYPES = ["spike", "drop", "stuck", "drift"]


def _base_signals(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    minutes = np.arange(n)
    daily = 2 * np.pi * minutes / (24 * 60)

    temperature = (
        22.0
        + 3.0 * np.sin(daily - 1.2)                # daily cycle, peak in the afternoon
        + np.linspace(0.0, 0.8, n)                 # slow upward drift over the period
        + rng.normal(0.0, 0.3, n)                  # measurement noise
    )
    humidity = (
        55.0
        - 8.0 * np.sin(daily - 1.2)                # inverse of temperature
        + rng.normal(0.0, 2.0, n)
    ).clip(15.0, 95.0)
    vibration = (
        0.50
        + 0.08 * np.sin(2 * np.pi * minutes / 180.0)  # machine cycle, 3 hours
        + rng.normal(0.0, 0.05, n)
    ).clip(0.05, None)

    return {"temperature": temperature, "humidity": humidity, "vibration": vibration}


def _inject_anomalies(
    signals: dict[str, np.ndarray], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    n = len(signals["temperature"])
    is_anomaly = np.zeros(n, dtype=int)
    anomaly_type = np.array([""] * n, dtype=object)

    def mark(start: int, length: int, kind: str) -> None:
        end = min(start + length, n)
        is_anomaly[start:end] = 1
        anomaly_type[start:end] = kind

    # Spikes on temperature: 6 events of 1-3 minutes, +8 to +12 degrees.
    for _ in range(6):
        start = int(rng.integers(0, n - 3))
        length = int(rng.integers(1, 4))
        signals["temperature"][start : start + length] += rng.uniform(8.0, 12.0)
        mark(start, length, "spike")

    # Drops on humidity: 4 events of 5-15 minutes, -20 to -30 points.
    for _ in range(4):
        start = int(rng.integers(0, n - 15))
        length = int(rng.integers(5, 16))
        signals["humidity"][start : start + length] -= rng.uniform(20.0, 30.0)
        mark(start, length, "drop")

    # Stuck vibration sensor: 3 events of 30-60 minutes at a constant value.
    for _ in range(3):
        start = int(rng.integers(0, n - 60))
        length = int(rng.integers(30, 61))
        signals["vibration"][start : start + length] = signals["vibration"][start]
        mark(start, length, "stuck")

    # Abnormal drift on temperature: 2 events of about 2 hours, +4 degrees ramp.
    for _ in range(2):
        start = int(rng.integers(0, n - 120))
        length = 120
        signals["temperature"][start : start + length] += np.linspace(0.0, 4.0, length)
        mark(start, length, "drift")

    signals["humidity"] = signals["humidity"].clip(0.0, 100.0)
    return is_anomaly, anomaly_type


def generate_telemetry(days: int = 7, seed: int = 42) -> pd.DataFrame:
    """Generate `days` of minute-level telemetry with labeled anomalies."""
    rng = np.random.default_rng(seed)
    n = days * 24 * 60
    signals = _base_signals(n, rng)
    is_anomaly, anomaly_type = _inject_anomalies(signals, rng)

    timestamps = pd.date_range("2026-10-01T00:00:00", periods=n, freq="min")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature": np.round(signals["temperature"], 3),
            "humidity": np.round(signals["humidity"], 3),
            "vibration": np.round(signals["vibration"], 4),
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/telemetry.csv")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate_telemetry(days=args.days, seed=args.seed)
    df.to_csv(args.output, index=False)

    total = len(df)
    anomalies = int(df["is_anomaly"].sum())
    print(f"Wrote {total} rows to {args.output}")
    print(f"Labeled anomalies: {anomalies} ({100.0 * anomalies / total:.2f}%)")
    print(df.loc[df["is_anomaly"] == 1, "anomaly_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
