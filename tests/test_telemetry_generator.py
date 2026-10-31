import pandas as pd

from src.telemetry_generator import ANOMALY_TYPES, SENSORS, generate_telemetry


def test_shape_and_columns():
    df = generate_telemetry(days=2, seed=1)
    assert len(df) == 2 * 24 * 60
    assert list(df.columns) == [
        "timestamp",
        "temperature",
        "humidity",
        "vibration",
        "is_anomaly",
        "anomaly_type",
    ]
    assert not df[SENSORS].isna().any().any()


def test_labels_are_consistent():
    df = generate_telemetry(days=2, seed=1)
    labeled = df[df["is_anomaly"] == 1]
    unlabeled = df[df["is_anomaly"] == 0]
    assert (labeled["anomaly_type"] != "").all()
    assert (unlabeled["anomaly_type"] == "").all()
    assert set(labeled["anomaly_type"].unique()) <= set(ANOMALY_TYPES)


def test_anomaly_fraction_is_realistic():
    df = generate_telemetry(days=7, seed=42)
    fraction = df["is_anomaly"].mean()
    assert 0.01 < fraction < 0.10


def test_deterministic_for_same_seed():
    a = generate_telemetry(days=1, seed=7)
    b = generate_telemetry(days=1, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_sensor_ranges_are_plausible():
    df = generate_telemetry(days=2, seed=3)
    assert df["temperature"].between(-5, 45).all()
    assert df["humidity"].between(0, 100).all()
    assert (df["vibration"] >= 0).all()
