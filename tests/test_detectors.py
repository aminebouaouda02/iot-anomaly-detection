import numpy as np
import pandas as pd

from src.baseline import rolling_zscore_detect
from src.evaluation import compute_metrics, per_type_recall
from src.features import build_features
from src.ml_detector import IsolationForestDetector
from src.telemetry_generator import generate_telemetry


def test_metrics_perfect_prediction():
    m = compute_metrics(np.array([0, 1, 1, 0]), np.array([0, 1, 1, 0]))
    assert m == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 2, "fp": 0, "fn": 0}


def test_metrics_counts_errors():
    m = compute_metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5


def test_baseline_flags_an_obvious_spike():
    n = 300
    df = pd.DataFrame(
        {
            "temperature": np.full(n, 22.0),
            "humidity": np.full(n, 55.0),
            "vibration": np.full(n, 0.5),
        }
    )
    df.loc[200, "temperature"] = 45.0
    flags = rolling_zscore_detect(df)
    assert flags[200] == 1


def test_features_shape():
    df = generate_telemetry(days=1, seed=5)
    feats = build_features(df)
    assert feats.shape[0] == len(df)
    assert feats.shape[1] == 12  # 4 features x 3 sensors


def test_isolation_forest_fit_predict_roundtrip(tmp_path):
    df = generate_telemetry(days=2, seed=11)
    feats = build_features(df).dropna()
    detector = IsolationForestDetector(contamination=0.04).fit(feats)
    preds = detector.predict(feats)
    assert preds.shape == (len(feats),)
    assert set(np.unique(preds)) <= {0, 1}

    path = str(tmp_path / "model.joblib")
    detector.save(path)
    loaded = IsolationForestDetector.load(path)
    np.testing.assert_array_equal(loaded.predict(feats), preds)


def test_per_type_recall():
    types = pd.Series(["spike", "spike", "drop", ""])
    y_true = np.array([1, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    result = per_type_recall(types, y_true, y_pred)
    assert result == {"drop": 1.0, "spike": 0.5}
