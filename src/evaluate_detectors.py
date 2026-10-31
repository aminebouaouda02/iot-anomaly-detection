"""Head-to-head evaluation: z-score baseline vs Isolation Forest.

Protocol (chronological, no leakage):
- The series is split by time: the first `train_days` are the training
  period, the rest is the test period.
- The baseline needs no training; it is scored on the test period.
- The Isolation Forest is fitted on the training period only (unsupervised)
  and scored on the test period.
- Both are measured against the injected ground-truth labels, overall and
  per anomaly type. Results are written to results/metrics.json.
"""

from __future__ import annotations

import json

import pandas as pd

from src.baseline import rolling_zscore_detect
from src.evaluation import compute_metrics, per_type_recall
from src.features import build_features
from src.ml_detector import IsolationForestDetector

DATA_PATH = "data/telemetry.csv"
RESULTS_PATH = "results/metrics.json"
MODEL_PATH = "models/isolation_forest.joblib"
TRAIN_DAYS = 3


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    split_at = df["timestamp"].min() + pd.Timedelta(days=TRAIN_DAYS)
    is_train = (df["timestamp"] < split_at).to_numpy()

    features = build_features(df).dropna()
    feat_index = features.index
    train_mask = is_train[feat_index]
    test_mask = ~train_mask

    y_all = df["is_anomaly"].to_numpy()
    types_all = df["anomaly_type"].fillna("")

    # Baseline, scored on test rows that have full feature context.
    baseline_pred_all = rolling_zscore_detect(df)
    y_test = y_all[feat_index][test_mask]
    types_test = types_all.iloc[feat_index][test_mask]
    baseline_pred = baseline_pred_all[feat_index][test_mask]

    # Isolation Forest, fitted on the training period only.
    detector = IsolationForestDetector(contamination=0.04)
    detector.fit(features[train_mask])
    ml_pred = detector.predict(features[test_mask])

    import os

    os.makedirs("models", exist_ok=True)
    detector.save(MODEL_PATH)

    results = {
        "dataset_rows": int(len(df)),
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "test_anomalies": int(y_test.sum()),
        "baseline_zscore": {
            **compute_metrics(y_test, baseline_pred),
            "per_type_recall": per_type_recall(types_test, y_test, baseline_pred),
        },
        "isolation_forest": {
            **compute_metrics(y_test, ml_pred),
            "per_type_recall": per_type_recall(types_test, y_test, ml_pred),
        },
    }

    os.makedirs("results", exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    for name, block in (("Baseline z-score", results["baseline_zscore"]),
                        ("Isolation Forest", results["isolation_forest"])):
        print(
            f"{name}: precision={block['precision']:.3f} recall={block['recall']:.3f} "
            f"f1={block['f1']:.3f} (tp={block['tp']}, fp={block['fp']}, fn={block['fn']})"
        )
        print(f"  per-type recall: {block['per_type_recall']}")
    print(f"Wrote {RESULTS_PATH} and {MODEL_PATH}")


if __name__ == "__main__":
    main()
