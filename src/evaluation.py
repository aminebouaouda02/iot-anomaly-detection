"""Evaluation helpers: score detector output against the ground-truth labels."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def per_type_recall(anomaly_type: pd.Series, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """For each anomaly type, the share of its labeled rows the detector caught."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    types = pd.Series(np.asarray(anomaly_type))
    result = {}
    for kind in sorted(t for t in types.unique() if t):
        mask = (types == kind).to_numpy() & (y_true == 1)
        total = int(mask.sum())
        if total:
            result[kind] = round(float(y_pred[mask].mean()), 4)
    return result
