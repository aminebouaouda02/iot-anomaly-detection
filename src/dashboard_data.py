"""Data helpers for the dashboard (kept free of Streamlit so they are testable)."""

from __future__ import annotations

import pandas as pd

from src.features import build_features
from src.ml_detector import IsolationForestDetector


def available_days(df: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(df["timestamp"].dt.normalize().unique())


def day_slice(df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    mask = df["timestamp"].dt.normalize() == pd.Timestamp(day)
    return df.loc[mask].reset_index(drop=True)


def score_frame(df: pd.DataFrame, detector: IsolationForestDetector) -> pd.DataFrame:
    """Return the rows that have full feature context, with a prediction column."""
    features = build_features(df).dropna()
    scored = df.loc[features.index].copy()
    scored["prediction"] = detector.predict(features)
    return scored
