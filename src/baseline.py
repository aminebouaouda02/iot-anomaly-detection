"""Baseline detector: rolling z-score per sensor.

For each sensor, a point is flagged when its value deviates from the mean of
the previous `window` minutes by more than `threshold` standard deviations.
Simple, interpretable, and the reference every ML detector must beat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.telemetry_generator import SENSORS


def rolling_zscore_detect(
    df: pd.DataFrame, window: int = 60, threshold: float = 3.5
) -> np.ndarray:
    flags = np.zeros(len(df), dtype=int)
    for col in SENSORS:
        s = df[col].astype(float)
        mean = s.rolling(window, min_periods=10).mean().shift(1)
        std = s.rolling(window, min_periods=10).std().shift(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = ((s - mean) / std).abs()
        flagged = (z > threshold).fillna(False).to_numpy().astype(int)
        flags = np.maximum(flags, flagged)
    return flags
