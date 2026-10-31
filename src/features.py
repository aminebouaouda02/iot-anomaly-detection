"""Feature engineering for the ML detector.

Per sensor: the raw value, a rolling mean and rolling standard deviation
(context — a stuck sensor shows up as a rolling std of exactly 0), and the
minute-to-minute change (rate). 12 features in total.
"""

from __future__ import annotations

import pandas as pd

from src.telemetry_generator import SENSORS


def build_features(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)
    for col in SENSORS:
        s = df[col].astype(float)
        feats[col] = s
        feats[f"{col}_roll_mean"] = s.rolling(window).mean()
        feats[f"{col}_roll_std"] = s.rolling(window).std()
        feats[f"{col}_diff"] = s.diff()
    return feats
