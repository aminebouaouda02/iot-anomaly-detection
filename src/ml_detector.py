"""ML detector: Isolation Forest on engineered telemetry features.

Trained unsupervised on a chronological training period (the model never sees
the test labels). Features are standardised with a scaler fitted on the
training period only. The trained model can be persisted with joblib and
reloaded by the API module.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class IsolationForestDetector:
    def __init__(self, contamination: float = 0.04, n_estimators: int = 200, random_state: int = 42):
        self.contamination = contamination
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self.scaler = StandardScaler()
        self.feature_columns: list[str] | None = None

    def fit(self, features: pd.DataFrame) -> "IsolationForestDetector":
        self.feature_columns = list(features.columns)
        x = self.scaler.fit_transform(features.to_numpy())
        self.model.fit(x)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        x = self.scaler.transform(features[self.feature_columns].to_numpy())
        raw = self.model.predict(x)  # 1 = normal, -1 = anomaly
        return (raw == -1).astype(int)

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "feature_columns": self.feature_columns,
                "contamination": self.contamination,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "IsolationForestDetector":
        data = joblib.load(path)
        detector = cls(contamination=data["contamination"])
        detector.model = data["model"]
        detector.scaler = data["scaler"]
        detector.feature_columns = data["feature_columns"]
        return detector
