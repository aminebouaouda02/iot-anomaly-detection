"""Streamlit dashboard: sensor series with detected anomalies highlighted.

Run:  streamlit run src/dashboard.py
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.api import get_detector
from src.dashboard_data import available_days, day_slice, score_frame
from src.telemetry_generator import SENSORS

st.set_page_config(page_title="IoT Anomaly Detection", layout="wide")
st.title("IoT / SCADA Telemetry — Anomaly Detection")


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv("data/telemetry.csv", parse_dates=["timestamp"])


@st.cache_resource
def load_detector():
    return get_detector()


df = load_data()
days = available_days(df)
day = st.sidebar.selectbox("Day", days, format_func=lambda d: str(pd.Timestamp(d).date()))
sensors = st.sidebar.multiselect("Sensors", SENSORS, default=list(SENSORS))
show_truth = st.sidebar.checkbox("Show ground-truth labels", value=True)

day_df = day_slice(df, day)
scored = score_frame(day_df, load_detector())

col1, col2, col3 = st.columns(3)
col1.metric("Points scored", len(scored))
col2.metric("Predicted anomalies", int(scored["prediction"].sum()))
col3.metric("Labeled anomalies (truth)", int(scored["is_anomaly"].sum()))

fig, axes = plt.subplots(len(sensors), 1, figsize=(12, 2.6 * len(sensors)), sharex=True)
if len(sensors) == 1:
    axes = [axes]
for ax, sensor in zip(axes, sensors):
    ax.plot(scored["timestamp"], scored[sensor], lw=0.8, label=sensor)
    flagged = scored[scored["prediction"] == 1]
    ax.scatter(flagged["timestamp"], flagged[sensor], color="red", s=14, label="predicted anomaly")
    if show_truth:
        truth = scored[scored["is_anomaly"] == 1]
        ax.scatter(
            truth["timestamp"], truth[sensor], facecolors="none",
            edgecolors="green", s=55, label="ground truth",
        )
    ax.set_ylabel(sensor)
    ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
st.pyplot(fig)

st.subheader("Predicted anomalies")
st.dataframe(
    scored.loc[scored["prediction"] == 1, ["timestamp", *SENSORS, "anomaly_type"]],
    width="stretch",
)

if os.path.exists("results/metrics.json"):
    st.subheader("Measured model metrics (test period)")
    with open("results/metrics.json", encoding="utf-8") as fh:
        st.json(json.load(fh))
