"""Live MQTT mode: score telemetry as it arrives on a broker topic.

Publish JSON records to the topic `factory/telemetry`, e.g.
{"timestamp": "2026-10-01T00:00:00", "temperature": 22.1, "humidity": 55.0,
 "vibration": 0.48}
— the same publish/subscribe pattern as the ENSAM SCADA coursework
(Mosquitto broker, one topic, JSON payloads).

Each new record is appended to a rolling buffer; once at least 31 records
are buffered, the newest point is scored with the trained detector and the
result is printed as a JSON line. The buffer/scoring core (TelemetryBuffer)
has no MQTT dependency so it can be unit-tested without a broker.

Run:  python -m src.mqtt_live --broker localhost --port 1883
"""

from __future__ import annotations

import argparse
import json
from collections import deque

import pandas as pd

from src.features import build_features
from src.ml_detector import IsolationForestDetector

DEFAULT_TOPIC = "factory/telemetry"
MIN_RECORDS = 31
SENSOR_KEYS = ("temperature", "humidity", "vibration")


def parse_payload(payload: bytes) -> dict:
    """Parse and validate one MQTT message payload."""
    record = json.loads(payload.decode("utf-8"))
    for key in SENSOR_KEYS:
        record[key] = float(record[key])
    return record


class TelemetryBuffer:
    """Rolling buffer that scores the newest point once it has context."""

    def __init__(self, detector: IsolationForestDetector, maxlen: int = 1000):
        self.detector = detector
        self.records: deque[dict] = deque(maxlen=maxlen)

    def add(self, record: dict) -> dict | None:
        self.records.append(record)
        if len(self.records) < MIN_RECORDS:
            return None
        df = pd.DataFrame(list(self.records))
        features = build_features(df).dropna()
        if features.empty:
            return None
        prediction = int(self.detector.predict(features)[-1])
        return {
            "index": len(self.records) - 1,
            "timestamp": record.get("timestamp"),
            "is_anomaly": bool(prediction),
        }


def run_live(broker: str, port: int, topic: str, detector: IsolationForestDetector) -> None:
    import paho.mqtt.client as mqtt

    buffer = TelemetryBuffer(detector)

    def on_message(_client, _userdata, message):
        try:
            record = parse_payload(message.payload)
        except (ValueError, KeyError, TypeError) as exc:
            print(json.dumps({"error": f"bad payload: {exc}"}), flush=True)
            return
        result = buffer.add(record)
        if result is not None:
            print(json.dumps(result), flush=True)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    client.connect(broker, port)
    client.subscribe(topic)
    print(json.dumps({"status": "subscribed", "topic": topic, "broker": broker}), flush=True)
    client.loop_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broker", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    args = parser.parse_args()

    from src.api import get_detector

    run_live(args.broker, args.port, args.topic, get_detector())


if __name__ == "__main__":
    main()
