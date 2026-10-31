"""Replay the committed dataset onto an MQTT topic (demo/testing feeder).

Publishes rows of data/telemetry.csv as JSON records to `factory/telemetry`,
so the live subscriber (src/mqtt_live.py) can be demonstrated end to end
with any MQTT broker, e.g. the Mosquitto container from the SCADA coursework.

Run:  python -m src.mqtt_simulator --broker localhost --rows 120 --interval 0.1
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broker", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic", default="factory/telemetry")
    parser.add_argument("--rows", type=int, default=120)
    parser.add_argument("--interval", type=float, default=0.1, help="seconds between messages")
    args = parser.parse_args()

    import paho.mqtt.client as mqtt

    df = pd.read_csv("data/telemetry.csv").head(args.rows)
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(args.broker, args.port)
    client.loop_start()
    for row in df.itertuples():
        record = {
            "timestamp": row.timestamp,
            "temperature": float(row.temperature),
            "humidity": float(row.humidity),
            "vibration": float(row.vibration),
        }
        client.publish(args.topic, json.dumps(record))
        time.sleep(args.interval)
    client.loop_stop()
    client.disconnect()
    print(f"Published {len(df)} records to {args.topic} on {args.broker}:{args.port}")


if __name__ == "__main__":
    main()
