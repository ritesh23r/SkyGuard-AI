
import json
import time
import random
from datetime import datetime

import pandas as pd
import requests

# -----------------------------------------
# CONFIGURATION
# -----------------------------------------

API_URL = "http://127.0.0.1:8000/predict"
CSV_FILE = "iot_telemetry_data.csv"
HISTORY_FILE = "sensor_history.jsonl"
INTERVAL_SECONDS = 5

FEATURES = [
    "co",
    "humidity",
    "light",
    "lpg",
    "motion",
    "smoke",
    "temp"
]

# -----------------------------------------
# LOAD RECORDED SENSOR DATA
# -----------------------------------------

print("Loading sensor dataset...")

df = pd.read_csv(CSV_FILE)

missing = [col for col in FEATURES if col not in df.columns]

if missing:
    raise ValueError(f"Missing dataset columns: {missing}")

df = df[FEATURES].dropna().copy()

df["light"] = df["light"].astype(bool)
df["motion"] = df["motion"].astype(bool)

print(f"Loaded {len(df)} sensor readings.")
print("Starting SkyGuard AI sensor simulation...")


# -----------------------------------------
# SEND READINGS AUTOMATICALLY
# -----------------------------------------

while True:
    row = df.iloc[random.randrange(len(df))]

    reading = {
        "co": float(row["co"]),
        "humidity": float(row["humidity"]),
        "light": bool(row["light"]),
        "lpg": float(row["lpg"]),
        "motion": bool(row["motion"]),
        "smoke": float(row["smoke"]),
        "temp": float(row["temp"])
    }

    try:
        response = requests.post(
            API_URL,
            json=reading,
            timeout=10
        )

        response.raise_for_status()
        result = response.json()

        record = {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Temperature": reading["temp"],
            "Humidity": reading["humidity"],
            "CO": reading["co"],
            "LPG": reading["lpg"],
            "Smoke": reading["smoke"],
            "Light": reading["light"],
            "Motion": reading["motion"],
            "Status": result["status"],
            "Score": result["anomaly_score"],
            "Reason": result.get(
                            "reason",
                            result.get("message", "No explanation returned.")
                        )
        }

        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        print(
            f"Temperature: {reading['temp']:.2f}°C | "
            f"Humidity: {reading['humidity']:.2f}% | "
            f"Status: {result['status']} | "
            f"Score: {result['anomaly_score']:.6f}"
        )

    except requests.RequestException as error:
        print("API request failed:", error)

    time.sleep(INTERVAL_SECONDS)
