
import os
import json
from collections import deque
from datetime import datetime, timezone
from contextlib import contextmanager

import joblib
import pandas as pd
import psycopg2
from psycopg2.extras import Json, RealDictCursor

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="SkyGuard AI API")

# Load the trained model and configuration
model = joblib.load("models/skyguard_isolation_forest.joblib")

with open("models/skyguard_config.json", "r") as f:
    config = json.load(f)

features = config["features"]

# Temporary cache; PostgreSQL will be the permanent history.
sensor_history = deque(maxlen=1000)


# ---------------- DATABASE ----------------

DATABASE_URL = os.getenv("DATABASE_URL")


@contextmanager
def get_db_connection():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is missing")

    connection = psycopg2.connect(DATABASE_URL)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


@app.on_event("startup")
def initialize_database():
    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sensor_history (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    sensor_reading JSONB NOT NULL,
                    prediction INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    anomaly_score DOUBLE PRECISION NOT NULL,
                    reason TEXT NOT NULL,
                    message TEXT NOT NULL
                );
            """)


# ---------------- INPUT MODEL ----------------

class SensorReading(BaseModel):
    co: float = Field(ge=0)
    humidity: float = Field(ge=0, le=100)
    light: bool
    lpg: float = Field(ge=0)
    motion: bool
    smoke: float = Field(ge=0)
    temp: float


# ---------------- HOME ----------------

@app.get("/")
def home():
    return {"message": "SkyGuard AI API is running"}


# ---------------- PREDICTION ----------------

@app.post("/predict")
def predict(reading: SensorReading):
    data = reading.model_dump()

    X_new = pd.DataFrame([data], columns=features)
    X_new["light"] = X_new["light"].astype(int)
    X_new["motion"] = X_new["motion"].astype(int)

    prediction = int(model.predict(X_new)[0])
    score = float(-model.score_samples(X_new)[0])

    if prediction == -1:
        status = "anomaly"
        message = "Unusual sensor pattern detected"
    else:
        status = "normal"
        message = "No anomaly detected by the model"

    result = {
        "prediction": prediction,
        "status": status,
        "anomaly_score": round(score, 6),
        "reason": (
            "The sensor reading differs from patterns "
            "learned by the Isolation Forest model."
            if prediction == -1
            else "The model did not flag this reading as unusual."
        ),
        "message": message,
    }

    timestamp = datetime.now(timezone.utc)

    record = {
        "timestamp": timestamp.isoformat(),
        "sensor_reading": data,
        **result,
    }

    # Save permanently to PostgreSQL.
    try:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO sensor_history
                    (timestamp, sensor_reading, prediction, status,
                     anomaly_score, reason, message)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        timestamp,
                        Json(data),
                        result["prediction"],
                        result["status"],
                        result["anomaly_score"],
                        result["reason"],
                        result["message"],
                    ),
                )
    except Exception:
        # Don't report success if the reading wasn't saved.
        raise HTTPException(
            status_code=503,
            detail="Prediction was calculated, but saving to the database failed.",
        )

    # Update the temporary cache after the database write succeeds.
    sensor_history.append(record)

    return result


# ---------------- HISTORY ----------------

@app.get("/history")
def get_history(limit: int = 1000):
    if limit < 1 or limit > 1000:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 1000",
        )

    try:
        with get_db_connection() as connection:
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT timestamp, sensor_reading, prediction,
                           status, anomaly_score, reason, message
                    FROM sensor_history
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()

        # Return oldest-to-newest, matching the dashboard's timeline.
        rows.reverse()

        history = []
        for row in rows:
            item = dict(row)
            item["timestamp"] = item["timestamp"].isoformat()
            history.append(item)

        return history

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Could not retrieve history from the database.",
        )


# ---------------- TEMPERATURE DROP ----------------

class TemperatureContext(BaseModel):
    previous_temp: float
    current_temp: float
    next_temp: float


@app.post("/check-temperature-drop")
def check_temperature_drop(reading: TemperatureContext):
    is_zero_temperature = reading.current_temp == 0

    is_sudden_drop = (
        reading.current_temp < 10
        and reading.previous_temp > 15
        and reading.next_temp > 15
    )

    if is_zero_temperature and is_sudden_drop:
        status = "possible_zero_temperature_sensor_issue"
        message = (
            "Temperature is zero between higher readings. "
            "Check the sensor and the recorded data."
        )
    elif is_sudden_drop:
        status = "possible_temperature_drop"
        message = (
            "A sudden temperature drop was detected. "
            "Investigate the sensor reading."
        )
    else:
        status = "no_sudden_drop_detected"
        message = (
            "The specified readings do not match "
            "the temperature-drop rule."
        )

    return {
        "possible_drop": is_sudden_drop,
        "possible_zero_temperature": is_zero_temperature,
        "status": status,
        "message": message,
    }
