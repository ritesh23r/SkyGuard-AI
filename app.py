
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import json
import pandas as pd

app = FastAPI(title="SkyGuard AI API")

# Load the trained model and configuration
model = joblib.load("skyguard_isolation_forest.joblib")

with open("skyguard_config.json", "r") as f:
    config = json.load(f)

features = config["features"]


# Define the incoming sensor data
class SensorReading(BaseModel):
    co: float = Field(ge=0)
    humidity: float = Field(ge=0, le=100)
    light: bool
    lpg: float = Field(ge=0)
    motion: bool
    smoke: float = Field(ge=0)
    temp: float


@app.get("/")
def home():
    return {"message": "SkyGuard AI API is running"}


@app.post("/predict")
def predict(reading: SensorReading):
    data = reading.model_dump()

    X_new = pd.DataFrame([data], columns=features)
    X_new["light"] = X_new["light"].astype(int)
    X_new["motion"] = X_new["motion"].astype(int)

    # ML anomaly detection
    prediction = int(model.predict(X_new)[0])
    score = float(-model.score_samples(X_new)[0])

    # Combine the results into one alert
    if prediction == -1:
        status = "anomaly"
        message = "Unusual sensor pattern detected"
    else:
        status = "normal"
        message = "No anomaly detected by the model"

    
    return {
        "prediction": prediction,
        "status": "anomaly" if prediction == -1 else "normal",
        "anomaly_score": round(score, 6),
        "reason": (
            "The sensor reading differs from patterns "
            "learned by the Isolation Forest model."
            if prediction == -1
            else "The model did not flag this reading as unusual."
        ),
        "message": (
            "Unusual sensor pattern detected"
            if prediction == -1
            else "No anomaly detected by the model"
        )
    }


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
