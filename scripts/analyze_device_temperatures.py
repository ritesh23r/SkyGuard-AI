
import pandas as pd

features = ["temp"]

# Load the original dataset.
df = pd.read_csv("data/iot_telemetry_data.csv")

df = df[["device", "ts", "temp"]].dropna().copy()

# Load candidate readings.
candidates = pd.read_csv(
    "skyguard_temperature_drop_candidates.csv"
)

print("========== TEMPERATURE RANGE BY DEVICE ==========")

summary = df.groupby("device")["temp"].agg(
    readings="count",
    minimum="min",
    median="median",
    maximum="max",
    standard_deviation="std",
)

print(summary.round(2).to_string())

print("\n========== CANDIDATE TEMPERATURES BY DEVICE ==========")

candidate_summary = candidates.groupby("device")["temp"].agg(
    candidate_readings="count",
    minimum="min",
    median="median",
    maximum="max",
)

print(candidate_summary.round(2).to_string())

print("\n========== TEMPERATURE DISTRIBUTION ==========")

for device, group in df.groupby("device"):
    print(f"\nDevice: {device}")
    print(
        group["temp"]
        .quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        .round(2)
        .to_string()
    )
