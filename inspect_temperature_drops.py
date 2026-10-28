
import pandas as pd

df = pd.read_csv("skyguard_temperature_drop_candidates.csv")

print("========== DEVICE SUMMARY ==========")

print(
    df.groupby("device").agg(
        candidate_readings=("temp", "size"),
        unique_temperatures=("temp", "nunique"),
        unique_timestamps=("ts", "nunique"),
        flagged_readings=(
            "prediction",
            lambda values: (values == -1).sum()
        ),
    ).to_string()
)

print("\n========== MOST COMMON CANDIDATE TEMPERATURES ==========")

print(
    df["temp"]
    .round(2)
    .value_counts()
    .head(10)
    .to_string()
)

print("\n========== REPEATED CANDIDATE TIMESTAMPS ==========")

print(
    df["ts"]
    .value_counts()
    .head(10)
    .to_string()
)

print("\n========== CANDIDATE EXAMPLES ==========")

columns = [
    "device",
    "ts",
    "previous_temp",
    "temp",
    "next_temp",
    "prediction",
    "anomaly_score",
]

print(df[columns].head(15).to_string(index=False))
