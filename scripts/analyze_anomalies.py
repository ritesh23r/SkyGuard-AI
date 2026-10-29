
import pandas as pd

# Load the saved validation results
df = pd.read_csv("skyguard_validation_results.csv")

# Keep only readings flagged as anomalies
anomalies = df[df["prediction"] == -1].copy()

print("========== ANOMALY SUMMARY ==========")
print(f"Total test readings: {len(df):,}")
print(f"Flagged anomalies: {len(anomalies):,}")
print(
    f"Anomaly percentage: "
    f"{len(anomalies) / len(df) * 100:.2f}%"
)

# Compare sensor values for normal vs anomalous readings
print("\n========== SENSOR VALUE COMPARISON ==========")

sensor_columns = [
    "co",
    "humidity",
    "lpg",
    "smoke",
    "temp",
]

normal = df[df["prediction"] == 1]

comparison = pd.DataFrame({
    "Normal median": normal[sensor_columns].median(),
    "Anomaly median": anomalies[sensor_columns].median(),
})

comparison["Difference"] = (
    comparison["Anomaly median"] - comparison["Normal median"]
)

print(comparison.round(6))

# See how anomaly scores are distributed
print("\n========== ANOMALY SCORE SUMMARY ==========")

print(
    anomalies["anomaly_score"].describe().round(6)
)

# Save flagged readings for manual investigation
anomalies.sort_values(
    "anomaly_score",
    ascending=False,
).to_csv(
    "skyguard_flagged_anomalies.csv",
    index=False,
)

print("\nSaved flagged readings to skyguard_flagged_anomalies.csv")
