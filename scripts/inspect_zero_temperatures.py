
import pandas as pd

df = pd.read_csv("iot_telemetry_data.csv")

df = df.sort_values(["device", "ts"]).copy()

# Find readings with temperature equal to zero.
zero = df[df["temp"] == 0].copy()

print("========== ZERO-TEMPERATURE SUMMARY ==========")
print("Total zero-temperature readings:", len(zero))

print("\nBy device:")
print(zero.groupby("device").size().to_string())

print("\n========== SENSOR VALUES DURING ZERO TEMPERATURE ==========")

columns = [
    "device", "ts", "temp", "humidity",
    "co", "lpg", "smoke", "light", "motion"
]

print(zero[columns].head(15).to_string(index=False))

print("\n========== NEIGHBORING TEMPERATURES ==========")

df["previous_temp"] = (
    df.groupby("device")["temp"].shift(1)
)
df["next_temp"] = (
    df.groupby("device")["temp"].shift(-1)
)

zero_with_neighbors = df[df["temp"] == 0][
    ["device", "ts", "previous_temp", "temp", "next_temp"]
]

print(zero_with_neighbors.head(15).to_string(index=False))
