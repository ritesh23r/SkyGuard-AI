
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# ==================================================
# 1. LOAD AND PREPARE THE DATASET
# ==================================================

print("Loading sensor dataset...")

df = pd.read_csv("iot_telemetry_data.csv")

features = [
    "co",
    "humidity",
    "light",
    "lpg",
    "motion",
    "smoke",
    "temp",
]

# Keep the device ID and timestamp for validation.
required_columns = ["ts", "device"] + features
data = df[required_columns].copy()

# Remove rows with missing required values.
data = data.dropna(subset=required_columns)

# Give every original reading a unique ID.
data = data.reset_index(drop=True)
data["reading_id"] = data.index

# Convert sensor columns to the same types used by the model.
data["light"] = data["light"].astype(int)
data["motion"] = data["motion"].astype(int)

# Sort chronologically for the train/test split.
data = data.sort_values("ts").reset_index(drop=True)

print(f"Total valid readings: {len(data):,}")

# ==================================================
# 2. SPLIT INTO TRAINING AND TEST DATA
# ==================================================

split_point = int(len(data) * 0.80)

train_data = data.iloc[:split_point].copy()
test_data = data.iloc[split_point:].copy()

X_train = train_data[features].copy()
X_test = test_data[features].copy()

print(f"Training readings: {len(X_train):,}")
print(f"Unseen test readings: {len(X_test):,}")

# ==================================================
# 3. TRAIN A TEMPORARY ISOLATION FOREST
# ==================================================

print("\nTraining a temporary Isolation Forest...")

test_model = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42,
    n_jobs=-1,
)

test_model.fit(X_train)

print("Training completed.")

# ==================================================
# 4. PREDICT ON UNSEEN TEST DATA
# ==================================================

predictions = test_model.predict(X_test)

# Isolation Forest:
#  1 = normal
# -1 = anomaly

scores = -test_model.score_samples(X_test)

anomaly_count = int((predictions == -1).sum())
normal_count = int((predictions == 1).sum())

anomaly_percentage = anomaly_count / len(X_test) * 100

print("\n========== VALIDATION RESULTS ==========")
print(f"Test readings: {len(X_test):,}")
print(f"Normal readings: {normal_count:,}")
print(f"Flagged anomalies: {anomaly_count:,}")
print(f"Anomaly percentage: {anomaly_percentage:.2f}%")
print(f"Mean anomaly score: {np.mean(scores):.6f}")
print(f"Median anomaly score: {np.median(scores):.6f}")

# ==================================================
# 5. SAVE VALIDATION RESULTS WITH DEVICE AND TIME
# ==================================================

results = test_data[
    ["reading_id", "ts", "device"] + features
].copy()

results = results.reset_index(drop=True)
results["prediction"] = predictions
results["anomaly_score"] = scores

results = results.rename(columns={"ts": "timestamp"})

# Show the 10 most unusual readings.
most_unusual = results.sort_values(
    "anomaly_score",
    ascending=False,
).head(10)

print("\n========== TOP 10 UNUSUAL READINGS ==========")
print(most_unusual.to_string(index=False))

results.to_csv(
    "skyguard_validation_results.csv",
    index=False,
)

print("\nSaved results to skyguard_validation_results.csv")

# ==================================================
# 6. FIND POSSIBLE TEMPERATURE DROPS PER DEVICE
# ==================================================

print("\n========== TEMPERATURE DROP CHECK ==========")

# Sort each device's readings by time.
sequence_data = data.sort_values(
    ["device", "ts"]
).copy()

# Compare each reading with the previous and next
# reading from the SAME device.
sequence_data["previous_temp"] = (
    sequence_data.groupby("device")["temp"].shift(1)
)

sequence_data["next_temp"] = (
    sequence_data.groupby("device")["temp"].shift(-1)
)

# Our exploratory temperature-drop rule:
# current temperature < 10
# previous temperature > 15
# next temperature > 15

sequence_data["possible_drop"] = (
    (sequence_data["temp"] < 10)
    & (sequence_data["previous_temp"] > 15)
    & (sequence_data["next_temp"] > 15)
)

# Only evaluate candidates whose current reading
# belongs to the unseen test set.
test_ids = set(test_data["reading_id"])

candidates = sequence_data[
    sequence_data["possible_drop"]
    & sequence_data["reading_id"].isin(test_ids)
].copy()

# Join the model's prediction and score for each candidate.
candidates = candidates.merge(
    results[
        ["reading_id", "prediction", "anomaly_score"]
    ],
    on="reading_id",
    how="left",
)

candidate_count = len(candidates)

print(f"Test-set temperature-drop candidates: {candidate_count}")

if candidate_count > 0:
    flagged_count = int((candidates["prediction"] == -1).sum())
    flagged_percentage = flagged_count / candidate_count * 100

    print(f"Flagged by Isolation Forest: {flagged_count}")
    print(f"Not flagged: {candidate_count - flagged_count}")
    print(f"Flagged percentage: {flagged_percentage:.2f}%")

    print("\nExample candidates:")
    print(
        candidates[
            [
                "device",
                "ts",
                "previous_temp",
                "temp",
                "next_temp",
                "prediction",
                "anomaly_score",
            ]
        ].head(10).to_string(index=False)
    )

    candidates.to_csv(
        "skyguard_temperature_drop_candidates.csv",
        index=False,
    )

    print(
        "\nSaved candidate details to "
        "skyguard_temperature_drop_candidates.csv"
    )

else:
    print(
        "No temperature-drop candidates were found "
        "in the test period."
    )

print("\nValidation finished.")
print("Your existing saved model was not changed.")
print(
    "Note: temperature-drop candidates are defined by a rule; "
    "they are not confirmed real-world faults."
)

