
import pandas as pd

df = pd.read_csv("outputs/skyguard_validation_results.csv")

# Sort by time so adjacent readings can be compared.
df = df.sort_values("timestamp").reset_index(drop=True)

# Find possible temperature drops.
df["previous_temp"] = df["temp"].shift(1)
df["next_temp"] = df["temp"].shift(-1)

df["possible_drop"] = (
    (df["temp"] < 10)
    & (df["previous_temp"] > 15)
    & (df["next_temp"] > 15)
)

candidates = df[df["possible_drop"]].copy()

print("========== TEMPERATURE DROP CHECK ==========")
print(f"Candidate temperature drops: {len(candidates)}")

if len(candidates) > 0:
    flagged = (candidates["prediction"] == -1).sum()

    print(f"Flagged by Isolation Forest: {flagged}")
    print(
        f"Flagged percentage: "
        f"{flagged / len(candidates) * 100:.2f}%"
    )

    print("\nExample candidates:")
    print(
        candidates[
            [
                "previous_temp",
                "temp",
                "next_temp",
                "prediction",
                "anomaly_score",
            ]
        ].head(10).to_string(index=False)
    )
else:
    print(
        "No candidates found. This may mean the test period "
        "contains no readings matching this rule."
    )

print(
    "\nImportant: these are rule-defined candidates, "
    "not confirmed real-world faults."
)
