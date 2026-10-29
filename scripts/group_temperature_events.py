
import pandas as pd

df = pd.read_csv("outputs/skyguard_temperature_drop_candidates.csv")

# Convert Unix timestamps to readable UTC times.
df["datetime"] = pd.to_datetime(
    df["ts"],
    unit="s",
    utc=True,
)

# Sort candidates by device and time.
df = df.sort_values(
    ["device", "ts"]
).reset_index(drop=True)

# Calculate the time since the previous candidate
# for the same device.
df["minutes_since_previous_candidate"] = (
    df.groupby("device")["datetime"]
    .diff()
    .dt.total_seconds()
    / 60
)

# Start a new event if more than 30 minutes have
# passed since the previous candidate.
df["new_event"] = (
    df["minutes_since_previous_candidate"].isna()
    | (df["minutes_since_previous_candidate"] > 30)
)

df["event_number"] = (
    df.groupby("device")["new_event"]
    .cumsum()
    .astype(int)
)

print("========== CANDIDATE EVENT SUMMARY ==========")

events = (
    df.groupby(["device", "event_number"])
    .agg(
        readings=("temp", "size"),
        start_time=("datetime", "min"),
        end_time=("datetime", "max"),
        lowest_temperature=("temp", "min"),
        highest_temperature=("temp", "max"),
    )
    .reset_index()
)

print(events.to_string(index=False))

print("\nTotal candidate readings:", len(df))
print("Candidate event groups:", len(events))

events.to_csv(
    "skyguard_temperature_events.csv",
    index=False,
)

print("\nSaved event groups to skyguard_temperature_events.csv")
print(
    "Note: the 30-minute grouping rule is exploratory, "
    "not a verified definition of a real incident."
)
