
import json
import os
import time
from collections import deque
from datetime import datetime

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SkyGuard AI",
    page_icon="🌍",
    layout="wide",
)

API_URL = "http://127.0.0.1:8000/predict"
HISTORY_FILE = "sensor_history.jsonl"
MAX_HISTORY_RECORDS = 1000
CHART_RECORDS = 100

# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------

st.title("SkyGuard AI")
st.caption("AI-powered environmental sensor anomaly monitoring")

st.markdown(
    """
    SkyGuard AI analyzes environmental sensor readings and flags
    unusual patterns using an Isolation Forest machine learning model.

    **Note:** Automatic readings are simulated using historical
    dataset records, not collected from physical sensors.
    """
)

# Keep manual testing history between Streamlit reruns.
if "history" not in st.session_state:
    st.session_state.history = []


# --------------------------------------------------
# LOAD SENSOR HISTORY WITHOUT USING TOO MUCH MEMORY
# --------------------------------------------------

def load_sensor_history():
    """Load only the latest sensor records from the JSONL file."""

    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame()

    records = deque(maxlen=MAX_HISTORY_RECORDS)

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    record = json.loads(line)
                    records.append(record)
                except (json.JSONDecodeError, TypeError):
                    continue
    except OSError as error:
        st.warning(f"Could not read sensor history: {error}")
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(list(records))

    # Convert columns to numeric values for charts and metrics.
    numeric_columns = [
        "Temperature",
        "Humidity",
        "CO",
        "LPG",
        "Smoke",
        "Score",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(
            df["Time"], errors="coerce"
        )
        df = df.dropna(subset=["Time"])
        df = df.sort_values("Time").reset_index(drop=True)

    return df


# --------------------------------------------------
# AUTOMATIC MONITORING SETTINGS
# --------------------------------------------------

st.header("Automatic Sensor Monitoring")

auto_refresh = st.checkbox(
    "Refresh dashboard automatically every 5 seconds",
    value=True,
    key="auto_refresh",
)

sensor_df = load_sensor_history()


# --------------------------------------------------
# ALERT PANEL
# --------------------------------------------------

st.subheader("Alert Panel")

if sensor_df.empty:
    st.info(
        "No sensor readings found yet. Start the FastAPI server "
        "and run sensor_simulator.py in separate terminals."
    )

else:
    latest = sensor_df.iloc[-1]
    latest_status = str(latest.get("Status", "unknown")).lower()

    if latest_status == "anomaly":
        st.error(
            "ANOMALY ALERT: The latest sensor reading has been "
            "flagged as unusual by the model."
        )
    elif latest_status == "normal":
        st.success(
            "SYSTEM STATUS: The latest reading was not flagged "
            "as anomalous by the model."
        )
    else:
        st.warning("The latest reading has an unknown status.")

    # Show recent anomaly events.
    if "Status" in sensor_df.columns:
        anomaly_df = sensor_df[
            sensor_df["Status"].astype(str).str.lower() == "anomaly"
        ].copy()
    else:
        anomaly_df = pd.DataFrame()

    if not anomaly_df.empty:
        st.markdown("**Recent anomaly events**")
        display_columns = [
            column
            for column in [
                "Time",
                "Temperature",
                "Humidity",
                "CO",
                "LPG",
                "Smoke",
                "Score",
                "Reason",
            ]
            if column in anomaly_df.columns
        ]

        st.dataframe(
            anomaly_df[display_columns].tail(10).iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No anomaly events are present in the loaded history.")


# --------------------------------------------------
# LIVE SUMMARY METRICS
# --------------------------------------------------

st.divider()
st.subheader("Live Sensor Summary")

if not sensor_df.empty:
    latest = sensor_df.iloc[-1]

    total_readings = len(sensor_df)

    if "Status" in sensor_df.columns:
        anomaly_count = int(
            (
                sensor_df["Status"].astype(str).str.lower()
                == "anomaly"
            ).sum()
        )
    else:
        anomaly_count = 0

    col1, col2, col3, col4 = st.columns(4)

    temperature = latest.get("Temperature", None)
    humidity = latest.get("Humidity", None)

    col1.metric(
        "Latest Temperature",
        f"{temperature:.2f} °C"
        if pd.notna(temperature)
        else "N/A",
    )

    col2.metric(
        "Latest Humidity",
        f"{humidity:.2f}%"
        if pd.notna(humidity)
        else "N/A",
    )

    col3.metric(
        "Loaded Readings",
        f"{total_readings:,}",
    )

    col4.metric(
        "Anomaly Readings",
        f"{anomaly_count:,}",
    )

    st.caption(
        f"Summary uses the latest {total_readings:,} available "
        "records loaded by the dashboard, not necessarily the "
        "entire history file."
    )


# --------------------------------------------------
# LATEST AUTOMATIC READINGS
# --------------------------------------------------

st.divider()
st.subheader("Latest Automatic Sensor Readings")

if not sensor_df.empty:
    st.dataframe(
        sensor_df.tail(20).iloc[::-1],
        use_container_width=True,
        hide_index=True,
    )

    csv_data = sensor_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Automatic Sensor History (CSV)",
        data=csv_data,
        file_name="skyguard_sensor_history.csv",
        mime="text/csv",
    )


# --------------------------------------------------
# SENSOR TREND ANALYSIS - PLOTLY ONLY
# --------------------------------------------------

st.divider()
st.header("Sensor Trend Analysis")

if not sensor_df.empty:
    chart_df = sensor_df.tail(CHART_RECORDS).copy()
    chart_df = chart_df.reset_index(drop=True)
    chart_df["Reading Number"] = range(1, len(chart_df) + 1)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        if "Temperature" in chart_df.columns:
            fig_temp = px.line(
                chart_df,
                x="Reading Number",
                y="Temperature",
                title="Temperature Trend",
                markers=True,
            )
            st.plotly_chart(
                fig_temp,
                use_container_width=True,
            )

    with chart_col2:
        if "Humidity" in chart_df.columns:
            fig_humidity = px.line(
                chart_df,
                x="Reading Number",
                y="Humidity",
                title="Humidity Trend",
                markers=True,
            )
            st.plotly_chart(
                fig_humidity,
                use_container_width=True,
            )

    if "Score" in chart_df.columns:
        fig_score = px.line(
            chart_df,
            x="Reading Number",
            y="Score",
            title="Anomaly Score Trend",
            markers=True,
        )
        st.plotly_chart(
            fig_score,
            use_container_width=True,
        )

    st.caption(
        f"Charts display at most the latest {CHART_RECORDS} readings. "
        "A higher anomaly score indicates a more unusual reading; "
        "the score is not a probability."
    )
else:
    st.info("Sensor trend charts will appear after readings arrive.")


# --------------------------------------------------
# MANUAL SENSOR TESTING
# --------------------------------------------------

st.divider()
st.header("Manual Sensor Testing")

st.caption(
    "Enter a sensor reading to test it against the trained model. "
    "This test sends the values to your local FastAPI server."
)

with st.form("manual_sensor_form"):
    col1, col2 = st.columns(2)

    with col1:
        temp = st.number_input(
            "Temperature (°C)",
            min_value=-50.0,
            max_value=100.0,
            value=25.0,
            step=0.1,
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=0.1,
        )

        co = st.number_input(
            "CO",
            min_value=0.0,
            value=0.01318,
            format="%.6f",
        )

        lpg = st.number_input(
            "LPG",
            min_value=0.0,
            value=0.015524,
            format="%.6f",
        )

    with col2:
        smoke = st.number_input(
            "Smoke",
            min_value=0.0,
            value=0.043461,
            format="%.6f",
        )

        light = st.toggle(
            "Light detected",
            value=False,
        )

        motion = st.toggle(
            "Motion detected",
            value=False,
        )

    submitted = st.form_submit_button(
        "Analyze Reading",
        use_container_width=True,
    )


# --------------------------------------------------
# DISPLAY MANUAL PREDICTION
# --------------------------------------------------

if submitted:
    payload = {
        "co": co,
        "humidity": humidity,
        "light": light,
        "lpg": lpg,
        "motion": motion,
        "smoke": smoke,
        "temp": temp,
    }

    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()

        status = result["status"]
        score = float(result["anomaly_score"])
        reason = result.get(
                    "reason",
                    result.get("message", "No explanation returned.")
                )

        record = {
            "Time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "Temperature": temp,
            "Humidity": humidity,
            "CO": co,
            "LPG": lpg,
            "Smoke": smoke,
            "Light": light,
            "Motion": motion,
            "Status": status,
            "Score": score,
            "Reason": reason,
        }

        st.session_state.history.append(record)

        st.subheader("Prediction Result")

        if status == "anomaly":
            st.error("Anomaly detected")
        else:
            st.success(
                "No anomaly detected by the model"
            )

        result_col1, result_col2 = st.columns(2)

        result_col1.metric(
            "Prediction",
            status.upper(),
        )

        result_col2.metric(
            "Anomaly Score",
            f"{score:.6f}",
        )

        st.write("**Model message:**", reason)

        st.caption(
            "This is an anomaly-detection result, not confirmation "
            "that a real sensor fault or environmental danger exists."
        )

    except requests.RequestException as error:
        st.error(
            "Could not connect to the API. Make sure FastAPI is "
            "running in another terminal."
        )
        st.code(str(error))

    except (ValueError, KeyError) as error:
        st.error(
            "The API returned an unexpected response. Check app.py "
            "and the FastAPI terminal."
        )
        st.code(str(error))


# --------------------------------------------------
# MANUAL TEST HISTORY
# --------------------------------------------------

st.divider()
st.header("Manual Test History")

if st.session_state.history:
    manual_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        manual_df.iloc[::-1],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="Download Manual Test History (CSV)",
        data=manual_df.to_csv(index=False).encode("utf-8"),
        file_name="skyguard_manual_history.csv",
        mime="text/csv",
    )

    manual_chart_df = manual_df.tail(CHART_RECORDS).copy()
    manual_chart_df["Test Number"] = range(
        1, len(manual_chart_df) + 1
    )

    manual_col1, manual_col2 = st.columns(2)

    with manual_col1:
        fig_manual_temp = px.line(
            manual_chart_df,
            x="Test Number",
            y="Temperature",
            title="Manual Test: Temperature",
            markers=True,
        )
        st.plotly_chart(
            fig_manual_temp,
            use_container_width=True,
        )

    with manual_col2:
        fig_manual_humidity = px.line(
            manual_chart_df,
            x="Test Number",
            y="Humidity",
            title="Manual Test: Humidity",
            markers=True,
        )
        st.plotly_chart(
            fig_manual_humidity,
            use_container_width=True,
        )

    fig_manual_score = px.line(
        manual_chart_df,
        x="Test Number",
        y="Score",
        title="Manual Test: Anomaly Score",
        markers=True,
    )
    st.plotly_chart(
        fig_manual_score,
        use_container_width=True,
    )

else:
    st.info("No manual tests performed in this session.")


# --------------------------------------------------
# AUTOMATIC REFRESH
# --------------------------------------------------

if auto_refresh:
    time.sleep(5)
    st.rerun()
