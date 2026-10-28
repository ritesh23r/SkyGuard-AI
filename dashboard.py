
import json
import os
import time

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# ---------------- CONFIGURATION ----------------

API_URL = "http://127.0.0.1:8000/predict"
HISTORY_FILE = "sensor_history.jsonl"

MAX_HISTORY_RECORDS = 1000
CHART_RECORDS = 100

st.set_page_config(
    page_title="SkyGuard AI",
    page_icon="🛡️",
    layout="wide",
)


# ---------------- DARK COMMAND CENTER THEME ----------------

st.markdown(
    """
    <style>
        .stApp {
            background-color: #0B1220;
            color: #E5EDF7;
        }

        [data-testid="stSidebar"] {
            background-color: #101B2D;
            border-right: 1px solid #243247;
        }

        [data-testid="stMetric"] {
            background-color: #111E30;
            border: 1px solid #243247;
            padding: 18px;
            border-radius: 12px;
        }

        [data-testid="stMetricLabel"] {
            color: #94A3B8;
        }

        [data-testid="stMetricValue"] {
            color: #F1F5F9;
        }

        h1, h2, h3 {
            color: #F1F5F9;
        }

        p, label {
            color: #CBD5E1;
        }

        div.stButton > button,
        div.stFormSubmitButton > button {
            background-color: #123B46;
            color: #67E8F9;
            border: 1px solid #236477;
            border-radius: 8px;
        }

        div.stButton > button:hover,
        div.stFormSubmitButton > button:hover {
            background-color: #185566;
            color: #FFFFFF;
            border-color: #67E8F9;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid #243247;
            border-radius: 10px;
        }

        hr {
            border-color: #243247;
        }
    </style>
    """,
    unsafe_allow_html=True,
)



# ---------------- PAGE HEADER ----------------

st.title("SkyGuard AI")
st.caption(
    "IoT Sensor Monitoring and Anomaly Detection System"
)

st.markdown(
    """
    Monitor environmental sensor readings, investigate unusual
    patterns, and test new sensor readings using an Isolation Forest model.
    """
)


# ---------------- LOAD HISTORY ----------------

@st.cache_data(ttl=2)
def load_sensor_history():
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame()

    records = []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            lines = file.readlines()[-MAX_HISTORY_RECORDS:]

        for line in lines:
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                continue

    except OSError:
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    sensor_df = pd.DataFrame(records)

    numeric_columns = [
        "Temperature",
        "Humidity",
        "CO",
        "LPG",
        "Smoke",
        "Score",
    ]

    for column in numeric_columns:
        if column in sensor_df.columns:
            sensor_df[column] = pd.to_numeric(
                sensor_df[column],
                errors="coerce",
            )

    if "Time" in sensor_df.columns:
        sensor_df["Time"] = pd.to_datetime(
            sensor_df["Time"],
            errors="coerce",
        )

    if "Status" in sensor_df.columns:
        sensor_df["Status"] = (
            sensor_df["Status"].astype(str).str.lower()
        )

    sensor_df = sensor_df.dropna(subset=["Time"])
    sensor_df = sensor_df.sort_values("Time").reset_index(drop=True)

    return sensor_df


sensor_df = load_sensor_history()


# ---------------- SIDEBAR ----------------

st.sidebar.header("Dashboard Controls")

auto_refresh = st.sidebar.checkbox(
    "Auto-refresh every 5 seconds",
    value=False,
)

if st.sidebar.button("Refresh dashboard"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption(
    "The simulator must be running to receive new readings."
)


# ---------------- SYSTEM STATUS ----------------

st.subheader("System Status")

if sensor_df.empty:
    st.info(
        "No sensor history found yet. Start the FastAPI server "
        "and sensor simulator, then wait for the first reading."
    )
else:
    latest = sensor_df.iloc[-1]
    latest_status = latest.get("Status", "unknown")

    if latest_status == "anomaly":
        st.error(
            "ANOMALY ALERT: The latest sensor reading was flagged "
            "as unusual by the model."
        )
    elif latest_status == "normal":
        st.success(
            "SYSTEM STATUS: The latest reading was not flagged "
            "as anomalous by the model."
        )
    else:
        st.warning("The latest reading has an unknown status.")

    # ---------------- LATEST READING METRICS ----------------

    st.subheader("Latest Sensor Reading")

    col1, col2, col3, col4 = st.columns(4)

    temperature = latest.get("Temperature")
    humidity = latest.get("Humidity")
    co_value = latest.get("CO")
    score = latest.get("Score")

    col1.metric(
        "Temperature",
        f"{temperature:.2f} °C"
        if pd.notna(temperature)
        else "N/A",
    )

    col2.metric(
        "Humidity",
        f"{humidity:.2f}%"
        if pd.notna(humidity)
        else "N/A",
    )

    col3.metric(
        "CO",
        f"{co_value:.4f}"
        if pd.notna(co_value)
        else "N/A",
    )

    col4.metric(
        "Anomaly Score",
        f"{score:.6f}"
        if pd.notna(score)
        else "N/A",
    )

    st.caption(
        "A higher anomaly score indicates a more unusual reading "
        "relative to the model's learned patterns. It is not a "
        "probability of failure."
    )

    # ---------------- SENSOR INVESTIGATION ----------------

    st.subheader("Sensor-Level Investigation")

    sensor_columns = [
        ("Temperature", "Temperature", "°C"),
        ("Humidity", "Humidity", "%"),
        ("CO", "CO", ""),
        ("LPG", "LPG", ""),
        ("Smoke", "Smoke", ""),
        ("Light", "Light", ""),
        ("Motion", "Motion", ""),
    ]

    sensor_cols = st.columns(4)

    for index, (column, label, unit) in enumerate(sensor_columns):
        with sensor_cols[index % 4]:
            value = latest.get(column)

            if pd.isna(value) if value is not None else True:
                display_value = "Not recorded"
            elif column in ("Light", "Motion"):
                display_value = str(value)
            elif unit:
                display_value = f"{float(value):.2f} {unit}"
            else:
                display_value = f"{float(value):.5f}"

            st.metric(label, display_value)

    st.caption(
        "These are the sensor values recorded for the latest reading. "
        "Light and Motion appear only in records created after those "
        "fields were added to the simulator."
    )

    # ---------------- LATEST ANOMALY DETAILS ----------------

    if latest_status == "anomaly":
        st.subheader("Latest Anomaly Details")

        detail_col1, detail_col2 = st.columns(2)

        timestamp = latest.get("Time")
        reason = latest.get(
            "Reason",
            "No explanation was saved for this reading.",
        )

        detail_col1.write("**Reading time**")

        if pd.notna(timestamp):
            detail_col1.write(timestamp.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            detail_col1.write("Not available")

        detail_col2.write("**Why was this flagged?**")
        detail_col2.write(reason)

    # ---------------- ANOMALY HISTORY ----------------

    st.subheader("Recent Anomaly Events")

    anomaly_df = sensor_df[
        sensor_df["Status"] == "anomaly"
    ].copy()

    if anomaly_df.empty:
        st.success("No anomalies have been recorded in the loaded history.")
    else:
        display_columns = [
            "Time",
            "Temperature",
            "Humidity",
            "CO",
            "LPG",
            "Smoke",
            "Light",
            "Motion",
            "Score",
            "Reason",
        ]

        available_columns = [
            column
            for column in display_columns
            if column in anomaly_df.columns
        ]

        st.dataframe(
            anomaly_df[available_columns].tail(20).iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )

    # ---------------- SENSOR CHARTS ----------------

    st.subheader("Sensor Trends")

    chart_df = sensor_df.tail(CHART_RECORDS).copy()

    chart_options = [
        column
        for column in [
            "Temperature",
            "Humidity",
            "CO",
            "LPG",
            "Smoke",
            "Score",
        ]
        if column in chart_df.columns
    ]

    if chart_options:
        selected_sensors = st.multiselect(
            "Select measurements to display",
            options=chart_options,
            default=[
                column
                for column in ["Temperature", "Humidity"]
                if column in chart_options
            ],
        )

        if selected_sensors:
            chart_long = chart_df[
                ["Time"] + selected_sensors
            ].melt(
                id_vars="Time",
                var_name="Sensor",
                value_name="Value",
            )

            figure = px.line(
                chart_long,
                x="Time",
                y="Value",
                color="Sensor",
                title="Recent Sensor Measurements",
                markers=True,
            )

            figure.update_layout(
                xaxis_title="Reading Time",
                yaxis_title="Sensor Value",
                legend_title="Measurement",
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )
        else:
            st.info("Select at least one measurement to display.")

    # ---------------- ANOMALY DISTRIBUTION ----------------

    st.subheader("Anomaly Overview")

    total_readings = len(sensor_df)
    total_anomalies = int(
        (sensor_df["Status"] == "anomaly").sum()
    )
    total_normal = int(
        (sensor_df["Status"] == "normal").sum()
    )

    anomaly_percentage = (
        total_anomalies / total_readings * 100
        if total_readings
        else 0
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric("Readings Loaded", total_readings)
    metric2.metric("Normal Readings", total_normal)
    metric3.metric("Anomalies", total_anomalies)
    metric4.metric("Anomaly Rate", f"{anomaly_percentage:.2f}%")

    distribution_df = pd.DataFrame(
        {
            "Status": ["Normal", "Anomaly"],
            "Count": [total_normal, total_anomalies],
        }
    )

    distribution_fig = px.bar(
        distribution_df,
        x="Status",
        y="Count",
        color="Status",
        title="Normal vs Anomalous Readings",
        text="Count",
    )

    st.plotly_chart(
        distribution_fig,
        use_container_width=True,
    )

    # ---------------- DOWNLOAD HISTORY ----------------

    st.subheader("Export Data")

    download_df = sensor_df.copy()

    if "Time" in download_df.columns:
        download_df["Time"] = download_df["Time"].astype(str)

    st.download_button(
        label="Download Sensor History as CSV",
        data=download_df.to_csv(index=False).encode("utf-8"),
        file_name="skyguard_sensor_history.csv",
        mime="text/csv",
    )


# ---------------- MANUAL SENSOR TEST ----------------

st.divider()
st.header("Manual Sensor Test")

st.write(
    "Enter sensor values to check whether the model flags the "
    "reading as unusual. This test does not add the reading to "
    "the simulator's history file."
)

with st.form("manual_sensor_test"):
    form_col1, form_col2, form_col3 = st.columns(3)

    with form_col1:
        co = st.number_input(
            "CO",
            min_value=0.0,
            value=0.004,
            format="%.6f",
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
        )

        light = st.selectbox(
            "Light detected?",
            options=[False, True],
            format_func=lambda value: "Yes" if value else "No",
        )

    with form_col2:
        lpg = st.number_input(
            "LPG",
            min_value=0.0,
            value=0.007,
            format="%.6f",
        )

        motion = st.selectbox(
            "Motion detected?",
            options=[False, True],
            format_func=lambda value: "Yes" if value else "No",
        )

    with form_col3:
        smoke = st.number_input(
            "Smoke",
            min_value=0.0,
            value=0.02,
            format="%.6f",
        )

        temp = st.number_input(
            "Temperature (°C)",
            value=22.0,
        )

    submitted = st.form_submit_button(
        "Analyze Sensor Reading",
        use_container_width=True,
    )

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
        with st.spinner("Analyzing sensor reading..."):
            response = requests.post(
                API_URL,
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            result = response.json()

        status = result.get("status", "unknown")
        score = result.get("anomaly_score")
        reason = result.get(
            "reason",
            result.get("message", "No explanation returned."),
        )

        st.subheader("Prediction Result")

        if status == "anomaly":
            st.error("Anomaly detected: unusual sensor pattern.")
        elif status == "normal":
            st.success("No anomaly detected by the model.")
        else:
            st.warning(f"Unexpected status: {status}")

        result_col1, result_col2 = st.columns(2)

        result_col1.metric("Prediction Status", status.title())

        if score is not None:
            result_col2.metric(
                "Anomaly Score",
                f"{float(score):.6f}",
            )

        st.write("**Model explanation:**", reason)

        with st.expander("View submitted sensor values"):
            st.json(payload)

    except requests.RequestException as error:
        st.error(
            "Could not connect to the SkyGuard AI API. "
            "Make sure FastAPI is running at http://127.0.0.1:8000."
        )
        st.caption(str(error))


# ---------------- AUTO REFRESH ----------------

if auto_refresh:
    time.sleep(5)
    st.rerun()
