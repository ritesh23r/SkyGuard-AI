import json
import os
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================
API_BASE_URL = "https://skyguard-ai-c1wd.onrender.com"
API_URL = f"{API_BASE_URL}/predict"
HISTORY_URL = f"{API_BASE_URL}/history"
HISTORY_FILE = "tests/sensor_history.jsonl"
EVENTS_FILE = "outputs/skyguard_temperature_events.csv"

MAX_HISTORY_RECORDS = 1000
CHART_RECORDS = 150

# Streamlit Community Cloud reads this setting from App Settings > Secrets.
# Local runs default to local JSONL history unless the environment variable is set.
try:
    remote_setting = st.secrets.get(
        "SKYGUARD_USE_REMOTE_HISTORY",
        os.getenv("SKYGUARD_USE_REMOTE_HISTORY", "false"),
    )
except Exception:
    remote_setting = os.getenv("SKYGUARD_USE_REMOTE_HISTORY", "false")

USE_REMOTE_HISTORY = str(remote_setting).lower() == "true"

st.set_page_config(
    page_title="SkyGuard AI | Sensor Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN SYSTEM — DARK COMMAND CENTER
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bg: #080E19;
        --panel: #101A2A;
        --panel-2: #131F32;
        --border: #26354B;
        --text: #ECF3FC;
        --muted: #91A2B9;
        --cyan: #56E0D0;
        --blue: #6EA8FE;
        --amber: #FFC76A;
        --red: #FF7885;
    }

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(31, 100, 128, .18), transparent 34%),
            radial-gradient(ellipse at 100% 15%, rgba(62, 76, 143, .12), transparent 30%),
            var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] { background: rgba(8, 14, 25, .88); }
    [data-testid="stSidebar"] {
        background: #0B1422;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }

    .block-container {
        max-width: 1500px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.035em;
        color: var(--text);
    }
    h1 { font-size: 2.2rem !important; }
    h2 { font-size: 1.5rem !important; }
    h3 { font-size: 1.1rem !important; }

    p, label, [data-testid="stMarkdownContainer"] { color: #D4DEEB; }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #142136, #0F1929);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px 20px;
        min-height: 118px;
        box-shadow: 0 8px 24px rgba(0,0,0,.12);
    }
    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-size: .76rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: .075em;
    }
    [data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricDelta"] { font-size: .78rem !important; }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--border) !important;
        border-radius: 16px !important;
        background: rgba(16, 26, 42, .44);
    }

    .sg-brand { display:flex; align-items:center; gap:13px; padding:4px 0 18px; }
    .sg-logo {
        width:48px; height:48px; border-radius:15px; display:flex;
        align-items:center; justify-content:center;
        background:linear-gradient(135deg,#174C58,#123044);
        border:1px solid #2D7884; color:#6FF2E1; font-size:25px;
        box-shadow:0 0 24px rgba(86,224,208,.12);
    }
    .sg-title {
        font-family:'Space Grotesk',sans-serif; color:#F2F7FF;
        font-size:1.35rem; line-height:1.2; font-weight:700; letter-spacing:-.04em;
    }
    .sg-subtitle { color:#8FA3BB; font-size:.66rem; letter-spacing:.12em; margin-top:5px; font-weight:700; }
    .sg-eyebrow { color:var(--cyan); text-transform:uppercase; letter-spacing:.15em; font-size:.68rem; font-weight:700; margin-bottom:7px; }
    .sg-page-title { font-family:'Space Grotesk',sans-serif; color:#F2F7FF; font-size:2.25rem; line-height:1.15; letter-spacing:-.055em; font-weight:700; margin-bottom:8px; }
    .sg-description { color:#9EADC1; font-size:.94rem; margin-bottom:0; line-height:1.6; }
    .sg-pill {
        display:inline-flex; align-items:center; gap:7px; border:1px solid #285B57;
        color:#8CF4D9; background:rgba(23,91,77,.22); border-radius:999px;
        padding:7px 11px; font-size:.7rem; font-weight:700; letter-spacing:.04em; white-space:nowrap;
    }
    .sg-dot { width:7px; height:7px; background:#56E0B5; border-radius:50%; box-shadow:0 0 10px rgba(86,224,181,.7); }
    .sg-panel-title { font-family:'Space Grotesk',sans-serif; color:#EDF4FC; font-size:1.04rem; font-weight:700; margin-bottom:3px; }
    .sg-panel-subtitle { color:#91A2B9; font-size:.8rem; margin-bottom:12px; line-height:1.45; }
    .sg-callout { border:1px solid #34415A; background:linear-gradient(110deg,rgba(19,31,50,.95),rgba(16,26,42,.95)); border-radius:14px; padding:17px; }
    .sg-callout-title { color:#EAF2FD; font-weight:700; margin-bottom:5px; }
    .sg-callout-copy { color:#9DADC2; font-size:.85rem; line-height:1.55; }
    .sg-muted { color:#91A2B9; font-size:.82rem; }
    .sg-footer { color:#71839B; font-size:.75rem; border-top:1px solid #26354B; margin-top:30px; padding-top:15px; }
    .sg-section-label { color:#71839B; font-size:.7rem; letter-spacing:.12em; font-weight:700; text-transform:uppercase; margin:10px 0 8px; }

    div.stButton > button, div.stFormSubmitButton > button,
    [data-testid="stDownloadButton"] button {
        border-radius:10px; border:1px solid #2C6974;
        background:linear-gradient(135deg,#174451,#123244);
        color:#A2FFF0; font-weight:700; min-height:2.6rem; transition:all .18s ease;
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover,
    [data-testid="stDownloadButton"] button:hover {
        border-color:#56E0D0; background:#1A5260; color:white;
    }

    [data-testid="stTabs"] [role="tablist"] { gap:6px; border-bottom:1px solid var(--border); }
    [data-testid="stTabs"] button[role="tab"] {
        border-radius:9px 9px 0 0; color:#9EADC1; padding:10px 13px; font-weight:600;
    }
    [data-testid="stTabs"] button[aria-selected="true"] { color:#7EF2E0; border-bottom-color:#56E0D0; }

    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border:1px solid var(--border); border-radius:12px; overflow:hidden;
    }
    input, textarea, [data-baseweb="select"] > div { background-color:#101B2B !important; }
    .stAlert { border-radius:12px; }
    hr { border-color:var(--border); margin:1.2rem 0; }
    footer { visibility:hidden; }

    @media (max-width: 800px) {
        .block-container { padding-left:1rem; padding-right:1rem; }
        .sg-page-title { font-size:1.7rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================
@st.cache_data(ttl=2)
def load_sensor_history():
    """Load recent records from the cloud API or local JSONL history."""
    records = []

    if USE_REMOTE_HISTORY:
        try:
            response = requests.get(HISTORY_URL, timeout=15)
            response.raise_for_status()
            api_records = response.json()

            for item in api_records:
                if not isinstance(item, dict):
                    continue
                reading = item.get("sensor_reading", {})
                if not isinstance(reading, dict):
                    reading = {}

                records.append({
                    "Time": item.get("timestamp"),
                    "Temperature": reading.get("temp"),
                    "Humidity": reading.get("humidity"),
                    "CO": reading.get("co"),
                    "LPG": reading.get("lpg"),
                    "Smoke": reading.get("smoke"),
                    "Light": reading.get("light"),
                    "Motion": reading.get("motion"),
                    "Status": item.get("status", "unknown"),
                    "Score": item.get("anomaly_score"),
                    "Reason": item.get("reason", item.get("message", "")),
                    "Device": "Render API",
                })
        except (requests.RequestException, ValueError) as error:
            st.warning(f"Could not load cloud history from the API: {error}")
            return pd.DataFrame()
    else:
        if not os.path.exists(HISTORY_FILE):
            return pd.DataFrame()

        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as file:
                lines = file.readlines()[-MAX_HISTORY_RECORDS:]
            for line in lines:
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        records.append(item)
                except json.JSONDecodeError:
                    continue
        except OSError:
            return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    numeric_columns = ["Temperature", "Humidity", "CO", "LPG", "Smoke", "Score"]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "Time" not in df.columns:
        df["Time"] = pd.NaT

    # Cloud timestamps include a UTC offset; local JSONL timestamps may not.
    # Normalize both safely so sorting and charts work consistently.
    parsed_time = pd.to_datetime(df["Time"], errors="coerce", utc=True)
    df["Time"] = parsed_time.dt.tz_convert(None)

    if "Status" not in df.columns:
        df["Status"] = "unknown"
    df["Status"] = df["Status"].astype(str).str.lower().str.strip()

    for column, default in {
        "Reason": "",
        "Device": "Not recorded",
        "Light": None,
        "Motion": None,
    }.items():
        if column not in df.columns:
            df[column] = default

    return (
        df.dropna(subset=["Time"])
        .sort_values("Time")
        .tail(MAX_HISTORY_RECORDS)
        .reset_index(drop=True)
    )

@st.cache_data(ttl=30)
def load_temperature_events():
    if not os.path.exists(EVENTS_FILE):
        return pd.DataFrame()
    try:
        return pd.read_csv(EVENTS_FILE)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def safe_number(value, digits=2, suffix=""):
    if value is None or pd.isna(value):
        return "—"
    try:
        return f"{float(value):,.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "—"


def style_plotly(fig, height=320):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#B8C6D9", size=12),
        title_font=dict(family="Space Grotesk, sans-serif", size=16, color="#EDF4FC"),
        margin=dict(l=8, r=8, t=28, b=8),
        height=height,
        legend=dict(
            bgcolor="rgba(0,0,0,0)", orientation="h",
            yanchor="bottom", y=1.02, xanchor="left", x=0,
        ),
        hoverlabel=dict(bgcolor="#152238", bordercolor="#354760",
                        font=dict(color="#F2F7FF")),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#26354B",
                     tickfont=dict(color="#91A2B9"))
    fig.update_yaxes(gridcolor="rgba(145,162,185,.12)", zeroline=False,
                     linecolor="#26354B", tickfont=dict(color="#91A2B9"))
    return fig


def section_heading(title, subtitle=None):
    st.markdown(f'<div class="sg-panel-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="sg-panel-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def display_sensor_value(row, column):
    value = row.get(column)
    if value is None or pd.isna(value):
        return "Not recorded"
    if column in ("Light", "Motion"):
        return str(value)
    unit = {"Temperature": " °C", "Humidity": "%"}.get(column, "")
    return safe_number(value, 2 if column in ("Temperature", "Humidity") else 5, unit)


def check_api():
    try:
        response = requests.get(API_BASE_URL, timeout=2)
        return response.ok
    except requests.RequestException:
        return False


# ============================================================
# SIDEBAR CONTROLS
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sg-brand">
            <div class="sg-logo">⛨</div>
            <div>
                <div class="sg-title">SkyGuard AI</div>
                <div class="sg-subtitle">SENSOR INTELLIGENCE</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("#### Workspace")
    st.caption("Environmental telemetry monitoring")

    auto_refresh = st.toggle("Auto-refresh dashboard", value=False)
    refresh_seconds = st.select_slider(
        "Refresh interval",
        options=[5, 10, 15, 30],
        value=5,
        format_func=lambda value: f"{value} seconds",
        disabled=not auto_refresh,
    )

    if st.button("↻  Refresh now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("#### Connections")
    api_reachable = check_api()
    if api_reachable:
        st.success("FastAPI · Connected")
    else:
        st.error("FastAPI · Unavailable")

    if USE_REMOTE_HISTORY:
        st.success("History source · Render API")
    elif os.path.exists(HISTORY_FILE):
        st.success("History source · Local JSONL found")
    else:
        st.warning("History source · Waiting for local JSONL")

    st.markdown("---")
    st.markdown("#### About this data")
    st.caption("This dashboard displays historical sensor telemetry replayed by the simulator.")
    st.caption("An anomaly means an unusual pattern; it does not confirm equipment failure.")
    st.markdown("---")
    st.caption("SKYGUARD AI  ·  HACKATHON BUILD")


# ============================================================
# LOAD AND SUMMARIZE DATA
# ============================================================
sensor_df = load_sensor_history()
events_df = load_temperature_events()
has_data = not sensor_df.empty
latest = sensor_df.iloc[-1] if has_data else None

total_readings = len(sensor_df)
total_anomalies = int((sensor_df["Status"] == "anomaly").sum()) if has_data else 0
total_normal = int((sensor_df["Status"] == "normal").sum()) if has_data else 0
anomaly_rate = total_anomalies / total_readings * 100 if total_readings else 0
device_count = (
    sensor_df["Device"].nunique()
    if has_data and "Device" in sensor_df.columns
    else 0
)


# ============================================================
# PAGE HEADER
# ============================================================
head_left, head_right = st.columns([3.5, 1.3], vertical_alignment="center")
with head_left:
    st.markdown('<div class="sg-eyebrow">INTELLIGENT MONITORING PLATFORM</div>', unsafe_allow_html=True)
    st.markdown('<div class="sg-page-title">System Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sg-description">Environmental telemetry, anomaly detection, and sensor investigation in one workspace.</div>',
        unsafe_allow_html=True,
    )
with head_right:
    if api_reachable and has_data:
        st.markdown('<div class="sg-pill"><span class="sg-dot"></span> API CONNECTED · REPLAY DATA</div>', unsafe_allow_html=True)
    elif api_reachable:
        st.markdown('<div class="sg-pill"><span class="sg-dot"></span> API CONNECTED</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="sg-pill" style="border-color:#75414B;color:#FF9AA4;background:rgba(117,65,75,.16)">● API DISCONNECTED</div>',
            unsafe_allow_html=True,
        )

meta_left, meta_right = st.columns([3, 1])
with meta_left:
    st.caption("WORKSPACE  /  SENSOR MONITORING  /  OVERVIEW")
with meta_right:
    st.caption(f"Dashboard refreshed · {datetime.now().strftime('%H:%M:%S')}")

if not api_reachable:
    st.warning("FastAPI is not responding. Start it with `uvicorn app:app --reload` to enable manual predictions. Existing history can still be reviewed.")

if not has_data:
    st.info("No sensor history is available yet. Start the simulator and ensure the dashboard is configured to read the same history source.")
else:
    latest_status = str(latest.get("Status", "unknown"))
    if latest_status == "anomaly":
        st.error("LATEST READING FLAGGED — The model marked the most recent recorded reading as unusual. Review the values and score before drawing conclusions.")
    elif latest_status == "normal":
        st.success("LATEST READING NORMAL — The model did not flag the latest recorded reading.")
    else:
        st.warning("The latest reading has an unknown status.")


# ============================================================
# KPI CARDS
# ============================================================
k1, k2, k3, k4 = st.columns(4)
k1.metric("Latest temperature", display_sensor_value(latest, "Temperature") if has_data else "—")
k2.metric("Anomalies flagged", f"{total_anomalies:,}", help="Count among the currently loaded history records, not model accuracy.")
k3.metric("Readings loaded", f"{total_readings:,}", help=f"Most recent records loaded from the selected history source, limited to {MAX_HISTORY_RECORDS}.")
k4.metric("Anomaly rate", f"{anomaly_rate:.1f}%", help="Share of loaded readings flagged by the model. This is not accuracy.")

st.markdown("")


# ============================================================
# MAIN WORKSPACE TABS
# ============================================================
tab_overview, tab_analytics, tab_anomalies, tab_events, tab_manual, tab_export = st.tabs(
    ["◈  Overview", "⌁  Sensor analytics", "⚑  Anomaly history",
     "◷  Temperature events", "＋  Manual test", "⇩  Data export"]
)


# ============================================================
# OVERVIEW
# ============================================================
with tab_overview:
    left, right = st.columns([1.65, 1], gap="large")

    with left:
        with st.container(border=True):
            section_heading("Sensor telemetry", "Recent measurements from the historical replay")
            if has_data:
                chart_df = sensor_df.tail(CHART_RECORDS).copy()
                trend_options = [c for c in ["Temperature", "Humidity", "CO", "LPG", "Smoke"] if c in chart_df.columns]
                default_options = [c for c in ["Temperature", "Humidity"] if c in trend_options]
                selected = st.multiselect("Measurements", trend_options, default=default_options, key="overview_measurements")
                if selected:
                    long_df = chart_df[["Time"] + selected].melt(id_vars="Time", var_name="Sensor", value_name="Value")
                    fig = px.line(long_df, x="Time", y="Value", color="Sensor")
                    fig.update_traces(line=dict(width=2.2))
                    style_plotly(fig, 330)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.info("Choose one or more measurements to plot.")
            else:
                st.markdown('<div class="sg-callout"><div class="sg-callout-title">Waiting for telemetry</div><div class="sg-callout-copy">When the API or local simulator records readings, sensor trends will appear here.</div></div>', unsafe_allow_html=True)

    with right:
        with st.container(border=True):
            section_heading("Latest reading", "Most recent record in the selected history source")
            if has_data:
                if latest_status == "anomaly":
                    st.warning("⚑  ANOMALY FLAGGED")
                elif latest_status == "normal":
                    st.success("●  NORMAL READING")
                else:
                    st.info("Status not recorded")

                a, b = st.columns(2)
                a.metric("Humidity", display_sensor_value(latest, "Humidity"))
                b.metric("Anomaly score", safe_number(latest.get("Score"), 5))
                c, d = st.columns(2)
                c.metric("CO", display_sensor_value(latest, "CO"))
                d.metric("Smoke", display_sensor_value(latest, "Smoke"))
                st.markdown("---")
                st.caption("READING TIMESTAMP")
                st.write(latest["Time"].strftime("%d %b %Y · %H:%M:%S"))
                if latest_status == "anomaly":
                    st.caption("MODEL REASON")
                    st.write(latest.get("Reason") or "No explanation was saved for this reading.")
                st.caption("Anomaly scores are relative unusualness scores, not probabilities of failure.")
            else:
                st.caption("No latest reading available.")

    st.markdown("")
    low_left, low_right = st.columns([1, 1], gap="large")

    with low_left:
        with st.container(border=True):
            section_heading("History composition", "Distribution of labels in loaded records")
            if has_data:
                distribution = pd.DataFrame({"Status": ["Normal", "Anomaly"], "Count": [total_normal, total_anomalies]})
                fig = px.bar(
                    distribution, x="Status", y="Count", color="Status", text="Count",
                    color_discrete_map={"Normal": "#56E0D0", "Anomaly": "#FFC76A"},
                )
                fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Readings")
                fig.update_traces(textposition="outside")
                style_plotly(fig, 280)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.caption("Distribution will appear after readings are recorded.")

    with low_right:
        with st.container(border=True):
            section_heading("Sensor snapshot", "Fields recorded for the latest reading")
            if has_data:
                sensor_names = ["Temperature", "Humidity", "CO", "LPG", "Smoke", "Light", "Motion"]
                snapshot = pd.DataFrame({
                    "Sensor": sensor_names,
                    "Latest value": [display_sensor_value(latest, name) for name in sensor_names],
                })
                st.dataframe(snapshot, use_container_width=True, hide_index=True, height=285)
            else:
                st.caption("Sensor values will appear here.")


# ============================================================
# SENSOR ANALYTICS
# ============================================================
with tab_analytics:
    st.markdown("### Sensor analytics")
    st.caption("Explore recent sensor trends. Different measurements use different units and scales, so compare them thoughtfully.")

    if has_data:
        available = [c for c in ["Temperature", "Humidity", "CO", "LPG", "Smoke", "Score"] if c in sensor_df.columns]
        selected_metrics = st.multiselect(
            "Choose measurements", available,
            default=[c for c in ["Temperature", "Humidity"] if c in available],
            key="analytics_measurements",
        )
        rows_to_show = st.slider("Number of recent readings", min_value=25, max_value=CHART_RECORDS, value=min(100, CHART_RECORDS), step=25)
        chart_df = sensor_df.tail(rows_to_show).copy()

        if selected_metrics:
            chart_mode = st.radio("Chart style", ["Separate charts", "Combined chart"], horizontal=True)
            if chart_mode == "Combined chart":
                melted = chart_df[["Time"] + selected_metrics].melt(id_vars="Time", var_name="Measurement", value_name="Value")
                fig = px.line(melted, x="Time", y="Value", color="Measurement")
                fig.update_traces(line=dict(width=2.1))
                style_plotly(fig, 420)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                for start in range(0, len(selected_metrics), 2):
                    cols = st.columns(2)
                    for j, metric in enumerate(selected_metrics[start:start + 2]):
                        with cols[j]:
                            fig = px.line(chart_df, x="Time", y=metric, title=metric)
                            fig.update_traces(line=dict(color="#56E0D0", width=2.2))
                            style_plotly(fig, 300)
                            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Choose at least one measurement.")
    else:
        st.info("Sensor analytics will be available once the history file contains valid readings.")


# ============================================================
# ANOMALY HISTORY
# ============================================================
with tab_anomalies:
    st.markdown("### Anomaly history")
    st.caption("Review readings flagged by the model. Flags indicate unusual patterns, not confirmed faults.")

    if has_data:
        anomaly_df = sensor_df[sensor_df["Status"] == "anomaly"].copy().sort_values("Time", ascending=False)
        filter_col1, filter_col2 = st.columns([1, 1])
        with filter_col1:
            search_text = st.text_input("Filter by reason or device", placeholder="Type a keyword…", key="anomaly_search")
        with filter_col2:
            max_rows = st.selectbox("Rows to display", [10, 20, 50, 100], index=1)

        if search_text:
            mask = (
                anomaly_df["Reason"].astype(str).str.contains(search_text, case=False, na=False)
                | anomaly_df["Device"].astype(str).str.contains(search_text, case=False, na=False)
            )
            anomaly_df = anomaly_df[mask]

        if anomaly_df.empty:
            st.success("No anomaly records match the current filter.")
        else:
            display_columns = ["Time", "Device", "Temperature", "Humidity", "CO", "LPG", "Smoke", "Light", "Motion", "Score", "Reason"]
            display_columns = [c for c in display_columns if c in anomaly_df.columns]
            st.dataframe(anomaly_df[display_columns].head(max_rows), use_container_width=True, hide_index=True)
            st.caption(f"Showing {min(max_rows, len(anomaly_df))} of {len(anomaly_df)} matching anomaly records.")
            if "Score" in anomaly_df.columns and anomaly_df["Score"].notna().any():
                with st.container(border=True):
                    section_heading("Highest anomaly scores", "Largest scores in the currently loaded anomaly history")
                    top_scores = anomaly_df.nlargest(10, "Score")
                    score_fig = px.bar(top_scores, x="Score", y=top_scores["Time"].dt.strftime("%d %b %H:%M:%S"), orientation="h")
                    score_fig.update_layout(xaxis_title="Anomaly score", yaxis_title="Reading time")
                    style_plotly(score_fig, 330)
                    st.plotly_chart(score_fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Anomaly history will appear after sensor readings have been recorded.")


# ============================================================
# TEMPERATURE EVENTS
# ============================================================
with tab_events:
    st.markdown("### Temperature event investigation")
    st.caption("Historical exploratory groups created by the temperature-drop investigation script.")

    if events_df.empty:
        st.info(
            f"No event summary file found at `{EVENTS_FILE}`. This section uses the existing historical CSV; "
            "it does not generate or update events automatically."
        )
    else:
        st.warning("These are candidate events based on a hand-chosen grouping rule, not confirmed equipment failures.")
        st.metric("Event groups loaded", f"{len(events_df):,}")
        st.dataframe(events_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download temperature event summary",
            data=events_df.to_csv(index=False).encode("utf-8"),
            file_name="skyguard_temperature_events.csv",
            mime="text/csv",
            key="download_events",
        )


# ============================================================
# MANUAL SENSOR TEST
# ============================================================
with tab_manual:
    st.markdown("### Manual sensor test")
    st.caption("Submit one sensor reading to FastAPI for a prediction. Manual tests are not automatically added to simulator history.")

    if not api_reachable:
        st.warning("FastAPI is currently unavailable. Start it with `uvicorn app:app --reload` before submitting a test.")

    with st.container(border=True):
        with st.form("manual_sensor_test"):
            c1, c2, c3 = st.columns(3)
            with c1:
                temperature = st.number_input("Temperature (°C)", value=22.0, step=0.1, format="%.2f")
                humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.5)
                co_value = st.number_input("CO", min_value=0.0, value=0.004, step=0.001, format="%.6f")
            with c2:
                lpg_value = st.number_input("LPG", min_value=0.0, value=0.007, step=0.001, format="%.6f")
                smoke_value = st.number_input("Smoke", min_value=0.0, value=0.02, step=0.001, format="%.6f")
            with c3:
                light_value = st.selectbox("Light detected", options=[False, True], format_func=lambda value: "Yes" if value else "No")
                motion_value = st.selectbox("Motion detected", options=[False, True], format_func=lambda value: "Yes" if value else "No")

            submitted = st.form_submit_button("Run anomaly detection", use_container_width=True)

    if submitted:
        payload = {
            "co": float(co_value),
            "humidity": float(humidity),
            "light": bool(light_value),
            "lpg": float(lpg_value),
            "motion": bool(motion_value),
            "smoke": float(smoke_value),
            "temp": float(temperature),
        }
        try:
            with st.spinner("Sending reading to the SkyGuard API…"):
                response = requests.post(API_URL, json=payload, timeout=10)

            if response.ok:
                result = response.json()
                status = str(result.get("status", "unknown")).lower()
                score = result.get("anomaly_score", result.get("score"))
                reason = result.get("reason", result.get("message", "No explanation returned by API."))

                st.markdown("---")
                if status == "anomaly":
                    st.error("⚑  ANOMALY FLAGGED — This reading differs from patterns learned by the model.")
                elif status == "normal":
                    st.success("●  NORMAL — The model did not flag this reading.")
                else:
                    st.warning(f"API returned status: {status}")

                result_col1, result_col2 = st.columns(2)
                result_col1.metric("Prediction status", status.upper())
                if score is not None:
                    result_col2.metric("Anomaly score", safe_number(score, 6))
                st.write("**Model explanation**")
                st.write(reason)
                st.caption("The anomaly score is not a probability of failure. Validate unusual readings before taking action.")
                with st.expander("View submitted sensor values"):
                    st.json(payload)
                with st.expander("View raw API response"):
                    st.json(result)
            else:
                st.error(f"API request failed with HTTP {response.status_code}.")
                st.code(response.text[:2000])
        except requests.RequestException as error:
            st.error("Could not connect to the SkyGuard AI API. Check the deployed API URL or your local FastAPI server.")
            st.caption(str(error))


# ============================================================
# DATA EXPORT
# ============================================================
with tab_export:
    st.markdown("### Data export")
    st.caption("Download the recent records loaded by this dashboard for further analysis.")

    if has_data:
        export_left, export_right = st.columns([1.4, 1], gap="large")
        with export_left:
            with st.container(border=True):
                section_heading("Sensor history", "Recent records loaded from the selected history source")
                st.metric("Rows available for export", f"{len(sensor_df):,}")
                export_df = sensor_df.copy()
                export_df["Time"] = export_df["Time"].astype(str)
                st.download_button(
                    "Download sensor history CSV",
                    data=export_df.to_csv(index=False).encode("utf-8"),
                    file_name="skyguard_sensor_history.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        with export_right:
            with st.container(border=True):
                section_heading("Export notes")
                st.write(f"**History source:** `{'Render API /history' if USE_REMOTE_HISTORY else HISTORY_FILE}`")
                st.write(f"**Maximum loaded records:** {MAX_HISTORY_RECORDS:,}")
                st.write(f"**Records currently loaded:** {len(sensor_df):,}")
                st.caption("This exports the records loaded by the dashboard, not necessarily the entire source CSV.")
    else:
        st.info("There is no sensor history to export yet.")


# ============================================================
# FOOTER / REFRESH
# ============================================================
st.markdown(
    """
    <div class="sg-footer">
        <strong>SKYGUARD AI</strong> · Environmental sensor anomaly monitoring
        <span style="float:right">Historical telemetry replay · Hackathon build</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if auto_refresh:
    time.sleep(refresh_seconds)
    st.cache_data.clear()
    st.rerun()
