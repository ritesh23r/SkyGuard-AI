# SkyGuard AI — Sensor Anomaly Detection

SkyGuard AI is a machine-learning prototype designed to identify unusual environmental sensor readings for Automatic Weather Station monitoring.

## Features

- Unsupervised anomaly detection using Isolation Forest
- FastAPI prediction service deployed on Render
- Streamlit monitoring dashboard
- Neon PostgreSQL for persistent prediction history
- Rule-based temperature-drop detection
- Historical telemetry replay through a sensor simulator

## Architecture

Historical CSV → Sensor Simulator → FastAPI → Isolation Forest → Prediction and Anomaly Score → Neon PostgreSQL → Streamlit Dashboard

## ML Model

The Isolation Forest model analyzes seven features:

`co`, `humidity`, `light`, `lpg`, `motion`, `smoke`, `temp`

The model identifies readings that are easier to isolate through randomized trees.

- `1`: not flagged as anomalous
- `-1`: flagged as anomalous

The API returns a readable status and an anomaly score. The score is not a probability of sensor failure.

## Deployment

- API: https://skyguard-ai-c1wd.onrender.com/
- API documentation: https://skyguard-ai-c1wd.onrender.com/docs
- Prediction history: https://skyguard-ai-c1wd.onrender.com/history

## Important Limitations

The simulator replays historical dataset readings rather than collecting measurements from physical weather-station hardware. An anomaly indicates an unusual pattern, not a confirmed sensor fault. Real-world performance requires validation against labelled faults or expert-reviewed readings.

## Future Work

- Integrate physical sensors or a trusted live data feed
- Evaluate false positives and missed faults
- Improve alert explanations and monitoring
- Add device-level identification and operational safeguards
