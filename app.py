from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import time

import pandas as pd
import plotly.express as px
import streamlit as st

from traffic_mlops.config import DEFAULT_DATASET, MODEL_NAME, configure_mlflow
from traffic_mlops.data import generate_demo_data, normalize_data
from traffic_mlops.history import (
    log_incident,
    log_operator_action,
    recent_incidents,
    recent_operator_actions,
    recent_predictions,
    recent_virtual_alerts,
    recent_virtual_readings,
    save_prediction,
    save_virtual_tick,
)
from traffic_mlops.monitoring import monitor_data
from traffic_mlops.prediction import predict_next_hour
from traffic_mlops.training import train_model
from traffic_mlops.simulation import generate_virtual_tick


st.set_page_config(page_title="Traffic Intelligence | Project 28", page_icon="🚦", layout="wide")
st.markdown("""
<style>
.block-container { max-width: 1440px; padding-top: 1.6rem; }
h1, h2, h3 { letter-spacing: -.025em; }
[data-testid="stMetric"] { border: 1px solid #e7ebef; border-radius: 12px; padding: .8rem 1rem; background: #fff; }
.project-meta { color: #667085; font-size: .92rem; margin-bottom: .8rem; }
.small-note { color: #667085; font-size: .85rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def demo_data() -> pd.DataFrame:
    return normalize_data(generate_demo_data())


def get_data() -> pd.DataFrame:
    uploaded = st.session_state.get("traffic_data")
    if uploaded is not None:
        return normalize_data(uploaded)
    if DEFAULT_DATASET.exists():
        try:
            return normalize_data(DEFAULT_DATASET)
        except (ValueError, OSError):
            pass
    return demo_data()


def age_hours(timestamp: object) -> float:
    observed = pd.Timestamp(timestamp)
    now = pd.Timestamp.now(tz=observed.tz) if observed.tz is not None else pd.Timestamp.now()
    return max(0.0, (now - observed).total_seconds() / 3600)


def latest_by_junction(data: pd.DataFrame) -> pd.DataFrame:
    return data.sort_values("timestamp").groupby("junction", as_index=False).tail(1).sort_values("junction")


def model_status() -> tuple[bool, str]:
    try:
        configure_mlflow()
        from mlflow.tracking import MlflowClient

        version = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
        return True, str(version.version)
    except Exception:
        try:
            from traffic_mlops.prediction import EDGE_CACHE

            metadata = EDGE_CACHE / "champion.json"
            if metadata.exists():
                import json

                return True, f"{json.loads(metadata.read_text(encoding='utf-8'))['version']} (local)"
        except Exception:
            pass
        return False, "Not trained"


def make_forecast(data: pd.DataFrame, junction: str, vehicles: float, temperature: float, event: int, offline: bool = False) -> dict[str, object]:
    started = time.perf_counter()
    inputs = {"junction": junction, "vehicles": vehicles, "temperature_c": temperature, "is_event": event}
    result = predict_next_hour(data, **inputs, offline=offline)
    save_prediction(result, inputs)
    st.session_state["last_inference"] = {
        "at": datetime.now(timezone.utc).isoformat(),
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "junction": junction,
    }
    st.session_state["last_forecast"] = result
    return result


def render_profile(junctions: list[str]) -> None:
    with st.sidebar:
        st.markdown("### 🚦 Traffic desk")
        st.caption("Project 28 · AI-Based Traffic Prediction")
        page = st.radio("Workspace", [
            "Dashboard", "Virtual sensors", "Forecast & operations", "Incidents & alerts",
            "Junction zones", "AI assistant", "Analytics & reports", "MLOps & data",
            "Simulation lab",
        ], label_visibility="collapsed")
        st.divider()
        st.markdown("#### Operator profile")
        st.text_input("Operator name", key="operator_name", value=st.session_state.get("operator_name", "Traffic operator"))
        st.text_input("Control room / region", key="operator_region", value=st.session_state.get("operator_region", "Project 28 demo"))
        st.selectbox("Primary junction", junctions, key="primary_junction")
        st.caption("Session-only demo profile; no real identity or login service is required.")
        st.divider()
        st.markdown("#### Data source")
        upload = st.file_uploader("Load traffic CSV", type="csv", key="dataset_upload", help="Required: timestamp, junction, vehicles")
        signature = (upload.name, upload.size, hashlib.sha256(upload.getvalue()).hexdigest()[:12]) if upload is not None else None
        if upload is not None and signature != st.session_state.get("loaded_upload_signature") and signature != st.session_state.get("ignored_upload_signature"):
            try:
                upload.seek(0)
                st.session_state["traffic_data"] = normalize_data(pd.read_csv(upload))
                st.session_state["loaded_upload_signature"] = signature
                st.success(f"Loaded {len(st.session_state['traffic_data']):,} observations")
                st.rerun()
            except (ValueError, pd.errors.ParserError) as error:
                st.error(str(error))
        if st.button("Use project dataset", use_container_width=True):
            st.session_state.pop("traffic_data", None)
            st.session_state["ignored_upload_signature"] = signature
            st.rerun()
        st.caption("CSV: timestamp, junction, vehicles; optional temperature_c, is_event.")
    return page


def current_virtual_readings() -> list[dict[str, object]]:
    rows = recent_virtual_readings(500)
    rows.sort(key=lambda item: (item.get("tick", 0), item.get("recorded_at", "")), reverse=True)
    seen: set[str] = set()
    latest: list[dict[str, object]] = []
    for row in rows:
        if row["junction"] not in seen:
            latest.append(row)
            seen.add(str(row["junction"]))
    return sorted(latest, key=lambda item: str(item["junction"]))


def advance_virtual_tick(data: pd.DataFrame, model_ready: bool) -> list[dict[str, object]]:
    history = recent_virtual_readings(2000)
    tick = max((int(row.get("tick", 0)) for row in history), default=0) + 1
    cloud_online = st.session_state.get("sim_cloud_online", True)
    failed = st.session_state.get("sim_failed_junction")
    readings, alerts = generate_virtual_tick(data, tick, cloud_online=cloud_online,
                                              failed_junction=failed, model_ready=model_ready)
    save_virtual_tick(tick, readings, alerts)
    st.session_state["last_sim_tick"] = tick
    return readings


def render_dashboard(data: pd.DataFrame, virtual: list[dict[str, object]], available: bool, version: str) -> None:
    newest = data.loc[data["timestamp"].idxmax()]
    hours_old = age_hours(newest["timestamp"])
    st.title("Traffic operations overview")
    st.markdown('<div class="project-meta">Project 28 · K.Mahija (2400080129) · Sk.Anjum (2400080003) · P.Hasitha (2400080005)</div>', unsafe_allow_html=True)
    st.success("Software simulation mode · virtual sensors, weather, network, and signal controls are generated from the traffic history. No hardware is required.")
    if hours_old > 24:
        st.info(f"Training history spans through {pd.Timestamp(newest['timestamp']).strftime('%d %b %Y %H:%M')}; the live-style panels below use reproducible virtual telemetry.")
    head = virtual[0] if virtual else {}
    critical = any(row["sensor_status"] != "online" or row["road_occupancy_pct"] >= 82 for row in virtual)
    warning = any(row["road_occupancy_pct"] >= 68 or row["cloud_link"] != "online" for row in virtual)
    network_health = "Critical" if critical else ("Warning" if warning else "Good")
    top = st.columns(7)
    top[0].metric("Junctions", f"{data['junction'].nunique():,}")
    top[1].metric("Observations", f"{len(data):,}")
    top[2].metric("Simulated traffic flow", f"{head.get('vehicles_per_hour', 0):.0f} veh/h" if head else "—")
    top[3].metric("Simulated occupancy", f"{head.get('road_occupancy_pct', 0):.1f}%" if head else "—")
    top[4].metric("Simulated weather", f"{head.get('temperature_c', 0):.1f} °C · {head.get('humidity_pct', 0):.0f}% RH" if head else "—")
    top[5].metric("Champion model", f"v{version}" if available else "Not trained")
    top[6].metric("Overall network health", network_health)

    left, right = st.columns([1.65, 1])
    with left:
        st.subheader("Traffic by junction")
        daily = data.groupby([pd.Grouper(key="timestamp", freq="D"), "junction"], as_index=False)["vehicles"].mean()
        st.plotly_chart(px.line(daily, x="timestamp", y="vehicles", color="junction",
                                labels={"vehicles": "Vehicles per hour", "timestamp": "Observation time"}), use_container_width=True)
    with right:
        st.subheader("Virtual operations alerts")
        alerts = recent_virtual_alerts(8)
        if not alerts:
            st.success("No simulated alerts on the current ticks.")
        for alert in alerts[:5]:
            message = f"**{alert['severity'].upper()} · {alert['junction']}**\n\n{alert['alert']}"
            if alert["severity"] == "critical":
                st.error(message)
            elif alert["severity"] == "warning":
                st.warning(message)
            else:
                st.info(message)

    st.subheader("Virtual junction fleet")
    if virtual:
        fleet = pd.DataFrame(virtual)
        st.dataframe(fleet[["junction", "sensor_status", "vehicles_per_hour", "average_speed_kmh", "road_occupancy_pct",
                            "temperature_c", "humidity_pct", "rain_probability_pct", "edge_model_status", "cloud_link"]],
                     use_container_width=True, hide_index=True)
        st.caption(f"Virtual tick {fleet['tick'].max()} · values are reproducible software simulations derived from historical traffic patterns.")

    st.subheader("Historical baseline")
    baseline = latest_by_junction(data).rename(columns={"junction": "Junction", "timestamp": "Last source observation", "vehicles": "Vehicles/hour"})
    st.dataframe(baseline[["Junction", "Last source observation", "Vehicles/hour"]], use_container_width=True, hide_index=True)


def render_live(data: pd.DataFrame, junctions: list[str], model_ready: bool) -> None:
    st.title("Virtual sensor monitoring")
    st.caption("Every reading below is generated in software from historical per-junction traffic patterns. Advance ticks in Simulation lab.")
    junction = st.selectbox("Junction", junctions, key="live_junction")
    history = pd.DataFrame([row for row in recent_virtual_readings(2000) if row["junction"] == junction])
    if history.empty:
        st.info("Generate a virtual sensor tick in Simulation lab to start the feed.")
        return
    history = history.sort_values("tick")
    last = history.iloc[-1]
    metrics = st.columns(6)
    metrics[0].metric("Virtual flow", f"{last['vehicles_per_hour']:.0f} veh/h")
    metrics[1].metric("Average speed", f"{last['average_speed_kmh']:.1f} km/h")
    metrics[2].metric("Occupancy", f"{last['road_occupancy_pct']:.1f}%")
    metrics[3].metric("Temperature / RH", f"{last['temperature_c']:.1f} °C / {last['humidity_pct']:.0f}%")
    metrics[4].metric("Rain probability", f"{last['rain_probability_pct']:.0f}%")
    metrics[5].metric("Virtual sensor", str(last["sensor_status"]))
    st.plotly_chart(px.line(history, x="recorded_at", y="vehicles_per_hour", markers=True,
                            labels={"recorded_at": "Simulation time", "vehicles_per_hour": "Vehicles per hour"}), use_container_width=True)
    st.plotly_chart(px.line(history, x="recorded_at", y=["average_speed_kmh", "road_occupancy_pct"], markers=True,
                            labels={"recorded_at": "Simulation time", "value": "Value", "variable": "Virtual sensor"}), use_container_width=True)
    st.plotly_chart(px.line(history, x="recorded_at", y=["temperature_c", "humidity_pct", "rain_probability_pct"], markers=True,
                            labels={"recorded_at": "Simulation time", "value": "Value", "variable": "Virtual environment"}), use_container_width=True)
    st.caption(f"Virtual tick {int(last['tick'])} · Last updated {pd.Timestamp(last['recorded_at']).strftime('%d %b %Y, %H:%M:%S %Z')}")


def render_forecast(data: pd.DataFrame, junctions: list[str]) -> None:
    st.title("Traffic forecast & operations")
    st.write("Estimate next-hour demand, compare it with recent junction volume, and record a human-reviewed operations advisory.")
    junction = st.selectbox("Junction", junctions, key="forecast_junction")
    latest = data[data["junction"] == junction].iloc[-1]
    virtual = next((row for row in current_virtual_readings() if row["junction"] == junction), {})
    left, right = st.columns([1, 1.15])
    with left:
        vehicles = st.number_input("Current virtual vehicles / hour", 0.0, value=float(virtual.get("vehicles_per_hour", latest["vehicles"])), step=10.0)
        temperature = st.number_input("Virtual temperature °C", value=float(virtual.get("temperature_c", latest["temperature_c"])), step=1.0)
        event = st.toggle("Simulate special event", value=bool(latest["is_event"]))
        offline = not st.session_state.get("sim_cloud_online", True)
        st.caption("Offline mode uses the host-local champion cache." if offline else "Online mode uses the MLflow champion alias.")
        run = st.button("Generate next-hour forecast", type="primary", use_container_width=True)
    with right:
        if run:
            try:
                result = make_forecast(data, junction, vehicles, temperature, int(event), offline=offline)
                baseline = data[data["junction"] == junction].tail(24)["vehicles"].mean()
                delta_pct = (result["predicted_vehicles"] - baseline) / max(baseline, 1) * 100
                st.metric("Predicted next-hour volume", f"{result['predicted_vehicles']:.0f} vehicles", f"{delta_pct:+.1f}% vs trailing 24 h avg")
                st.caption(f"Forecast timestamp: {pd.Timestamp(result['forecast_for']).strftime('%d %b %Y, %H:%M')} · Inference {st.session_state['last_inference']['latency_ms']} ms · {'local edge cache' if offline else 'MLflow champion'}")
                if delta_pct >= 15:
                    st.warning("Demand is materially above the recent baseline. Review signal timing and diversion options.")
                else:
                    st.success("Forecast is near or below the recent baseline. Continue routine monitoring.")
            except Exception as error:
                st.error(f"Forecast unavailable: {error}. Train a champion model in MLOps & data.")
        elif st.session_state.get("last_forecast"):
            result = st.session_state["last_forecast"]
            st.metric("Last forecast", f"{result['predicted_vehicles']:.0f} vehicles")
            st.caption(f"{result['junction']} · {result['forecast_for']}")
        else:
            st.info("Run a forecast to see the estimate and a traffic-plan advisory.")

    st.divider()
    st.subheader("Virtual signal controller")
    mode = st.radio("Control mode", ["Automatic", "Manual"], horizontal=True, key="signal_mode")
    occupancy = float(virtual.get("road_occupancy_pct", 45))
    auto_green = int(max(20, min(70, 20 + occupancy * .42)))
    manual_green = st.slider("Manual green phase (seconds)", 15, 90, 45, disabled=(mode == "Automatic"), key="manual_green")
    selected_green = auto_green if mode == "Automatic" else manual_green
    st.metric("Simulated green phase", f"{selected_green} seconds", "computed from occupancy" if mode == "Automatic" else "manual setting")
    st.caption("Virtual controller only. No signal hardware receives this setting.")

    st.subheader("Record an operator action")
    with st.form("operator_action"):
        action = st.selectbox("Advisory action", ["Apply simulated signal phase", "Review signal timing", "Prepare diversion notice", "Dispatch virtual field check", "No action; monitor"])
        note = st.text_input("Operator note", placeholder="Reason or follow-up")
        submitted = st.form_submit_button("Save to action history")
        if submitted:
            action_note = f"Mode={mode}; green={selected_green}s. {note}".strip()
            log_operator_action(junction, action, action_note)
            st.success("Virtual action recorded in the software simulator.")
    actions = pd.DataFrame(recent_operator_actions(10))
    if not actions.empty:
        st.dataframe(actions, use_container_width=True, hide_index=True)


def render_incidents(data: pd.DataFrame, junctions: list[str]) -> None:
    st.title("Incident intake & alerts")
    st.caption("Upload or capture a virtual incident scenario; the traffic-flow model estimates numeric congestion risk from the selected junction history.")
    left, right = st.columns([1, 1])
    with left:
        image = st.file_uploader("Upload incident image", type=["png", "jpg", "jpeg"], key="incident_upload")
        if image is None:
            image = st.camera_input("Capture incident image", key="incident_camera")
        if image is not None:
            st.image(image, caption=image.name if hasattr(image, "name") else "Camera capture", use_container_width=True)
    with right:
        risk_junction = st.selectbox("Run AI traffic-risk estimate for", junctions, key="incident_risk_junction")
        if st.button("Estimate next-hour congestion risk", type="primary"):
            try:
                prediction = predict_next_hour(data, risk_junction,
                    vehicles=float(next((r["vehicles_per_hour"] for r in current_virtual_readings() if r["junction"] == risk_junction), 0)) or None,
                    offline=not st.session_state.get("sim_cloud_online", True))
                p95 = float(data[data["junction"] == risk_junction]["vehicles"].quantile(.95))
                ratio = float(prediction["predicted_vehicles"]) / max(p95, 1) * 100
                label = "Critical" if ratio >= 100 else ("Elevated" if ratio >= 80 else "Within recent range")
                st.metric("Model-based congestion risk", label, f"{ratio:.1f}% of historical 95th percentile")
                st.caption(f"Predicted flow: {prediction['predicted_vehicles']:.0f} veh/h · Forecast time: {prediction['forecast_for']}. This is a threshold score, not a calibrated probability.")
            except Exception as error:
                st.warning(f"Risk inference unavailable: {error}")
        with st.form("incident_form"):
            junction = st.selectbox("Junction", junctions, key="incident_junction")
            category = st.selectbox("Incident type", ["Congestion", "Collision (operator reported)", "Road closure", "Vehicle breakdown", "Weather hazard", "Other"])
            severity = st.select_slider("Operator-assessed severity", ["Low", "Moderate", "High", "Critical"], value="Moderate")
            summary = st.text_area("Description", placeholder="Describe what was observed; do not include sensitive personal details.")
            submitted = st.form_submit_button("Save incident report", type="primary")
            if submitted:
                if not summary.strip():
                    st.error("Add a short description before saving.")
                else:
                    image_name = getattr(image, "name", "camera-capture.jpg") if image is not None else None
                    log_incident(junction, category, severity, summary.strip(), image_name)
                    st.success("Virtual incident scenario saved to the alert history.")
    st.subheader("Virtual sensor alert history")
    alerts = pd.DataFrame(recent_virtual_alerts(100))
    if not alerts.empty:
        st.dataframe(alerts, use_container_width=True, hide_index=True)
    else:
        st.success("No threshold alerts in latest available observations.")
    st.subheader("Incident history")
    incidents = pd.DataFrame(recent_incidents(100))
    if incidents.empty:
        st.info("No operator incident reports yet.")
    else:
        st.dataframe(incidents, use_container_width=True, hide_index=True)


def render_zones(data: pd.DataFrame, virtual: list[dict[str, object]]) -> None:
    st.title("Junction zones")
    st.caption("Schematic junction map using simulated sensor zones; it does not require physical devices or geographic coordinates.")
    if not virtual:
        st.info("Generate a virtual sensor tick in Simulation lab to populate the zones.")
        return
    latest = pd.DataFrame(virtual).copy()
    latest["status"] = latest["road_occupancy_pct"].apply(lambda x: "Critical" if x >= 82 else ("Needs attention" if x >= 68 else "Healthy"))
    latest["zone"] = latest["junction"].astype(str)
    colors = {"Healthy": "#16a34a", "Needs attention": "#eab308", "Critical": "#dc2626"}
    fig = px.treemap(latest, path=["zone"], values="vehicles_per_hour", color="status", color_discrete_map=colors,
                     hover_data={"vehicles_per_hour": ":.0f", "road_occupancy_pct": ":.1f", "temperature_c": ":.1f", "recorded_at": True, "status": True})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(latest[["junction", "recorded_at", "vehicles_per_hour", "average_speed_kmh", "road_occupancy_pct", "status"]].rename(columns={
        "junction": "Virtual junction", "recorded_at": "Simulation time", "vehicles_per_hour": "Flow veh/h", "average_speed_kmh": "Speed km/h", "road_occupancy_pct": "Occupancy %"
    }), use_container_width=True, hide_index=True)


def assistant_response(question: str, data: pd.DataFrame, junction: str) -> str:
    selected = data[data["junction"] == junction].sort_values("timestamp")
    current = selected.iloc[-1]
    mean24 = selected.tail(24)["vehicles"].mean()
    q = question.lower()
    if any(word in q for word in ("weather", "rain", "wind", "humidity")):
        virtual = next((r for r in current_virtual_readings() if r["junction"] == junction), None)
        if virtual:
            return (f"Virtual weather scenario for {junction}: {virtual['temperature_c']:.1f} °C, humidity {virtual['humidity_pct']:.0f}%, "
                    f"rain probability {virtual['rain_probability_pct']:.0f}%. These are simulated values, not a live forecast.")
        return "The source CSV has no weather fields. Generate a virtual telemetry tick to inspect simulated weather conditions."
    if any(word in q for word in ("predict", "next hour", "congestion", "traffic", "route", "divert", "signal")):
        if "congestion" in q or "route" in q or "divert" in q:
            virtual = next((r for r in current_virtual_readings() if r["junction"] == junction), None)
            flow = virtual["vehicles_per_hour"] if virtual else current["vehicles"]
            suffix = f" Virtual occupancy is {virtual['road_occupancy_pct']:.0f}%." if virtual else ""
            return (f"At {junction}, current {'simulated ' if virtual else ''}flow is {flow:.0f} vehicles/hour and the training-history trailing 24-hour mean is "
                    f"{mean24:.0f}.{suffix} Review simulator incidents and compare alternate junctions before changing a virtual signal plan.")
        try:
            pred = predict_next_hour(data, junction, offline=not st.session_state.get("sim_cloud_online", True))
            return (f"The champion model estimates {pred['predicted_vehicles']:.0f} vehicles/hour at {pred['forecast_for']} for {junction}. "
                    "This is a model forecast from the latest historical row. Use Simulation lab to exercise virtual sensor and offline modes.")
        except Exception as error:
            return f"A forecast is unavailable right now ({error}). Train a champion model and provide at least one week of hourly history."
    return (f"I can summarize this project's traffic data for {junction}: latest flow {current['vehicles']:.0f} vehicles/hour, "
            f"trailing 24-hour average {mean24:.0f}. "
            "Ask about traffic, the next-hour forecast, congestion, or available weather fields.")


def render_assistant(data: pd.DataFrame, junctions: list[str]) -> None:
    st.title("Traffic AI assistant")
    st.caption("Answers use the local traffic dataset and registered forecast model; this is a grounded project assistant, not a general live-data agent.")
    junction = st.selectbox("Context junction", junctions, key="assistant_junction")
    if "assistant_messages" not in st.session_state:
        st.session_state["assistant_messages"] = []
    for message in st.session_state["assistant_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    question = st.chat_input("Ask about traffic, forecasts, routes, or weather data")
    if question:
        answer = assistant_response(question, data, junction)
        st.session_state["assistant_messages"].extend([
            {"role": "user", "content": question}, {"role": "assistant", "content": answer}
        ])
        st.rerun()
    if st.button("Clear conversation"):
        st.session_state["assistant_messages"] = []
        st.rerun()


def render_simulation_lab(data: pd.DataFrame, model_ready: bool) -> None:
    st.title("Software simulation lab")
    st.caption("Reproducible virtual IoT, weather, 5G, edge, alert, and signal-control scenarios. No physical hardware is required.")
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            if "sim_cloud_online" not in st.session_state:
                st.session_state["sim_cloud_online"] = True
            st.toggle("Simulated cloud link online", key="sim_cloud_online",
                      help="Turn off to test local edge inference and offline-to-cloud alerts.")
        with right:
            st.selectbox("Inject virtual sensor fault", ["None"] + sorted(data["junction"].unique().tolist()), key="sim_fault_selector")
        if st.button("Generate next virtual minute", type="primary", use_container_width=True):
            st.session_state["sim_failed_junction"] = None if st.session_state["sim_fault_selector"] == "None" else st.session_state["sim_fault_selector"]
            advance_virtual_tick(data, model_ready)
            st.rerun()
    virtual = current_virtual_readings()
    if virtual:
        frame = pd.DataFrame(virtual)
        latest = frame.iloc[0]
        network = st.columns(5)
        network[0].metric("Virtual tick", str(int(latest["tick"])))
        network[1].metric("5G signal", f"{latest['signal_strength_dbm']} dBm")
        network[2].metric("Upload", f"{latest['upload_mbps']:.1f} Mbps")
        network[3].metric("Download", f"{latest['download_mbps']:.1f} Mbps")
        network[4].metric("Network latency", f"{latest['network_latency_ms']:.1f} ms" if latest["network_latency_ms"] is not None else "Offline")
        st.subheader("Virtual device fleet")
        st.dataframe(frame[["junction", "sensor_status", "edge_model_status", "edge_inference_latency_ms", "cloud_link",
                            "signal_strength_dbm", "upload_mbps", "download_mbps", "network_latency_ms"]],
                     use_container_width=True, hide_index=True)
    else:
        st.info("Generate the first virtual tick to populate the simulated sensors.")
    st.subheader("Edge AI and offline failover")
    from traffic_mlops.prediction import EDGE_CACHE

    cache_ready = (EDGE_CACHE / "champion.json").exists()
    ready, version = model_status()
    st.write({"MLflow champion": f"version {version}" if ready else "not trained",
              "Host-local edge cache": "ready" if cache_ready else "not created yet",
              "Simulated cloud link": "online" if st.session_state.get("sim_cloud_online", True) else "offline",
              "Offline inference": "Available" if cache_ready else "Train/promote a model to create the local cache"})
    if not cache_ready:
        st.info("After a model is promoted, the MLOps pipeline caches its MLflow artifact locally for offline simulation.")
    st.subheader("Virtual alerts")
    alerts = pd.DataFrame(recent_virtual_alerts(100))
    if alerts.empty:
        st.success("No simulated alerts on the generated ticks.")
    else:
        st.dataframe(alerts, use_container_width=True, hide_index=True)
    st.caption("All network, weather, sensor fault, and signal values are generated by the project simulator and labeled virtual.")


def render_analytics(data: pd.DataFrame) -> None:
    st.title("Analytics & reports")
    junctions = sorted(data["junction"].unique().tolist())
    selected = st.multiselect("Junctions", junctions, default=junctions)
    hours = st.select_slider("History window", options=[24, 72, 168, 720, 5000], value=720,
                             format_func=lambda n: f"{n} hours" if n < 5000 else "Full history")
    filtered = data[data["junction"].isin(selected)].copy()
    if hours < 5000:
        max_time = filtered["timestamp"].max()
        filtered = filtered[filtered["timestamp"] >= max_time - pd.Timedelta(hours=hours)]
    if filtered.empty:
        st.info("Select at least one junction.")
        return
    smoothed = filtered.sort_values("timestamp").copy()
    smoothed["rolling_24"] = smoothed.groupby("junction")["vehicles"].transform(lambda s: s.rolling(24, min_periods=1).mean())
    tab_flow, tab_weather, tab_forecasts, tab_history = st.tabs(["Traffic flow", "Weather context", "Forecast history", "Operations history"])
    with tab_flow:
        st.plotly_chart(px.line(smoothed, x="timestamp", y="vehicles", color="junction", labels={"vehicles": "Vehicles/hour"}), use_container_width=True)
        st.plotly_chart(px.line(smoothed, x="timestamp", y="rolling_24", color="junction", labels={"rolling_24": "24-hour rolling mean"}), use_container_width=True)
    with tab_weather:
        virtual = pd.DataFrame(recent_virtual_readings(2000))
        if virtual.empty:
            st.info("Generate simulated telemetry in Simulation lab.")
        else:
            st.plotly_chart(px.line(virtual.sort_values("tick"), x="recorded_at", y=["temperature_c", "humidity_pct", "rain_probability_pct"],
                                    color="junction", labels={"recorded_at": "Simulation time", "value": "Virtual reading", "variable": "Metric"}), use_container_width=True)
            st.caption("Weather variables are synthetic simulator readings; the source CSV does not contain weather observations.")
    with tab_forecasts:
        predictions = pd.DataFrame(recent_predictions(500))
        if predictions.empty:
            st.info("Forecasts will appear after the first prediction.")
        else:
            st.plotly_chart(px.scatter(predictions, x="created_at", y="predicted_vehicles", color="junction",
                                       hover_data=["forecast_for", "id"]), use_container_width=True)
            st.dataframe(predictions.drop(columns=["inputs"]), use_container_width=True, hide_index=True)
    with tab_history:
        incidents = pd.DataFrame(recent_incidents(500))
        actions = pd.DataFrame(recent_operator_actions(500))
        st.markdown("**Incident reports**")
        st.dataframe(incidents, use_container_width=True, hide_index=True) if not incidents.empty else st.caption("No reports recorded.")
        st.markdown("**Operator action history**")
        st.dataframe(actions, use_container_width=True, hide_index=True) if not actions.empty else st.caption("No actions recorded.")

    st.divider()
    st.subheader("Download report")
    report_date = filtered["timestamp"].max().strftime("%Y-%m-%d")
    summary = (f"# Traffic report — {report_date}\n\n"
               f"Observations: {len(filtered):,}  \nJunctions: {filtered['junction'].nunique()}  \n"
               f"Mean flow: {filtered['vehicles'].mean():.1f} vehicles/hour  \n"
               f"Peak flow: {filtered['vehicles'].max():.0f} vehicles/hour\n\n"
               "The source data has no weather fields. Virtual telemetry is simulated separately.\n")
    virtual = pd.DataFrame(recent_virtual_readings(2000))
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download report (Markdown)", summary, file_name=f"traffic-report-{report_date}.md", mime="text/markdown")
    c2.download_button("Download observations (CSV)", filtered.to_csv(index=False), file_name=f"traffic-observations-{report_date}.csv", mime="text/csv")
    if not virtual.empty:
        c3.download_button("Download virtual telemetry (CSV)", virtual.to_csv(index=False), file_name="virtual-telemetry.csv", mime="text/csv")


def render_mlops(data: pd.DataFrame) -> None:
    st.title("MLOps & data")
    st.write("Train and evaluate on a chronological holdout, register every version, promote only an improved champion, and track feature drift.")
    train_col, monitor_col = st.columns(2)
    with train_col:
        if st.button("Train and register model", type="primary", use_container_width=True):
            with st.spinner("Training and logging the model to MLflow…"):
                try:
                    st.session_state["last_training"] = train_model(data)
                except Exception as error:
                    st.error(f"Training failed: {error}")
        result = st.session_state.get("last_training")
        if result:
            st.success(f"{'Promoted to champion' if result['promoted'] else 'Registered as candidate'} · version {result['model_version']}")
            metrics = st.columns(3)
            for column, name in zip(metrics, ("rmse", "mae", "r2")):
                column.metric(name.upper(), f"{result['metrics'][name]:.3f}")
            st.caption(f"Run ID: {result['run_id']}")
    with monitor_col:
        if st.button("Run drift check", use_container_width=True):
            with st.spinner("Comparing historical and recent feature distributions…"):
                try:
                    st.session_state["last_monitor"] = monitor_data(data)
                except Exception as error:
                    st.error(f"Monitoring failed: {error}")
        drift = st.session_state.get("last_monitor")
        if drift:
            st.metric("Drift status", drift["status"].replace("_", " ").title(), f"max PSI {drift['max_psi']:.3f}")
            st.caption(drift["recommendation"])
            st.json(drift["features"])
    st.divider()
    st.markdown("**MLflow:** start `mlflow ui --backend-store-uri sqlite:///mlflow_project28.db` and open `http://localhost:5000`.")
    st.caption(f"Registered model: {MODEL_NAME} · champion alias updates only when validation RMSE improves.")
    st.subheader("Data quality")
    st.write({"rows": len(data), "junctions": data["junction"].nunique(), "missing_values": int(data.isna().sum().sum()),
              "first_record": str(data["timestamp"].min()), "last_record": str(data["timestamp"].max())})
    st.download_button("Download normalized dataset", data.to_csv(index=False), file_name="traffic-normalized.csv", mime="text/csv")


data = get_data()
junctions = sorted(data["junction"].unique().tolist())
configure_mlflow()
model_ready, model_version = model_status()
virtual = current_virtual_readings()
if not virtual:
    advance_virtual_tick(data, model_ready)
    virtual = current_virtual_readings()
page = render_profile(junctions)

if page == "Dashboard":
    render_dashboard(data, virtual, model_ready, model_version)
elif page == "Virtual sensors":
    render_live(data, junctions, model_ready)
elif page == "Forecast & operations":
    render_forecast(data, junctions)
elif page == "Incidents & alerts":
    render_incidents(data, junctions)
elif page == "Junction zones":
    render_zones(data, virtual)
elif page == "AI assistant":
    render_assistant(data, junctions)
elif page == "Analytics & reports":
    render_analytics(data)
elif page == "MLOps & data":
    render_mlops(data)
else:
    render_simulation_lab(data, model_ready)
