from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from traffic_mlops import ui
from traffic_mlops.advisory import (
    categorize_alerts,
    corridor_plans,
    device_recommendation,
    signal_plan,
    triage_image,
    weather_index,
    weather_suggestions,
)
from traffic_mlops.config import DEFAULT_DATASET, MODEL_NAME, configure_mlflow
from traffic_mlops.data import generate_demo_data, normalize_data
from traffic_mlops.history import (
    acknowledge_alert,
    list_devices,
    log_incident,
    log_operator_action,
    recent_incidents,
    recent_operator_actions,
    recent_predictions,
    recent_virtual_alerts,
    recent_virtual_readings,
    reset_virtual_history,
    save_prediction,
    save_virtual_tick,
    set_device_state,
)
from traffic_mlops.monitoring import monitor_data
from traffic_mlops.prediction import EDGE_CACHE, predict_horizon, predict_next_hour
from traffic_mlops.simulation import generate_virtual_tick
from traffic_mlops.training import champion_summary, train_model


st.set_page_config(page_title="Flow28 · Traffic Command Deck", page_icon="🚦", layout="wide")
st.session_state.setdefault("deck_theme", ui.DEFAULT_THEME)
st.session_state.setdefault("deck_depth", 78)
st.session_state.setdefault("deck_motion", True)
_linked_theme = st.query_params.get("theme")
if _linked_theme in ui.theme_names() and _linked_theme != st.session_state.get("theme_from_link"):
    st.session_state["deck_theme"] = _linked_theme
    st.session_state["theme_from_link"] = _linked_theme
ui.inject(st.session_state["deck_theme"], int(st.session_state["deck_depth"]), bool(st.session_state["deck_motion"]))

NAV = {
    "Operator access": "◈  Operator access",
    "Dashboard": "◉  Dashboard",
    "Live sensors": "▣  Live sensors",
    "Weather monitor": "☁  Weather monitor",
    "Alert centre": "△  Alert centre",
    "Junction map": "▦  Junction map",
    "Forecast & outlook": "◈  Forecast & outlook",
    "Smart control": "☰  Smart control",
    "Incident vision": "◫  Incident vision",
    "Corridor advisory": "✦  Corridor advisory",
    "AI assistant": "◇  AI assistant",
    "Edge & 5G": "≋  Edge & 5G",
    "Analytics & reports": "≡  Analytics & reports",
    "MLOps & data": "⊞  MLOps & data",
    "Simulation lab": "◎  Simulation lab",
}

PROFILE_DEFAULTS = {
    "operator_name": "Traffic operator",
    "operator_id": "OP-28-01",
    "operator_region": "Project 28 control room",
    "operator_role": "Traffic operator",
    "operator_shift": "Day (06:00–14:00)",
    "operator_coverage": 42.0,
}

ROLE_MATRIX = {
    "Traffic operator": "Monitor corridors, forecast, acknowledge alerts, operate devices",
    "Control room supervisor": "Operator rights plus advisory sign-off and device overrides",
    "Data engineer": "Train, register, monitor drift, manage the data source",
    "Analyst": "Explore analytics and export reports",
}

SHIFTS = ["Day (06:00–14:00)", "Evening (14:00–22:00)", "Night (22:00–06:00)"]

CATEGORY_TONE = {"congestion": "bad", "weather": "warn", "safety": "bad",
                 "sensor": "violet", "network": "info", "other": "good"}


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


def num(row: dict[str, object], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def pretty_time(value: object) -> str:
    stamp = pd.to_datetime(str(value), errors="coerce", utc=True)
    return "—" if pd.isna(stamp) else stamp.strftime("%d %b %Y, %H:%M")


def tidy_table(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("created_at", "updated_at", "recorded_at", "forecast_for"):
        if column in frame.columns:
            frame[column] = frame[column].map(pretty_time)
    return frame


def latest_by_junction(data: pd.DataFrame) -> pd.DataFrame:
    return data.sort_values("timestamp").groupby("junction", as_index=False).tail(1).sort_values("junction")


def junction_select(label: str, junctions: list[str], key: str) -> str:
    if key in st.session_state and st.session_state[key] not in junctions:
        del st.session_state[key]
    return st.selectbox(label, junctions, key=key)


def push_flash(message: str, kind: str = "success") -> None:
    st.session_state["flash"] = (kind, message)


def pop_flash() -> None:
    item = st.session_state.pop("flash", None)
    if item:
        kind, message = item
        getattr(st, kind)(message)


@st.cache_data(ttl=60, show_spinner=False)
def model_state() -> tuple[bool, dict[str, object]]:
    try:
        summary = champion_summary()
        if summary:
            return True, summary
    except Exception:
        pass
    metadata = EDGE_CACHE / "champion.json"
    if metadata.exists():
        try:
            cached = json.loads(metadata.read_text(encoding="utf-8"))
            return True, {
                "version": str(cached.get("version", "?")),
                "rmse": None,
                "mae": None,
                "r2": None,
                "dataset": "host-local edge cache",
            }
        except (OSError, ValueError):
            pass
    return False, {}


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


def virtual_history(ticks: int = 18) -> dict[str, list[float]]:
    rows = recent_virtual_readings(2000)
    rows.sort(key=lambda item: (int(item.get("tick", 0)), str(item.get("recorded_at", ""))))
    series: dict[str, list[float]] = {}
    for row in rows:
        series.setdefault(str(row["junction"]), []).append(float(row.get("vehicles_per_hour", 0) or 0))
    return {name: values[-ticks:] for name, values in series.items()}


def network_health(virtual: list[dict[str, object]]) -> str:
    if not virtual:
        return "Awaiting tick"
    if any(row["sensor_status"] != "online" or row["road_occupancy_pct"] >= 78 for row in virtual):
        return "Critical"
    if any(row["road_occupancy_pct"] >= 62 or row["cloud_link"] != "online" for row in virtual):
        return "Warning"
    return "Good"


def capacity_by_junction(data: pd.DataFrame) -> dict[str, float]:
    quantiles = data.groupby("junction")["vehicles"].quantile(.98)
    return {str(name): max(float(value) * 1.45, 1.0) for name, value in quantiles.items()}


def spare_capacity(data: pd.DataFrame, virtual: list[dict[str, object]]) -> float | None:
    capacity = capacity_by_junction(data)
    values = []
    for row in virtual:
        limit = capacity.get(str(row["junction"]))
        if limit:
            values.append(max(0.0, 1 - num(row, "vehicles_per_hour") / limit) * 100)
    return min(values) if values else None


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


def seed_virtual_history(data: pd.DataFrame, model_ready: bool, ticks: int = 24) -> None:
    """Rebuild the virtual feed after a simulator schema change, one tick every three simulated hours."""
    reset_virtual_history()
    anchor = datetime.now(timezone.utc)
    for offset in range(ticks, 0, -1):
        tick = ticks - offset + 1
        readings, alerts = generate_virtual_tick(
            data, tick, cloud_online=True, model_ready=model_ready,
            now=anchor - timedelta(hours=3 * offset),
        )
        save_virtual_tick(tick, readings, alerts)


def virtual_feed_current() -> bool:
    rows = recent_virtual_readings(200)
    return bool(rows) and all("wind_speed_kmph" in row and "visibility_km" in row for row in rows)


def init_profile(junctions: list[str]) -> None:
    for key, value in PROFILE_DEFAULTS.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("operator_corridor", junctions[0] if junctions else "Central")


def render_sidebar(junctions: list[str]) -> str:
    with st.sidebar:
        st.markdown(
            '<div class="side-brand"><div class="mark">◈</div><div><b>FLOW28</b>'
            "<small>TRAFFIC COMMAND DECK</small></div></div>",
            unsafe_allow_html=True,
        )
        selected = st.radio("Workspace", list(NAV.values()), label_visibility="collapsed")
        page = next(key for key, label in NAV.items() if label == selected)
        st.divider()
        st.markdown("#### Session operator")
        st.markdown(
            f"**{st.session_state['operator_name']}**  \n"
            f"`{st.session_state['operator_id']}` · {st.session_state['operator_role']}  \n"
            f"{st.session_state['operator_region']}",
        )
        st.caption("Edit the profile on the Operator access page. Session-only demo identity.")
        junction_select("Primary junction", junctions, "primary_junction")
        st.divider()
        st.markdown("#### Appearance")
        theme = st.selectbox(
            "Deck theme",
            ui.theme_names(),
            format_func=lambda name: ui.theme_labels()[name],
            key="deck_theme",
            help="Recolours every surface and swaps the generated background artwork.",
        )
        preview = ui.theme_preview(theme)
        if preview is not None:
            st.image(str(preview), width="stretch")
            st.slider("Background depth", 55, 95, key="deck_depth",
                      help="Higher values push the artwork further behind the interface.")
        ui.theme_swatches(theme)
        st.toggle("Interface motion", key="deck_motion", help="Ticker, signal lamps, and schematic animation.")
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
        if st.button("Use project dataset", width="stretch"):
            st.session_state.pop("traffic_data", None)
            st.session_state["ignored_upload_signature"] = signature
            st.rerun()
        st.caption("CSV: timestamp, junction, vehicles; optional temperature_c, is_event.")
        st.markdown(
            '<div class="footer-note">PROJECT 28 · SIMULATED TELEMETRY<br>MLFLOW MODEL REGISTRY · FASTAPI</div>',
            unsafe_allow_html=True,
        )
    return page


def render_operator_access(data: pd.DataFrame, junctions: list[str], ready: bool, summary: dict[str, object]) -> None:
    ui.page_head("MODULE 01 · ACCESS", "Operator access & profile",
                 "A session profile keeps the workspace personal. No password, identity service, or personal record is stored — this is a demo access layer.")
    profile = {
        "name": st.session_state["operator_name"],
        "operator_id": st.session_state["operator_id"],
        "meta": [
            ("Region", st.session_state["operator_region"]),
            ("Role", st.session_state["operator_role"]),
            ("Shift", st.session_state["operator_shift"]),
            ("Corridor", st.session_state["operator_corridor"]),
            ("Coverage", f"{float(st.session_state['operator_coverage']):.0f} km"),
        ],
        "tags": ["session only", "no password store"],
    }
    ui.profile_card(profile)
    ui.section("MODULE 01 · EDIT", "Session details")
    with st.form("profile_form"):
        left, right = st.columns(2, gap="large")
        with left:
            name = st.text_input("Operator name", value=st.session_state["operator_name"])
            role = st.selectbox("Role", list(ROLE_MATRIX),
                                index=list(ROLE_MATRIX).index(st.session_state["operator_role"])
                                if st.session_state["operator_role"] in ROLE_MATRIX else 0)
            region = st.text_input("Control room / region", value=st.session_state["operator_region"])
            shift = st.selectbox("Shift", SHIFTS,
                                 index=SHIFTS.index(st.session_state["operator_shift"])
                                 if st.session_state["operator_shift"] in SHIFTS else 0)
        with right:
            operator_id = st.text_input("Operator ID", value=st.session_state["operator_id"])
            corridor = st.selectbox("Assigned corridor", junctions,
                                    index=junctions.index(st.session_state["operator_corridor"])
                                    if st.session_state["operator_corridor"] in junctions else 0)
            coverage = st.number_input("Network coverage (km)", 1.0, 500.0,
                                       float(st.session_state["operator_coverage"]), 1.0)
            acknowledged = st.checkbox("I understand this is a session-only demo profile", value=True)
        if st.form_submit_button("Save session profile", type="primary", width="stretch"):
            if not name.strip() or not operator_id.strip():
                st.error("Operator name and ID cannot be empty.")
            elif not acknowledged:
                st.warning("Confirm the demo-session notice before saving.")
            else:
                st.session_state.update({
                    "operator_name": name.strip(),
                    "operator_id": operator_id.strip(),
                    "operator_region": region.strip() or "Project 28 control room",
                    "operator_role": role,
                    "operator_shift": shift,
                    "operator_corridor": corridor,
                    "operator_coverage": float(coverage),
                })
                push_flash("Session profile updated for this browser session.")
                st.rerun()

    ui.section("MODULE 01 · CONTEXT", "Workspace context")
    version = f"v{summary.get('version')}" if ready else "not trained"
    ui.kpi_grid([
        {"label": "Operator", "value": st.session_state["operator_name"], "sub": st.session_state["operator_id"], "tone": "info"},
        {"label": "Assigned corridor", "value": st.session_state["operator_corridor"], "sub": f"{len(junctions)} corridors in scope", "tone": "violet"},
        {"label": "Coverage", "value": f"{float(st.session_state['operator_coverage']):.0f} km", "sub": st.session_state["operator_shift"], "tone": "good"},
        {"label": "Champion model", "value": version, "sub": MODEL_NAME, "tone": "good" if ready else "bad"},
        {"label": "Source records", "value": f"{len(data):,}", "sub": "hourly observations", "tone": "info"},
        {"label": "Session mode", "value": "Demo", "sub": "no authentication service", "tone": "warn"},
    ])

    ui.section("MODULE 01 · ROLES", "Role capabilities")
    roles = pd.DataFrame(
        [{"Role": role_name, "Capabilities": caps, "Active": "yes" if role_name == st.session_state["operator_role"] else ""}
         for role_name, caps in ROLE_MATRIX.items()]
    )
    st.table(roles, hide_index=True)
    st.caption("The role selection changes the workspace narrative only; every demo capability stays available so reviewers can explore the full deck.")


def render_dashboard(data: pd.DataFrame, virtual: list[dict[str, object]], ready: bool, summary: dict[str, object]) -> None:
    newest = data.loc[data["timestamp"].idxmax()]
    hours_old = age_hours(newest["timestamp"])
    health = network_health(virtual)
    version = str(summary.get("version", "—")) if ready else ""
    ui.hero(
        version=version,
        model_ready=ready,
        junctions=int(data["junction"].nunique()),
        observations=len(data),
        health=health,
        scope="virtual ops",
    )
    alerts = recent_virtual_alerts(10)
    current_tick = max((int(row.get("tick", 0)) for row in virtual), default=0)
    active_alerts = recent_virtual_alerts(6)
    ticker_items: list[tuple[str, str]] = []
    seen_alerts: set[tuple[str, str]] = set()
    for row in alerts:
        signature = (str(row["junction"]), str(row["alert"]))
        if signature in seen_alerts:
            continue
        seen_alerts.add(signature)
        ticker_items.append(({"critical": "bad", "warning": "warn"}.get(row["severity"], "info"),
                             f"{row['junction']} · {row['alert']}"))
    ui.ticker(ticker_items[:8])

    flows = [num(row, "vehicles_per_hour") for row in virtual]
    occupancies = [num(row, "road_occupancy_pct") for row in virtual]
    spare = spare_capacity(data, virtual)
    weather_values = [weather_index(row) for row in virtual]
    worst_weather = max(weather_values, key=lambda item: float(item["value"])) if weather_values else None
    rmse = summary.get("rmse") if ready else None
    ui.kpi_grid([
        {"label": "Corridors", "value": f"{data['junction'].nunique()}", "sub": "distinct junction profiles", "tone": "info"},
        {"label": "Source records", "value": f"{len(data):,}", "sub": f"latest {pd.Timestamp(newest['timestamp']).strftime('%d %b %Y %H:%M')}", "tone": "violet"},
        {"label": "Fleet flow", "value": f"{sum(flows) / len(flows):.0f} veh/h" if flows else "—", "sub": "mean of virtual junctions", "tone": "good"},
        {"label": "Peak occupancy", "value": f"{max(occupancies):.0f}%" if occupancies else "—", "sub": "simulated sensor load", "tone": "warn" if occupancies and max(occupancies) >= 62 else "good"},
        {"label": "Spare capacity", "value": f"{spare:.0f}%" if spare is not None else "—", "sub": "headroom at the tightest junction", "tone": "bad" if spare is not None and spare < 15 else ("warn" if spare is not None and spare < 30 else "good")},
        {"label": "Weather index", "value": f"{float(worst_weather['value']):.0f}" if worst_weather else "—", "sub": str(worst_weather["band"]) if worst_weather else "no readings", "tone": str(worst_weather["tone"]) if worst_weather else "info"},
        {"label": "Champion", "value": f"v{version}" if ready else "not trained", "sub": f"holdout RMSE {rmse:.2f}" if isinstance(rmse, (int, float)) else "train in MLOps & data", "tone": "info"},
        {"label": "Network health", "value": health, "sub": "sensor + link status", "tone": "good" if health == "Good" else ("warn" if health == "Warning" else "bad")},
    ])
    if hours_old > 24:
        st.info(f"Training history spans through {pd.Timestamp(newest['timestamp']).strftime('%d %b %Y %H:%M')}. The panels below use reproducible virtual telemetry derived from that history.")

    ui.section("LIVE · CORRIDOR PULSE", "Virtual junction fleet")
    if virtual:
        ui.junction_cards(virtual, virtual_history())
    else:
        st.info("Generate a virtual tick in Simulation lab to populate the fleet.")

    left, right = st.columns([1.7, 1], gap="large")
    with left:
        ui.section("BASELINE · 4 JUNCTIONS", "Observed traffic history")
        daily = data.groupby([pd.Grouper(key="timestamp", freq="D"), "junction"], as_index=False)["vehicles"].mean()
        fig = px.line(daily, x="timestamp", y="vehicles", color="junction",
                      labels={"vehicles": "Vehicles per hour", "timestamp": "", "junction": "Junction"})
        fig.update_traces(line=dict(width=1.7))
        st.plotly_chart(ui.style_fig(fig, 330), width="stretch", config={"displayModeBar": False})
    with right:
        ui.section("SIMULATED · THRESHOLDS", "Operations alerts")
        if not active_alerts:
            st.success(f"No simulated alerts recorded yet. Current tick: {current_tick}.")
        for alert in active_alerts[:4]:
            message = f"**{alert['severity'].upper()} · {alert['junction']} · tick {alert['tick']}**\n\n{alert['alert']}"
            if alert["severity"] == "critical":
                st.error(message)
            elif alert["severity"] == "warning":
                st.warning(message)
            else:
                st.info(message)

    plans = corridor_plans(virtual)
    if plans:
        ui.section("ADVISORY · RULE-BASED", "Next best actions")
        ui.advice_list([
            {"label": f"{plan['name']} · {plan['score']}%", "detail": plan["detail"], "tone": plan["tone"]}
            for plan in plans[:3]
        ])
        st.caption("Scores blend simulated occupancy, speed, weather, sensor health, and light. Rule-based guidance, not a trained classifier.")

    ui.section("SOURCE · LATEST ROW", "Junction baseline")
    baseline = latest_by_junction(data).copy()
    baseline["Last source observation"] = pd.to_datetime(baseline["timestamp"]).dt.strftime("%d %b %Y %H:%M")
    baseline = baseline.rename(columns={"junction": "Junction", "vehicles": "Vehicles/hour"})
    st.table(baseline[["Junction", "Last source observation", "Vehicles/hour"]], hide_index=True)
    st.caption("The source data has no weather or network fields; those values are generated by the simulator and labeled virtual.")


def render_live(data: pd.DataFrame, junctions: list[str]) -> None:
    ui.page_head("MODULE 03 · TELEMETRY", "Live sensor monitoring",
                 "Every reading is generated in software from historical per-junction traffic patterns. Advance ticks in the Simulation lab.")
    junction = junction_select("Junction", junctions, "live_junction")
    rows = [row for row in recent_virtual_readings(2000) if row["junction"] == junction]
    if not rows:
        st.info("Generate a virtual sensor tick in Simulation lab to start the feed.")
        return
    frame = pd.DataFrame([
        {"tick": row.get("tick"), "recorded_at": row.get("recorded_at"),
         "vehicles_per_hour": row.get("vehicles_per_hour"), "average_speed_kmh": row.get("average_speed_kmh"),
         "road_occupancy_pct": row.get("road_occupancy_pct"), "temperature_c": row.get("temperature_c"),
         "humidity_pct": row.get("humidity_pct"), "rain_probability_pct": row.get("rain_probability_pct"),
         "wind_speed_kmph": row.get("wind_speed_kmph"), "visibility_km": row.get("visibility_km"),
         "light_intensity_lux": row.get("light_intensity_lux")}
        for row in rows
    ]).sort_values("tick")
    last = frame.iloc[-1].to_dict()
    weather = weather_index(last)
    ui.kpi_grid([
        {"label": "Virtual flow", "value": f"{num(last, 'vehicles_per_hour'):.0f} veh/h", "sub": "sensor reading", "tone": "info"},
        {"label": "Average speed", "value": f"{num(last, 'average_speed_kmh'):.1f} km/h", "sub": "derived from occupancy", "tone": "good"},
        {"label": "Occupancy", "value": f"{num(last, 'road_occupancy_pct'):.1f}%", "sub": "of estimated capacity", "tone": "warn" if num(last, "road_occupancy_pct") >= 62 else "good"},
        {"label": "Temperature / RH", "value": f"{num(last, 'temperature_c'):.1f} °C", "sub": f"{num(last, 'humidity_pct'):.0f}% relative humidity", "tone": "violet"},
        {"label": "Rain probability", "value": f"{num(last, 'rain_probability_pct'):.0f}%", "sub": "synthetic weather", "tone": "info"},
        {"label": "Wind / visibility", "value": f"{num(last, 'wind_speed_kmph'):.0f} km/h", "sub": f"{num(last, 'visibility_km', 16):.1f} km visibility", "tone": "warn" if num(last, "wind_speed_kmph") >= 50 else "info"},
        {"label": "Weather index", "value": f"{float(weather['value']):.0f}", "sub": str(weather["band"]), "tone": str(weather["tone"])},
        {"label": "Light intensity", "value": f"{num(last, 'light_intensity_lux'):,.0f} lux", "sub": "ambient estimate", "tone": "warn"},
    ])
    latest_row = next((row for row in current_virtual_readings() if row["junction"] == junction), {})
    if str(latest_row.get("sensor_status", "online")) != "online":
        st.error(f"Virtual sensor status: {latest_row.get('sensor_status')} on tick {latest_row.get('tick')}. The edge cache keeps forecasting from the last good reading.")
    ui.section("SERIES · FLOW", "Throughput over virtual ticks")
    st.plotly_chart(ui.style_fig(px.line(frame, x="tick", y="vehicles_per_hour", markers=True,
                                         labels={"tick": "Virtual tick", "vehicles_per_hour": "Vehicles per hour"}), 300),
                    width="stretch", config={"displayModeBar": False})
    left, right = st.columns(2, gap="large")
    with left:
        ui.section("SERIES · ROAD", "Speed and occupancy")
        st.plotly_chart(ui.style_fig(px.line(frame, x="tick", y=["average_speed_kmh", "road_occupancy_pct"], markers=True,
                                             labels={"tick": "Virtual tick", "value": "Value", "variable": "Sensor"}), 300),
                        width="stretch", config={"displayModeBar": False})
    with right:
        ui.section("SERIES · WEATHER", "Synthetic environment")
        st.plotly_chart(ui.style_fig(px.line(frame, x="tick", y=["temperature_c", "humidity_pct", "rain_probability_pct"], markers=True,
                                             labels={"tick": "Virtual tick", "value": "Value", "variable": "Metric"}), 300),
                        width="stretch", config={"displayModeBar": False})
    st.caption(f"Virtual tick {int(num(last, 'tick', 0))} · Last updated {pd.Timestamp(str(last['recorded_at'])).strftime('%d %b %Y, %H:%M:%S %Z')}")


def render_weather(junctions: list[str]) -> None:
    ui.page_head("MODULE 04 · WEATHER", "Weather monitoring",
                 "A simulated weather layer for each corridor: temperature, humidity, rain probability, wind, and visibility with rule-based suggestions.")
    junction = junction_select("Junction", junctions, "weather_junction")
    rows = [row for row in recent_virtual_readings(2000) if row["junction"] == junction]
    if not rows:
        st.info("Generate virtual telemetry in Simulation lab to build a weather history.")
        return
    frame = pd.DataFrame([
        {"tick": row.get("tick"), "recorded_at": row.get("recorded_at"),
         "temperature_c": row.get("temperature_c"), "humidity_pct": row.get("humidity_pct"),
         "rain_probability_pct": row.get("rain_probability_pct"), "wind_speed_kmph": row.get("wind_speed_kmph"),
         "visibility_km": row.get("visibility_km"), "water_on_road_pct": row.get("water_on_road_pct")}
        for row in rows
    ]).sort_values("tick")
    frame = frame.dropna(subset=["wind_speed_kmph", "visibility_km"])
    if frame.empty:
        st.info("Generate a virtual tick to record wind and visibility history.")
        return
    last = frame.iloc[-1].to_dict()
    weather = weather_index(last)
    ui.kpi_grid([
        {"label": "Temperature", "value": f"{num(last, 'temperature_c'):.1f} °C", "sub": "simulated road-side sensor", "tone": "violet"},
        {"label": "Humidity", "value": f"{num(last, 'humidity_pct'):.0f}%", "sub": "relative humidity", "tone": "info"},
        {"label": "Rain probability", "value": f"{num(last, 'rain_probability_pct'):.0f}%", "sub": "synthetic forecast", "tone": "warn" if num(last, "rain_probability_pct") >= 65 else "good"},
        {"label": "Wind speed", "value": f"{num(last, 'wind_speed_kmph'):.0f} km/h", "sub": "simulated gusts", "tone": "bad" if num(last, "wind_speed_kmph") >= 55 else "info"},
        {"label": "Visibility", "value": f"{num(last, 'visibility_km', 16):.1f} km", "sub": "driver sight distance", "tone": "warn" if num(last, "visibility_km", 16) <= 7 else "good"},
        {"label": "Weather index", "value": f"{float(weather['value']):.0f}", "sub": str(weather["band"]), "tone": str(weather["tone"])},
    ])
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        ui.section("MODULE 04 · TRENDS", "Weather over virtual ticks")
        st.plotly_chart(ui.style_fig(px.line(frame, x="tick", y=["temperature_c", "humidity_pct", "rain_probability_pct"],
                                             markers=True, labels={"tick": "Virtual tick", "value": "Reading", "variable": "Metric"}), 320),
                        width="stretch", config={"displayModeBar": False})
        st.plotly_chart(ui.style_fig(px.line(frame, x="tick", y=["wind_speed_kmph", "visibility_km"],
                                             markers=True, labels={"tick": "Virtual tick", "value": "Reading", "variable": "Metric"}), 260),
                        width="stretch", config={"displayModeBar": False})
    with right:
        ui.section("MODULE 04 · GUIDANCE", "Operator suggestions")
        ui.advice_list(weather_suggestions(last))
        st.caption("Suggestions are rule-based on simulated readings. They do not change signal or device state automatically.")

    ui.section("MODULE 04 · FLEET", "Latest weather by junction")
    fleet = []
    for row in current_virtual_readings():
        index = weather_index(row)
        fleet.append({
            "Junction": row["junction"], "Temp °C": num(row, "temperature_c"),
            "Humidity %": num(row, "humidity_pct"), "Rain %": num(row, "rain_probability_pct"),
            "Wind km/h": num(row, "wind_speed_kmph"), "Visibility km": num(row, "visibility_km", 16),
            "Index": float(index["value"]), "Band": index["band"],
        })
    st.table(pd.DataFrame(fleet), hide_index=True)
    st.caption("These values come from the project simulator. The source dataset contains no weather observations.")


def render_alerts(junctions: list[str]) -> None:
    ui.page_head("MODULE 05 · ALERTS", "Alert centre",
                 "Filter simulated threshold alerts by category, junction, and severity, then acknowledge the ones an operator has reviewed.")
    raw = recent_virtual_alerts(300)
    alerts = categorize_alerts(raw)
    if not alerts:
        st.success("No threshold alerts have been generated yet. Advance a tick in Simulation lab.")
        return
    current_tick = max(int(row.get("tick", 0)) for row in alerts)
    unacknowledged = [row for row in alerts if not row.get("acknowledged")]
    critical = [row for row in alerts if str(row.get("severity")) == "critical"]
    ui.kpi_grid([
        {"label": "Alerts retained", "value": str(len(alerts)), "sub": "latest 300 rows", "tone": "info"},
        {"label": "Current tick", "value": str(current_tick), "sub": f"{sum(1 for row in alerts if int(row.get('tick', 0)) == current_tick)} alert(s) on this tick", "tone": "violet"},
        {"label": "Unacknowledged", "value": str(len(unacknowledged)), "sub": "awaiting operator review", "tone": "warn" if unacknowledged else "good"},
        {"label": "Critical", "value": str(len(critical)), "sub": "highest severity", "tone": "bad" if critical else "good"},
    ])
    counts = Counter(str(row["category"]) for row in alerts)
    ui.section("MODULE 05 · TAXONOMY", "Alerts by category")
    ui.category_chips([(name.title(), counts.get(name, 0), tone) for name, tone in CATEGORY_TONE.items()])

    ui.section("MODULE 05 · FILTER", "Review alerts")
    filter_cols = st.columns(4)
    with filter_cols[0]:
        severity = st.selectbox(
            "Severity", ["All severities"] + sorted({str(row["severity"]) for row in alerts}), key="alert_severity")
    with filter_cols[1]:
        category = st.selectbox(
            "Category", ["All categories"] + [name.title() for name in CATEGORY_TONE], key="alert_category")
    with filter_cols[2]:
        junction_choice = st.selectbox("Junction", ["All junctions"] + junctions, key="alert_junction")
    with filter_cols[3]:
        current_only = st.toggle("Current tick only", value=False, key="alert_current_tick")
        pending_only = st.toggle("Unacknowledged only", value=False, key="alert_pending")
    filtered = [
        row for row in alerts
        if (severity == "All severities" or str(row["severity"]) == severity)
        and (category == "All categories" or str(row["category"]).title() == category)
        and (junction_choice == "All junctions" or str(row["junction"]) == junction_choice)
        and (not current_only or int(row.get("tick", 0)) == current_tick)
        and (not pending_only or not row.get("acknowledged"))
    ]
    if not filtered:
        st.info("No alerts match the current filters.")
    else:
        table = pd.DataFrame([{
            "ID": row["id"], "Tick": row["tick"], "Time": pretty_time(row["created_at"]), "Junction": row["junction"],
            "Category": row["category"], "Severity": row["severity"], "Alert": row["alert"],
            "Acknowledged": "yes" if row.get("acknowledged") else "no",
        } for row in filtered])
        st.dataframe(table, width="stretch", hide_index=True)

    ui.section("MODULE 05 · ACKNOWLEDGE", "Operator review")
    open_alerts = [row for row in alerts if not row.get("acknowledged")]
    if not open_alerts:
        st.success("Every retained alert has been acknowledged.")
    else:
        options = {int(row["id"]): f"#{row['id']} · {row['junction']} · {row['severity']} · {str(row['alert'])[:64]}" for row in open_alerts}
        selected = st.selectbox("Alert to acknowledge", list(options), format_func=lambda key: options[key], key="ack_alert")
        note = st.text_input("Review note (optional)", placeholder="What the operator checked", key="ack_note")
        if st.button("Acknowledge alert", type="primary"):
            if acknowledge_alert(int(selected)):
                log_operator_action("Network-wide", "Acknowledge alert",
                                    f"Alert #{selected}. {note}".strip())
                push_flash(f"Alert #{selected} acknowledged.")
                st.rerun()
            else:
                st.error("That alert could not be found.")
    st.caption("Acknowledgement is stored in the local SQLite history; it does not clear the underlying simulated condition.")


def render_map(virtual: list[dict[str, object]]) -> None:
    ui.page_head("MODULE 06 · MAP", "Junction map",
                 "Schematic junction zones built from simulated sensor state — no coordinates, GIS service, or physical devices required.")
    if not virtual:
        st.info("Generate a virtual sensor tick in Simulation lab to populate the zones.")
        return
    levels = []
    for row in virtual:
        occupancy = num(row, "road_occupancy_pct")
        online = str(row.get("sensor_status", "")).lower().startswith("online")
        levels.append("Critical" if (not online or occupancy >= 78) else ("Needs attention" if occupancy >= 62 else "Healthy"))
    tones = {"Healthy": "good", "Needs attention": "warn", "Critical": "bad"}
    counts = Counter(levels)
    ui.kpi_grid([
        {"label": "Zones", "value": str(len(virtual)), "sub": "simulated junction sensors", "tone": "info"},
        {"label": "Healthy", "value": str(counts.get("Healthy", 0)), "sub": "occupancy below 62%", "tone": "good"},
        {"label": "Needs attention", "value": str(counts.get("Needs attention", 0)), "sub": "occupancy 62–78%", "tone": "warn"},
        {"label": "Critical", "value": str(counts.get("Critical", 0)), "sub": "over 78% or offline", "tone": "bad"},
    ])
    ui.junction_map(virtual)
    ui.category_chips([(name, count, tones[name]) for name, count in counts.items()])
    st.divider()
    latest = pd.DataFrame(virtual).copy()
    latest["status"] = levels
    latest["spare_headroom_%"] = [round(max(0.0, 100 - num(row, "road_occupancy_pct")), 0) for row in virtual]
    latest["recorded_at"] = latest["recorded_at"].map(pretty_time)
    st.dataframe(latest[["junction", "recorded_at", "vehicles_per_hour", "average_speed_kmh", "road_occupancy_pct", "status", "spare_headroom_%"]].rename(columns={
        "junction": "Virtual junction", "recorded_at": "Simulation time", "vehicles_per_hour": "Flow veh/h",
        "average_speed_kmh": "Speed km/h", "road_occupancy_pct": "Occupancy %",
    }), width="stretch", hide_index=True)


def render_horizon(data: pd.DataFrame, junction: str, ready: bool, summary: dict[str, object]) -> None:
    ui.section("MODULE 07 · OUTLOOK", "48-hour recursive demand curve")
    st.caption("The champion rolls forward hour by hour; each estimate becomes the input for the next step. The shaded band is ± one holdout RMSE.")
    if not ready:
        st.info("Train and promote a champion in MLOps & data to unlock the day-ahead outlook.")
        return
    hours = st.select_slider("Horizon", options=[6, 12, 24, 48], value=24,
                             format_func=lambda value: f"{value} hours", key="horizon_hours")
    version = str(summary.get("version", ""))
    key = (junction, hours, version, len(data), str(pd.Timestamp(data["timestamp"].max())))
    cached = st.session_state.get("horizon")
    if cached and cached.get("key") == key:
        result = cached["result"]
    else:
        result = None
    if st.button("Project demand curve", type="primary", width="stretch"):
        with st.spinner("Rolling the champion forward…"):
            try:
                result = predict_horizon(data, junction, hours=hours, offline=not st.session_state.get("sim_cloud_online", True))
                st.session_state["horizon"] = {"key": key, "result": result}
            except Exception as error:
                st.error(f"Outlook unavailable: {error}")
    if not result:
        st.caption("Project the next hours to compare the forecast peak with historical demand.")
        return

    frame = pd.DataFrame(result["points"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    rmse = summary.get("rmse")
    fig = go.Figure()
    if isinstance(rmse, (int, float)):
        upper = frame["predicted_vehicles"] + rmse
        lower = (frame["predicted_vehicles"] - rmse).clip(lower=0)
        fig.add_trace(go.Scatter(x=frame["timestamp"], y=upper, line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=frame["timestamp"], y=lower, fill="tonexty", fillcolor="rgba(34,211,238,.12)",
                                 line=dict(width=0), name="±1 holdout RMSE", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=frame["timestamp"], y=frame["predicted_vehicles"], mode="lines+markers", name="Forecast",
                             line=dict(color="#22d3ee", width=2.4, shape="spline"), marker=dict(size=5)))
    peak = result["peak"]
    fig.add_trace(go.Scatter(x=[pd.Timestamp(peak["timestamp"])], y=[peak["predicted_vehicles"]], mode="markers", name="Peak",
                             marker=dict(symbol="star", size=14, color="#fbbf24", line=dict(color="#0b1a28", width=1))))
    fig.update_layout(hovermode="x unified", yaxis_title="Vehicles per hour")
    st.plotly_chart(ui.style_fig(fig, 340), width="stretch", config={"displayModeBar": False})

    p95 = float(data[data["junction"] == junction]["vehicles"].quantile(.95))
    peak_volume = float(peak["predicted_vehicles"])
    ui.kpi_grid([
        {"label": "Peak window", "value": pd.Timestamp(peak["timestamp"]).strftime("%a %H:%M"), "sub": "highest projected hour", "tone": "warn"},
        {"label": "Peak volume", "value": f"{peak_volume:.0f} veh/h", "sub": f"{peak_volume / max(p95, 1) * 100:.0f}% of historical p95", "tone": "bad" if peak_volume >= p95 else "info"},
        {"label": "Mean projected", "value": f"{result['mean_predicted']:.0f} veh/h", "sub": f"trailing 24h mean {result['baseline_mean_24h']:.0f}", "tone": "violet"},
        {"label": "Band width", "value": f"±{rmse:.1f}" if isinstance(rmse, (int, float)) else "n/a", "sub": "holdout RMSE", "tone": "good"},
    ])
    if peak_volume >= p95:
        st.warning("A projected hour reaches the historical 95th percentile. Plan a simulated signal adjustment and review diversion options.")
    elif peak_volume > result["baseline_mean_24h"] * 1.15:
        st.info("Demand is projected above the trailing 24-hour mean but stays within its normal peak envelope. Continue routine monitoring.")
    else:
        st.success("Projected demand stays within the recent operating envelope.")
    st.caption(f"{result['hours']} recursive steps · generated {pd.Timestamp(result['generated_at']).strftime('%d %b %Y %H:%M UTC')} · {'local edge cache' if result['offline'] else 'MLflow champion'}")


def render_forecast(data: pd.DataFrame, junctions: list[str], ready: bool, summary: dict[str, object]) -> None:
    ui.page_head("MODULE 07 · FORECAST", "Demand forecast & outlook",
                 "Estimate next-hour demand with the champion, project the day ahead, and preview the signal plan that the outlook implies.")
    junction = junction_select("Junction", junctions, "forecast_junction")
    latest = data[data["junction"] == junction].iloc[-1]
    virtual = next((row for row in current_virtual_readings() if row["junction"] == junction), {})
    left, right = st.columns([1, 1.12], gap="large")
    with left, st.container(border=True):
        st.markdown("##### Signal inputs")
        vehicles = st.number_input("Current virtual vehicles / hour", 0.0, value=float(virtual.get("vehicles_per_hour", latest["vehicles"])), step=10.0)
        temperature = st.number_input("Virtual temperature °C", value=float(virtual.get("temperature_c", latest["temperature_c"])), step=1.0)
        event = st.toggle("Simulate special event", value=bool(latest["is_event"]))
        offline = not st.session_state.get("sim_cloud_online", True)
        st.caption("Offline mode uses the host-local champion cache." if offline else "Online mode uses the MLflow champion alias.")
        run = st.button("Generate next-hour forecast", type="primary", width="stretch")
    with right:
        if run:
            try:
                result = make_forecast(data, junction, vehicles, temperature, int(event), offline=offline)
                baseline = data[data["junction"] == junction].tail(24)["vehicles"].mean()
                delta_pct = (result["predicted_vehicles"] - baseline) / max(baseline, 1) * 100
                ui.kpi_grid([
                    {"label": "Next-hour forecast", "value": f"{result['predicted_vehicles']:.0f} veh/h",
                     "sub": pd.Timestamp(result["forecast_for"]).strftime("%d %b %Y %H:%M"), "tone": "info"},
                    {"label": "Vs trailing 24h", "value": f"{delta_pct:+.1f}%", "sub": f"baseline {baseline:.0f} veh/h",
                     "tone": "warn" if delta_pct >= 15 else "good"},
                    {"label": "Inference", "value": f"{st.session_state['last_inference']['latency_ms']} ms",
                     "sub": "local edge cache" if offline else "MLflow champion", "tone": "violet"},
                ])
                if delta_pct >= 15:
                    st.warning("Demand is materially above the recent baseline. Review signal timing and diversion options on Smart control.")
                else:
                    st.success("Forecast is near or below the recent baseline. Continue routine monitoring.")
            except Exception as error:
                st.error(f"Forecast unavailable: {error}. Train a champion model in MLOps & data.")
        elif st.session_state.get("last_forecast"):
            result = st.session_state["last_forecast"]
            ui.kpi_grid([
                {"label": "Last forecast", "value": f"{result['predicted_vehicles']:.0f} veh/h", "sub": result["junction"], "tone": "info"},
                {"label": "Forecast time", "value": pd.Timestamp(result["forecast_for"]).strftime("%d %b %H:%M"), "sub": "model estimate", "tone": "violet"},
            ])
        else:
            st.info("Run a forecast to see the estimate and an implied signal plan.")

        plan = signal_plan(num(virtual, "road_occupancy_pct", 45), weather_value=float(weather_index(virtual)["value"]),
                           event=bool(event))
        st.markdown("**Implied automatic plan**")
        ui.kpi_grid([
            {"label": "Green phase", "value": f"{plan['green_seconds']} s", "sub": str(plan["phase"]), "tone": str(plan["tone"])},
            {"label": "Cycle", "value": f"{plan['cycle_seconds']} s", "sub": "green + clearance", "tone": "info"},
            {"label": "Weather index", "value": f"{float(weather_index(virtual)['value']):.0f}", "sub": str(weather_index(virtual)["band"]), "tone": str(weather_index(virtual)["tone"])},
        ])
        st.caption(f"{plan['reason'].capitalize()}. Apply or override it on the Smart control page.")

    render_horizon(data, junction, ready, summary)


def render_control(data: pd.DataFrame, junctions: list[str], virtual: list[dict[str, object]]) -> None:
    ui.page_head("MODULE 08 · CONTROL", "Smart control",
                 "Review the rule-based signal plan, apply per-device recommendations, and operate the simulated device rack.")
    junction = junction_select("Junction", junctions, "control_junction")
    reading = next((row for row in virtual if row["junction"] == junction), {})
    occupancy = num(reading, "road_occupancy_pct", 45.0)
    weather = weather_index(reading) if reading else {"value": 0.0, "band": "Clear", "tone": "good"}

    ui.section("MODULE 08 · SIGNAL", "Signal plan")
    mode = st.radio("Control mode", ["Automatic", "Manual"], horizontal=True, key="signal_mode")
    manual_green = st.slider("Manual green phase (seconds)", 15, 90, 45, key="manual_green") if mode == "Manual" else None
    plan = signal_plan(occupancy, weather_value=float(weather["value"]), override=manual_green)
    ui.signal_card(occupancy, int(plan["green_seconds"]), str(plan["mode"]))
    ui.kpi_grid([
        {"label": "Green phase", "value": f"{plan['green_seconds']} s", "sub": f"derived {plan['derived_seconds']} s", "tone": str(plan["tone"])},
        {"label": "Cycle", "value": f"{plan['cycle_seconds']} s", "sub": "green + clearance", "tone": "info"},
        {"label": "Phase", "value": str(plan["phase"]).title(), "sub": "occupancy band", "tone": str(plan["tone"])},
        {"label": "Weather index", "value": f"{float(weather['value']):.0f}", "sub": str(weather["band"]), "tone": str(weather["tone"])},
    ])
    st.caption(f"{plan['reason'].capitalize()}. No signal hardware is contacted; this is a software plan.")

    devices = list_devices()
    recommendations = device_recommendation(devices, virtual)
    pending = [
        device for device in devices
        if device["slug"] in recommendations
        and (recommendations[device["slug"]]["state"] != device["state"]
             or recommendations[device["slug"]]["mode"] != device["mode"])
    ]
    ui.section("MODULE 08 · DEVICES", "Device rack")
    ui.device_grid(devices, recommendations)
    left, right = st.columns([1, 1], gap="large")
    with left:
        if pending:
            st.warning(f"{len(pending)} device(s) differ from the current recommendation.")
            if st.button(f"Apply all {len(pending)} recommendations", type="primary", width="stretch"):
                for device in pending:
                    rec = recommendations[device["slug"]]
                    set_device_state(device["slug"], state=rec["state"], mode=rec["mode"],
                                     updated_by=st.session_state["operator_id"])
                    log_operator_action("Network-wide", f"Apply device plan · {device['name']}",
                                        f"{device['state']}/{device['mode']} → {rec['state']}/{rec['mode']}. {rec['reason']}")
                push_flash(f"Applied recommendations to {len(pending)} device(s).")
                st.rerun()
        else:
            st.success("Every device already matches the current recommendation.")
    with right:
        st.caption("Recommendations follow fixed rules on simulated occupancy, weather, light, and patrol priority. Applying one records an operator action.")

    ui.section("MODULE 08 · OVERRIDE", "Per-device control")
    columns = st.columns(3)
    for index, device in enumerate(devices):
        rec = recommendations.get(device["slug"], {})
        with columns[index % 3]:
            with st.container(border=True):
                st.markdown(f"**{device['name']}**")
                st.caption(f"{device['kind']} · last changed by {device['updated_by']}")
                if rec:
                    st.caption(f"Recommended: {rec['state']} · {rec['mode']} — {rec['reason']}")
                state = st.selectbox("State", ["on", "off"],
                                     index=0 if str(device["state"]) == "on" else 1,
                                     key=f"dev_state_{device['slug']}")
                device_mode = st.selectbox("Mode", ["auto", "manual"],
                                           index=0 if str(device["mode"]) == "auto" else 1,
                                           key=f"dev_mode_{device['slug']}")
                if st.button("Update device", key=f"dev_apply_{device['slug']}", width="stretch"):
                    set_device_state(device["slug"], state=state, mode=device_mode,
                                     updated_by=st.session_state["operator_id"])
                    log_operator_action("Network-wide", f"Manual device control · {device['name']}",
                                        f"Set state={state}, mode={device_mode}.")
                    push_flash(f"{device['name']} updated to {state} · {device_mode}.")
                    st.rerun()

    ui.section("MODULE 08 · LOG", "Record an operator action")
    with st.form("operator_action"):
        action = st.selectbox("Advisory action", ["Apply simulated signal phase", "Review signal timing", "Prepare diversion notice", "Dispatch virtual field check", "No action; monitor"])
        note = st.text_input("Operator note", placeholder="Reason or follow-up")
        submitted = st.form_submit_button("Save to action history")
        if submitted:
            action_note = f"Mode={plan['mode']}; green={plan['green_seconds']}s. {note}".strip()
            log_operator_action(junction, action, action_note)
            push_flash("Virtual action recorded in the software simulator.")
            st.rerun()
    actions = pd.DataFrame(recent_operator_actions(10))
    if not actions.empty:
        st.dataframe(
            tidy_table(actions).rename(columns={"id": "ID", "created_at": "Recorded", "junction": "Junction",
                                                "action": "Action", "note": "Note"}),
            width="stretch", hide_index=True,
        )


def render_incident_vision(data: pd.DataFrame, junctions: list[str]) -> None:
    ui.page_head("MODULE 09 · VISION", "Incident vision",
                 "Upload an incident frame for heuristic scene triage, then pair it with a model-based congestion estimate and an operator report.")
    left, right = st.columns([1, 1], gap="large")
    payload = None
    image = None
    with left:
        image = st.file_uploader("Upload incident image", type=["png", "jpg", "jpeg"], key="incident_upload")
        camera = None
        with st.expander("Or capture from this device's camera"):
            camera = st.camera_input("Capture incident image", key="incident_camera")
        source = image or camera
        if source is not None:
            payload = source.getvalue()
            st.image(payload, caption=getattr(source, "name", "Camera capture"), width="stretch")
    with right:
        if payload is None:
            st.info("Add a frame to run scene triage. The uploaded image is processed in memory; only its file name and triage summary are stored.")
        else:
            try:
                triage = triage_image(payload)
            except ValueError as error:
                st.error(str(error))
                triage = None
            if triage:
                ui.kpi_grid([
                    {"label": "Scene risk", "value": str(triage["risk"]), "sub": "heuristic band", "tone": str(triage["tone"])},
                    {"label": "Triage index", "value": f"{float(triage['score']):.0f}/100", "sub": "exposure + detail + clutter", "tone": str(triage["tone"])},
                    {"label": "Signal confidence", "value": f"{float(triage['confidence']):.0f}%", "sub": "distance from the borderline", "tone": "info"},
                    {"label": "Frame", "value": f"{int(triage['width'])}×{int(triage['height'])}", "sub": f"signature {triage['signature']}", "tone": "violet"},
                ])
                ui.section("MODULE 09 · METRICS", "Frame measurements")
                metrics = triage["metrics"]
                ui.score_bars([
                    {"name": "Brightness", "score": float(metrics["brightness"]) * 100, "tone": "info"},
                    {"name": "Contrast", "score": float(metrics["contrast"]) * 100, "tone": "violet"},
                    {"name": "Edge density", "score": float(metrics["edge_density"]) * 100, "tone": "good"},
                    {"name": "Dark pixels", "score": float(metrics["dark_share"]) * 100, "tone": "warn"},
                ])
                st.caption("Scores below are rule-based pixel measurements, not a trained classifier and not a calibrated probability.")
                ui.section("MODULE 09 · FINDINGS", "Automated observations")
                ui.advice_list([{"label": "Observation", "detail": finding, "tone": str(triage["tone"])} for finding in triage["findings"]])
                ui.section("MODULE 09 · CHECKLIST", "Before filing")
                ui.advice_list([{"label": f"Step {index + 1}", "detail": step, "tone": "info"} for index, step in enumerate(triage["checklist"])])

    ui.section("MODULE 09 · CONGESTION", "Model-based risk estimate")
    risk_junction = junction_select("Run the traffic-risk estimate for", junctions, "incident_risk_junction")
    if st.button("Estimate next-hour congestion risk", type="primary", width="stretch"):
        try:
            virtual_flow = next((r["vehicles_per_hour"] for r in current_virtual_readings() if r["junction"] == risk_junction), None)
            prediction = predict_next_hour(data, risk_junction, vehicles=virtual_flow,
                                           offline=not st.session_state.get("sim_cloud_online", True))
            p95 = float(data[data["junction"] == risk_junction]["vehicles"].quantile(.95))
            ratio = float(prediction["predicted_vehicles"]) / max(p95, 1) * 100
            label = "Critical" if ratio >= 100 else ("Elevated" if ratio >= 80 else "Within range")
            tone = "bad" if label == "Critical" else ("warn" if label == "Elevated" else "good")
            ui.kpi_grid([
                {"label": "Congestion risk", "value": label, "sub": f"{ratio:.1f}% of historical p95", "tone": tone},
                {"label": "Predicted flow", "value": f"{prediction['predicted_vehicles']:.0f} veh/h",
                 "sub": pd.Timestamp(prediction["forecast_for"]).strftime("%d %b %Y %H:%M"), "tone": "info"},
            ])
            recommendation = ("Critical: review the simulated signal phase and incident/diversion plan immediately." if label == "Critical" else
                              "Elevated: prepare a timing adjustment and watch the next virtual sensor tick." if label == "Elevated" else
                              "Within range: keep the automatic plan and continue monitoring.")
            st.info(f"Recommended action: {recommendation}")
            st.caption("Threshold score from the flow model, not a calibrated probability or image classifier.")
        except Exception as error:
            st.warning(f"Risk inference unavailable: {error}")

    ui.section("MODULE 09 · REPORT", "File an incident report")
    with st.form("incident_form"):
        cols = st.columns(2)
        with cols[0]:
            junction = junction_select("Junction", junctions, "incident_junction")
            category = st.selectbox("Incident type", ["Congestion", "Collision (operator reported)", "Road closure", "Vehicle breakdown", "Weather hazard", "Other"])
        with cols[1]:
            severity = st.select_slider("Operator-assessed severity", ["Low", "Moderate", "High", "Critical"], value="Moderate")
        summary = st.text_area("Description", placeholder="Describe what was observed; do not include sensitive personal details.")
        submitted = st.form_submit_button("Save incident report", type="primary")
        if submitted:
            if not summary.strip():
                st.error("Add a short description before saving.")
            else:
                image_name = getattr(source, "name", "camera-capture.jpg") if payload is not None else None
                triage_summary = None
                if payload is not None:
                    try:
                        detail = triage_image(payload)
                        triage_summary = {"risk": detail["risk"], "score": detail["score"],
                                          "confidence": detail["confidence"], "signature": detail["signature"]}
                    except ValueError:
                        triage_summary = None
                log_incident(junction, category, severity, summary.strip(), image_name, triage=triage_summary)
                push_flash("Incident scenario saved to the local history.")
                st.rerun()

    ui.section("MODULE 09 · HISTORY", "Incident reports")
    incidents = pd.DataFrame(recent_incidents(200))
    if incidents.empty:
        st.info("No operator incident reports yet.")
    else:
        triage_rows = incidents.pop("triage")
        incidents["scene risk"] = [row.get("risk", "—") if isinstance(row, dict) else "—" for row in triage_rows]
        incidents["triage score"] = [row.get("score", "—") if isinstance(row, dict) else "—" for row in triage_rows]
        st.dataframe(
            tidy_table(incidents).rename(columns={
                "id": "ID", "created_at": "Recorded", "junction": "Junction",
                "image_name": "Frame",
            }),
            width="stretch", hide_index=True,
        )


def render_corridor(virtual: list[dict[str, object]]) -> None:
    ui.page_head("MODULE 10 · ADVISORY", "Corridor recommendation",
                 "Rank intervention plans by how strongly the current simulated readings support them — a rule-based analogue of a crop-suitability list.")
    if not virtual:
        st.info("Generate virtual telemetry in Simulation lab to score the corridor plans.")
        return
    plans = corridor_plans(virtual)
    top = plans[0]
    peak = max(num(row, "road_occupancy_pct") for row in virtual)
    weather = max((weather_index(row) for row in virtual), key=lambda item: float(item["value"]))
    ui.kpi_grid([
        {"label": "Top plan", "value": str(top["name"]), "sub": f"{top['score']}% support", "tone": "good"},
        {"label": "Second plan", "value": str(plans[1]["name"]), "sub": f"{plans[1]['score']}% support", "tone": "info"},
        {"label": "Peak occupancy", "value": f"{peak:.0f}%", "sub": "busiest virtual junction", "tone": "warn" if peak >= 62 else "good"},
        {"label": "Weather band", "value": str(weather["band"]), "sub": f"index {float(weather['value']):.0f}/100", "tone": str(weather["tone"])},
    ])
    ui.section("MODULE 10 · RANKING", "Plan suitability")
    ui.score_bars([{"name": plan["name"], "score": plan["score"], "tone": plan["tone"], "note": plan["detail"]} for plan in plans])
    st.caption("Scores are deterministic 0-99 support values from fixed weights; they are not probabilities and not model output.")
    ui.section("MODULE 10 · TOP PLANS", "What the readings support")
    ui.advice_list([{"label": f"{plan['name']} · {plan['score']}%", "detail": plan["detail"], "tone": plan["tone"]} for plan in plans[:3]])

    ui.section("MODULE 10 · PER JUNCTION", "Junction-level view")
    rows = []
    for row in virtual:
        reading_weather = weather_index(row)
        plan = signal_plan(num(row, "road_occupancy_pct"), weather_value=float(reading_weather["value"]))
        rows.append({
            "Junction": row["junction"], "Flow veh/h": num(row, "vehicles_per_hour"),
            "Occupancy %": num(row, "road_occupancy_pct"), "Speed km/h": num(row, "average_speed_kmh"),
            "Weather band": reading_weather["band"], "Suggested green s": plan["green_seconds"],
            "Phase": plan["phase"],
        })
    st.table(pd.DataFrame(rows), hide_index=True)

    if st.button(f"Adopt “{top['name']}” and log the advisory", type="primary"):
        log_operator_action("Network-wide", f"Adopt plan · {top['name']}",
                            f"Support {top['score']}%. {top['detail']} Peak occupancy {peak:.0f}%.")
        push_flash("Advisory recorded in the operator action history.")
        st.rerun()


def assistant_response(question: str, data: pd.DataFrame, junction: str) -> str:
    selected = data[data["junction"] == junction].sort_values("timestamp")
    current = selected.iloc[-1]
    mean24 = selected.tail(24)["vehicles"].mean()
    query = question.lower()
    if any(word in query for word in ("weather", "rain", "wind", "humidity", "visibility", "fog")):
        virtual = next((row for row in current_virtual_readings() if row["junction"] == junction), None)
        if virtual:
            index = weather_index(virtual)
            advice = weather_suggestions(virtual)[0]
            return (f"Virtual weather scenario for {junction}: {num(virtual, 'temperature_c'):.1f} °C, humidity {num(virtual, 'humidity_pct'):.0f}%, "
                    f"rain probability {num(virtual, 'rain_probability_pct'):.0f}%, wind {num(virtual, 'wind_speed_kmph'):.0f} km/h, "
                    f"visibility {num(virtual, 'visibility_km', 16):.1f} km. Weather index {float(index['value']):.0f}/100 ({index['band']}). "
                    f"Suggested: {advice['label']} — {advice['detail']} These are simulated values, not a live forecast.")
        return "The source CSV has no weather fields. Generate a virtual telemetry tick to inspect simulated weather conditions."
    if any(word in query for word in ("device", "signal", "green", "barrier", "lighting", "vms", "control")):
        virtual = next((row for row in current_virtual_readings() if row["junction"] == junction), None)
        occupancy = num(virtual, "road_occupancy_pct", 45)
        index = weather_index(virtual) if virtual else {"value": 0.0}
        plan = signal_plan(occupancy, weather_value=float(index["value"]))
        devices = list_devices()
        recommendations = device_recommendation(devices, current_virtual_readings())
        changes = [f"{device['name']} → {recommendations[device['slug']]['state']}"
                   for device in devices
                   if device["slug"] in recommendations and recommendations[device["slug"]]["state"] != device["state"]]
        change_line = f"Pending device changes: {', '.join(changes)}." if changes else "Every device already matches its recommendation."
        return (f"Signal plan for {junction}: {plan['green_seconds']} s green, {plan['cycle_seconds']} s cycle, phase {plan['phase']} "
                f"({plan['reason']}). {change_line} Open Smart control to apply or override any device.")
    if "alert" in query:
        alerts = categorize_alerts(recent_virtual_alerts(200))
        counts = Counter(str(row["category"]) for row in alerts)
        summary = ", ".join(f"{name} {counts.get(name, 0)}" for name in CATEGORY_TONE)
        open_count = sum(1 for row in alerts if not row.get("acknowledged"))
        return (f"Across the retained virtual alerts: {summary}. {open_count} still need acknowledgement. "
                "Use the Alert centre to filter by category and junction before acknowledging.")
    if any(word in query for word in ("should", "recommend", "advice", "plan", "priority", "attention", "divert")):
        plans = corridor_plans(current_virtual_readings())
        if plans:
            top = plans[:3]
            listing = "; ".join(f"{plan['name']} {plan['score']}%" for plan in top)
            return (f"Rule-based priority for the current simulated tick: {listing}. Start with {top[0]['name']} — {top[0]['detail']} "
                    "Open Corridor advisory to log the choice.")
        return "No virtual telemetry is available yet, so no plan can be scored. Generate a tick in Simulation lab first."
    if any(word in query for word in ("predict", "next hour", "congestion", "traffic", "route", "peak", "outlook")):
        if any(word in query for word in ("congestion", "route", "peak", "outlook")):
            virtual = next((row for row in current_virtual_readings() if row["junction"] == junction), None)
            flow = virtual["vehicles_per_hour"] if virtual else current["vehicles"]
            suffix = f" Virtual occupancy is {num(virtual, 'road_occupancy_pct'):.0f}%." if virtual else ""
            return (f"At {junction}, current {'simulated ' if virtual else ''}flow is {float(flow):.0f} vehicles/hour and the training-history trailing 24-hour mean is "
                    f"{mean24:.0f}.{suffix} Review simulator incidents and compare alternate junctions before changing a virtual signal plan.")
        try:
            prediction = predict_next_hour(data, junction, offline=not st.session_state.get("sim_cloud_online", True))
            return (f"The champion model estimates {prediction['predicted_vehicles']:.0f} vehicles/hour at {prediction['forecast_for']} for {junction}. "
                    "This is a model forecast from the latest historical row. Use Simulation lab to exercise virtual sensor and offline modes.")
        except Exception as error:
            return f"A forecast is unavailable right now ({error}). Train a champion model and provide at least one week of hourly history."
    return (f"I can summarize this project's traffic data for {junction}: latest flow {current['vehicles']:.0f} vehicles/hour, "
            f"trailing 24-hour average {mean24:.0f}. "
            "Ask about traffic, the next-hour forecast, congestion, weather context, devices, alerts, or recommended plans.")


def assistant_greeting(data: pd.DataFrame, junction: str, summary: dict[str, object]) -> str:
    selected = data[data["junction"] == junction].sort_values("timestamp")
    current = selected.iloc[-1]
    mean24 = selected.tail(24)["vehicles"].mean()
    version = summary.get("version")
    model_line = f"Champion model **v{version}** is live." if version else "No champion model is registered yet."
    return (f"**Command deck assistant online.** Context junction **{junction}** — latest observed flow "
            f"**{current['vehicles']:.0f} veh/h**, trailing 24-hour mean **{mean24:.0f} veh/h**. {model_line} "
            "Ask about the next-hour forecast, congestion risk, weather context, devices, alerts, or which plan the readings support.")


def render_assistant(data: pd.DataFrame, junctions: list[str], summary: dict[str, object]) -> None:
    ui.page_head("MODULE 11 · ASSISTANT", "Traffic AI assistant",
                 "Answers are grounded in the local traffic dataset, the registered forecast model, and the current virtual telemetry.")
    junction = junction_select("Context junction", junctions, "assistant_junction")
    if "assistant_messages" not in st.session_state:
        st.session_state["assistant_messages"] = []
    suggestions = ["What should I do next?", "Will traffic rise in the next hour?", "Which device should change?"]
    pending = None
    columns = st.columns(3)
    for column, text in zip(columns, suggestions):
        if column.button(text, width="stretch"):
            pending = text
    messages = st.session_state["assistant_messages"]
    if not messages:
        with st.chat_message("assistant"):
            st.markdown(assistant_greeting(data, junction, summary))
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    question = st.chat_input("Ask about traffic, forecasts, devices, alerts, or weather context")
    question = pending or question
    if question:
        answer = assistant_response(question, data, junction)
        st.session_state["assistant_messages"].extend([
            {"role": "user", "content": question}, {"role": "assistant", "content": answer}
        ])
        st.rerun()
    if st.session_state["assistant_messages"] and st.button("Clear conversation"):
        st.session_state["assistant_messages"] = []
        st.rerun()


def render_edge(data: pd.DataFrame, junctions: list[str], ready: bool, summary: dict[str, object]) -> None:
    ui.page_head("MODULE 12 · EDGE", "Edge AI & 5G network",
                 "Prove local inference from the host cache, inspect the simulated 5G link, and see where each junction's telemetry would run.")
    virtual = current_virtual_readings()
    cache_ready = (EDGE_CACHE / "champion.json").exists()
    cached_files = [path for path in EDGE_CACHE.glob("**/*") if path.is_file()] if EDGE_CACHE.exists() else []
    ui.kpi_grid([
        {"label": "MLflow champion", "value": f"v{summary.get('version', '—')}" if ready else "not trained",
         "sub": MODEL_NAME, "tone": "good" if ready else "bad"},
        {"label": "Edge cache", "value": "ready" if cache_ready else "empty",
         "sub": f"{len(cached_files)} cached file(s)", "tone": "good" if cache_ready else "warn"},
        {"label": "Cloud link", "value": "online" if st.session_state.get("sim_cloud_online", True) else "offline",
         "sub": "simulated toggle", "tone": "good" if st.session_state.get("sim_cloud_online", True) else "bad"},
        {"label": "Offline inference", "value": "available" if cache_ready else "unavailable",
         "sub": "falls back to the cached champion", "tone": "good" if cache_ready else "warn"},
    ])
    ui.section("MODULE 12 · LOCAL PROOF", "Run an inference on this host")
    junction = junction_select("Junction", junctions, "edge_junction")
    if st.button("Run local (offline) inference", type="primary"):
        started = time.perf_counter()
        try:
            result = predict_next_hour(data, junction, offline=True)
            st.session_state["edge_test"] = {
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "junction": junction,
                "vehicles": result["predicted_vehicles"],
                "forecast_for": str(result["forecast_for"]),
            }
        except Exception as error:
            st.session_state["edge_test"] = {"error": str(error)}
    test = st.session_state.get("edge_test")
    if test and "error" in test:
        st.error(f"Local inference failed: {test['error']}")
    elif test:
        ui.kpi_grid([
            {"label": "Local latency", "value": f"{test['latency_ms']} ms", "sub": "in-process, no network", "tone": "good"},
            {"label": "Estimate", "value": f"{test['vehicles']:.0f} veh/h", "sub": test["junction"], "tone": "info"},
            {"label": "Forecast for", "value": pd.Timestamp(test["forecast_for"]).strftime("%d %b %H:%M"), "sub": "edge cache artifact", "tone": "violet"},
        ])
        st.caption("This call is served entirely from the host-local edge cache — useful when the simulated 5G link is down.")
    else:
        st.caption("Run the proof to measure a real in-process inference from the local cache.")

    if not virtual:
        st.info("Generate a virtual tick in Simulation lab to inspect the simulated network link.")
        return
    latest = virtual[0]
    ui.section("MODULE 12 · LINK", "Simulated 5G link")
    ui.kpi_grid([
        {"label": "Virtual tick", "value": str(int(num(latest, "tick", 0))), "sub": "simulation clock", "tone": "violet"},
        {"label": "Signal", "value": f"{int(num(latest, 'signal_strength_dbm'))} dBm", "sub": "simulated 5G", "tone": "info"},
        {"label": "Upload", "value": f"{num(latest, 'upload_mbps'):.1f} Mbps", "sub": "edge to cloud", "tone": "good"},
        {"label": "Download", "value": f"{num(latest, 'download_mbps'):.1f} Mbps", "sub": "cloud to edge", "tone": "good"},
        {"label": "Latency", "value": f"{num(latest, 'network_latency_ms'):.1f} ms" if latest.get("network_latency_ms") is not None else "Offline",
         "sub": "round trip", "tone": "warn" if latest.get("network_latency_ms") is None else "info"},
        {"label": "Link state", "value": str(latest.get("cloud_link", "online")), "sub": f"{len(virtual)} junctions reporting", "tone": "good" if str(latest.get("cloud_link")) == "online" else "bad"},
    ])
    ui.section("MODULE 12 · FLEET", "Edge and network status by junction")
    frame = pd.DataFrame([{
        "Junction": row["junction"], "Sensor": row.get("sensor_status"), "Edge model": row.get("edge_model_status"),
        "Edge latency ms": row.get("edge_inference_latency_ms"), "Cloud": row.get("cloud_link"),
        "Signal dBm": row.get("signal_strength_dbm"), "Upload Mbps": row.get("upload_mbps"),
        "Download Mbps": row.get("download_mbps"), "Network ms": row.get("network_latency_ms"),
    } for row in virtual])
    st.dataframe(frame, width="stretch", hide_index=True)
    if cached_files:
        ui.section("MODULE 12 · CACHE", "Host-local artifacts")
        st.table(pd.DataFrame([{
            "File": str(path.relative_to(EDGE_CACHE)), "Size KB": round(path.stat().st_size / 1024, 1)
        } for path in sorted(cached_files)]), hide_index=True)
    st.caption("All network values are generated by the simulator. The local inference proof above is real and runs on this machine.")


def render_analytics(data: pd.DataFrame) -> None:
    ui.page_head("MODULE 13 · ANALYTICS", "Analytics & reports",
                 "Explore the historical baseline, persisted virtual telemetry, forecasts, alerts, incidents, and device activity — then export a daily or weekly pack.")
    junctions = sorted(data["junction"].unique().tolist())
    selected = st.multiselect("Junctions", junctions, default=junctions)
    hours = st.select_slider("History window", options=[24, 72, 168, 720, 5000], value=720,
                             format_func=lambda count: f"{count} hours" if count < 5000 else "Full history")
    filtered = data[data["junction"].isin(selected)].copy()
    if hours < 5000:
        max_time = filtered["timestamp"].max()
        filtered = filtered[filtered["timestamp"] >= max_time - pd.Timedelta(hours=hours)]
    if filtered.empty:
        st.info("Select at least one junction.")
        return
    smoothed = filtered.sort_values("timestamp").copy()
    smoothed["rolling_24"] = smoothed.groupby("junction")["vehicles"].transform(lambda series: series.rolling(24, min_periods=1).mean())
    tab_flow, tab_weather, tab_forecasts, tab_alerts, tab_ops = st.tabs(
        ["Traffic flow", "Weather context", "Forecast history", "Alerts & incidents", "Devices & actions"]
    )
    with tab_flow:
        st.plotly_chart(ui.style_fig(px.line(smoothed, x="timestamp", y="vehicles", color="junction",
                                             labels={"vehicles": "Vehicles/hour", "timestamp": "", "junction": "Junction"}), 320),
                        width="stretch", config={"displayModeBar": False})
        st.plotly_chart(ui.style_fig(px.line(smoothed, x="timestamp", y="rolling_24", color="junction",
                                             labels={"rolling_24": "24-hour rolling mean", "timestamp": "", "junction": "Junction"}), 320),
                        width="stretch", config={"displayModeBar": False})
    with tab_weather:
        virtual = pd.DataFrame(recent_virtual_readings(2000))
        if virtual.empty:
            st.info("Generate simulated telemetry in Simulation lab.")
        else:
            for column in ("temperature_c", "humidity_pct", "rain_probability_pct", "wind_speed_kmph", "visibility_km"):
                if column not in virtual:
                    virtual[column] = None
            st.plotly_chart(ui.style_fig(px.line(virtual.sort_values("tick"), x="recorded_at",
                                                 y=["temperature_c", "humidity_pct", "rain_probability_pct", "wind_speed_kmph"],
                                                 labels={"recorded_at": "", "value": "Virtual reading", "variable": "Metric"}), 340),
                            width="stretch", config={"displayModeBar": False})
            st.caption("Weather variables are synthetic simulator readings; the source CSV does not contain weather observations.")
    with tab_forecasts:
        predictions = pd.DataFrame(recent_predictions(500))
        if predictions.empty:
            st.info("Forecasts will appear after the first prediction.")
        else:
            st.plotly_chart(ui.style_fig(px.scatter(predictions, x="created_at", y="predicted_vehicles", color="junction",
                                                    hover_data=["forecast_for", "id"], labels={"created_at": "", "predicted_vehicles": "Vehicles/hour"}), 320),
                            width="stretch", config={"displayModeBar": False})
            st.dataframe(tidy_table(predictions.drop(columns=["inputs"]).rename(columns={"id": "ID"})),
                         width="stretch", hide_index=True)
    with tab_alerts:
        alerts = pd.DataFrame(categorize_alerts(recent_virtual_alerts(300)))
        incidents = pd.DataFrame(recent_incidents(300))
        st.markdown("**Alert history**")
        if alerts.empty:
            st.caption("No alerts recorded.")
        else:
            st.dataframe(
                tidy_table(alerts[["id", "tick", "created_at", "junction", "category", "severity", "alert", "acknowledged"]])
                .rename(columns={"id": "ID", "created_at": "Time", "junction": "Junction"}),
                width="stretch", hide_index=True,
            )
        st.markdown("**Incident reports**")
        if incidents.empty:
            st.caption("No incident reports recorded.")
        else:
            st.dataframe(tidy_table(incidents).rename(columns={"id": "ID", "created_at": "Recorded", "junction": "Junction"}),
                         width="stretch", hide_index=True)
    with tab_ops:
        devices = pd.DataFrame(list_devices())
        actions = pd.DataFrame(recent_operator_actions(300))
        st.markdown("**Device rack state**")
        st.dataframe(
            tidy_table(devices[["name", "kind", "state", "mode", "updated_by", "updated_at"]])
            .rename(columns={"name": "Device", "kind": "Kind", "state": "State", "mode": "Mode",
                             "updated_by": "Changed by", "updated_at": "Updated"}),
            width="stretch", hide_index=True,
        )
        st.markdown("**Operator action history**")
        if actions.empty:
            st.caption("No actions recorded.")
        else:
            st.dataframe(tidy_table(actions).rename(columns={"id": "ID", "created_at": "Recorded", "junction": "Junction"}),
                         width="stretch", hide_index=True)

    ui.section("MODULE 13 · EXPORT", "Report pack")
    report_day = filtered["timestamp"].max().strftime("%Y-%m-%d")
    st.caption("Daily and weekly packs summarise the filtered traffic baseline plus the persisted simulator activity.")
    columns = st.columns(4)
    columns[0].download_button("Daily report (MD)", render_period_report(filtered, days=1, label="Daily"),
                               file_name=f"flow28-daily-{report_day}.md", mime="text/markdown")
    columns[1].download_button("Weekly report (MD)", render_period_report(filtered, days=7, label="Weekly"),
                               file_name=f"flow28-weekly-{report_day}.md", mime="text/markdown")
    columns[2].download_button("Observations (CSV)", filtered.to_csv(index=False),
                               file_name=f"traffic-observations-{report_day}.csv", mime="text/csv")
    virtual_frame = pd.DataFrame(recent_virtual_readings(2000))
    if not virtual_frame.empty:
        columns[3].download_button("Sensor history (CSV)", virtual_frame.to_csv(index=False),
                                   file_name="virtual-sensor-history.csv", mime="text/csv")
    columns = st.columns(4)
    predictions = pd.DataFrame(recent_predictions(500))
    if not predictions.empty:
        columns[0].download_button("Forecasts (CSV)", predictions.drop(columns=["inputs"]).to_csv(index=False),
                                   file_name="forecast-history.csv", mime="text/csv")
    incidents = pd.DataFrame(recent_incidents(500))
    if not incidents.empty:
        columns[1].download_button("Incidents (CSV)", incidents.to_csv(index=False),
                                   file_name="incident-history.csv", mime="text/csv")
    actions = pd.DataFrame(recent_operator_actions(500))
    if not actions.empty:
        columns[2].download_button("Actions & green time (CSV)", actions.to_csv(index=False),
                                   file_name="operator-actions.csv", mime="text/csv")
    devices_frame = pd.DataFrame(list_devices())
    if not devices_frame.empty:
        columns[3].download_button("Device states (CSV)", devices_frame.to_csv(index=False),
                                   file_name="device-states.csv", mime="text/csv")


def render_period_report(data: pd.DataFrame, *, days: int, label: str) -> str:
    end = data["timestamp"].max()
    start = end.normalize() - pd.Timedelta(days=days - 1)
    window = data[(data["timestamp"] >= start) & (data["timestamp"] <= end)]
    if window.empty:
        window = data.tail(24)
    per_junction = window.groupby("junction")["vehicles"].agg(["mean", "max", "count"]).round(1)
    table = "\n".join(
        f"| {name} | {row['mean']:.1f} | {row['max']:.0f} | {int(row['count'])} |"
        for name, row in per_junction.iterrows()
    )
    alerts = categorize_alerts(recent_virtual_alerts(500))
    incidents = recent_incidents(500)
    predictions = recent_predictions(500)
    actions = recent_operator_actions(500)
    counts = Counter(str(row["category"]) for row in alerts)
    categories = ", ".join(f"{name} {counts.get(name, 0)}" for name in CATEGORY_TONE)
    return (
        f"# Flow28 {label.lower()} traffic report\n\n"
        f"Window: **{start.strftime('%d %b %Y')} → {end.strftime('%d %b %Y %H:%M')}**  \n"
        f"Junctions in scope: **{window['junction'].nunique()}**  \n"
        f"Observations: **{len(window):,}**\n\n"
        "## Traffic baseline\n\n"
        f"- Mean flow: **{window['vehicles'].mean():.1f} vehicles/hour**\n"
        f"- Peak flow: **{window['vehicles'].max():.0f} vehicles/hour**\n"
        f"- Quietest hour: **{window['vehicles'].min():.0f} vehicles/hour**\n\n"
        "| Junction | Mean veh/h | Peak veh/h | Observations |\n"
        "| --- | --- | --- | --- |\n"
        f"{table}\n\n"
        "## Persisted simulator activity\n\n"
        f"- Threshold alerts retained: **{len(alerts)}** ({categories})\n"
        f"- Incident reports: **{len(incidents)}**\n"
        f"- Forecasts served: **{len(predictions)}**\n"
        f"- Operator actions: **{len(actions)}**\n\n"
        "## Notes\n\n"
        "Weather, network, device, and sensor values are generated by the project simulator and labeled virtual. "
        "Alert categories and corridor plans are deterministic rule-based guidance, not trained model output. "
        "Image triage is a heuristic pixel measurement, not a classifier. "
        "No personal data is stored; incident history keeps only the uploaded file name.\n"
    )


def render_mlops(data: pd.DataFrame, ready: bool, summary: dict[str, object]) -> None:
    ui.page_head("MODULE 14 · LIFECYCLE", "MLOps & data",
                 "Train on a chronological holdout, register every version, promote only an improved champion, and track feature drift.")
    if ready:
        rmse = summary.get("rmse")
        mae = summary.get("mae")
        r2 = summary.get("r2")
        ui.kpi_grid([
            {"label": "Champion version", "value": f"v{summary.get('version', '—')}", "sub": "registry alias", "tone": "good"},
            {"label": "Holdout RMSE", "value": f"{rmse:.2f}" if isinstance(rmse, (int, float)) else "—", "sub": "vehicles/hour", "tone": "info"},
            {"label": "Holdout MAE", "value": f"{mae:.2f}" if isinstance(mae, (int, float)) else "—", "sub": "vehicles/hour", "tone": "violet"},
            {"label": "Holdout R²", "value": f"{r2:.3f}" if isinstance(r2, (int, float)) else "—", "sub": "variance explained", "tone": "good"},
        ])
        st.caption(f"Champion trained on {summary.get('dataset') or 'the current dataset'} · promoted only when holdout RMSE improves.")
    else:
        st.info("No champion yet. Train and register a model to populate the registry panel.")

    train_col, monitor_col = st.columns(2, gap="large")
    with train_col:
        ui.section("MODULE 14 · TRAIN", "Register a new version")
        if st.button("Train and register model", type="primary", width="stretch"):
            with st.spinner("Training and logging the model to MLflow…"):
                try:
                    st.session_state["last_training"] = train_model(data)
                    model_state.clear()
                except Exception as error:
                    st.error(f"Training failed: {error}")
        result = st.session_state.get("last_training")
        if result:
            st.success(f"{'Promoted to champion' if result['promoted'] else 'Registered as candidate'} · version {result['model_version']}")
            ui.kpi_grid([
                {"label": "RMSE", "value": f"{result['metrics']['rmse']:.3f}", "sub": "holdout", "tone": "info"},
                {"label": "MAE", "value": f"{result['metrics']['mae']:.3f}", "sub": "holdout", "tone": "violet"},
                {"label": "R²", "value": f"{result['metrics']['r2']:.3f}", "sub": "holdout", "tone": "good"},
            ])
            st.caption(f"Run ID: {result['run_id']} · train rows {result['training_rows']:,} · test rows {result['test_rows']:,}")
    with monitor_col:
        ui.section("MODULE 14 · DRIFT", "Population stability index")
        if st.button("Run drift check", width="stretch"):
            with st.spinner("Comparing historical and recent feature distributions…"):
                try:
                    st.session_state["last_monitor"] = monitor_data(data)
                except Exception as error:
                    st.error(f"Monitoring failed: {error}")
        drift = st.session_state.get("last_monitor")
        if drift:
            tone = "bad" if drift["status"] == "drift_detected" else "good"
            ui.kpi_grid([
                {"label": "Drift status", "value": drift["status"].replace("_", " ").title(), "sub": drift["recommendation"], "tone": tone},
                {"label": "Max PSI", "value": f"{drift['max_psi']:.3f}", "sub": f"review threshold {drift['threshold']}", "tone": tone},
            ])
            ui.psi_bars(drift["features"], drift["threshold"])

    st.divider()
    st.markdown("**MLflow tracking UI**")
    st.code("mlflow ui --backend-store-uri sqlite:///mlflow_project28.db --host 127.0.0.1 --port 5000", language="powershell")
    st.caption(f"Registered model: {MODEL_NAME} · the champion alias moves only when holdout RMSE improves on the same dataset.")
    ui.section("MODULE 14 · SOURCE", "Data quality")
    ui.kpi_grid([
        {"label": "Rows", "value": f"{len(data):,}", "sub": "hourly observations", "tone": "info"},
        {"label": "Junctions", "value": f"{data['junction'].nunique()}", "sub": "continuous histories", "tone": "violet"},
        {"label": "Missing values", "value": f"{int(data.isna().sum().sum())}", "sub": "after normalization", "tone": "good"},
        {"label": "First record", "value": pd.Timestamp(data["timestamp"].min()).strftime("%d %b %Y"), "sub": "training window start", "tone": "warn"},
        {"label": "Last record", "value": pd.Timestamp(data["timestamp"].max()).strftime("%d %b %Y"), "sub": "training window end", "tone": "warn"},
    ])
    st.download_button("Download normalized dataset", data.to_csv(index=False), file_name="traffic-normalized.csv", mime="text/csv")


def render_simulation_lab(data: pd.DataFrame, model_ready: bool) -> None:
    ui.page_head("MODULE 15 · SIMULATION", "Software simulation lab",
                 "Reproducible virtual IoT, weather, 5G, edge, alert, and signal-control scenarios — no physical hardware required.")
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            if "sim_cloud_online" not in st.session_state:
                st.session_state["sim_cloud_online"] = True
            st.toggle("Simulated cloud link online", key="sim_cloud_online",
                      help="Turn off to test local edge inference and offline-to-cloud alerts.")
        with right:
            fault_options = ["None"] + sorted(data["junction"].unique().tolist())
            if "sim_fault_selector" in st.session_state and st.session_state["sim_fault_selector"] not in fault_options:
                del st.session_state["sim_fault_selector"]
            st.selectbox("Inject virtual sensor fault", fault_options, key="sim_fault_selector")
        if st.button("Generate next virtual minute", type="primary", width="stretch"):
            st.session_state["sim_failed_junction"] = None if st.session_state["sim_fault_selector"] == "None" else st.session_state["sim_fault_selector"]
            advance_virtual_tick(data, model_ready)
            st.rerun()
    virtual = current_virtual_readings()
    if virtual:
        frame = pd.DataFrame(virtual)
        latest = virtual[0]
        ui.section("MODULE 15 · NETWORK", "Simulated 5G link")
        ui.kpi_grid([
            {"label": "Virtual tick", "value": str(int(num(latest, "tick", 0))), "sub": "simulation clock", "tone": "violet"},
            {"label": "Signal", "value": f"{int(num(latest, 'signal_strength_dbm'))} dBm", "sub": "simulated 5G", "tone": "info"},
            {"label": "Upload", "value": f"{num(latest, 'upload_mbps'):.1f} Mbps", "sub": "edge to cloud", "tone": "good"},
            {"label": "Download", "value": f"{num(latest, 'download_mbps'):.1f} Mbps", "sub": "cloud to edge", "tone": "good"},
            {"label": "Latency", "value": f"{num(latest, 'network_latency_ms'):.1f} ms" if latest.get("network_latency_ms") is not None else "Offline",
             "sub": "round trip", "tone": "warn" if latest.get("network_latency_ms") is None else "info"},
        ])
        ui.section("MODULE 15 · DEVICES", "Virtual device fleet")
        st.dataframe(frame[["junction", "sensor_status", "edge_model_status", "edge_inference_latency_ms", "cloud_link",
                            "signal_strength_dbm", "upload_mbps", "download_mbps", "network_latency_ms"]],
                     width="stretch", hide_index=True)
    else:
        st.info("Generate the first virtual tick to populate the simulated sensors.")

    ui.section("MODULE 15 · FAILOVER", "Edge AI and offline operation")
    cache_ready = (EDGE_CACHE / "champion.json").exists()
    ready, summary = model_state()
    ui.kpi_grid([
        {"label": "MLflow champion", "value": f"v{summary.get('version', '—')}" if ready else "not trained",
         "sub": MODEL_NAME, "tone": "good" if ready else "bad"},
        {"label": "Edge cache", "value": "ready" if cache_ready else "empty", "sub": "host-local artifact copy", "tone": "good" if cache_ready else "warn"},
        {"label": "Cloud link", "value": "online" if st.session_state.get("sim_cloud_online", True) else "offline",
         "sub": "simulated toggle", "tone": "good" if st.session_state.get("sim_cloud_online", True) else "bad"},
        {"label": "Offline inference", "value": "available" if cache_ready else "unavailable",
         "sub": "falls back to cached champion", "tone": "good" if cache_ready else "warn"},
    ])
    if not cache_ready:
        st.info("After a model is promoted, the MLOps pipeline caches its MLflow artifact locally for offline simulation.")
    ui.section("MODULE 15 · EVENTS", "Virtual alerts")
    alerts = pd.DataFrame(recent_virtual_alerts(100))
    if alerts.empty:
        st.success("No simulated alerts on the generated ticks.")
    else:
        st.dataframe(
            tidy_table(alerts).rename(columns={"id": "ID", "created_at": "Recorded", "junction": "Junction"}),
            width="stretch", hide_index=True,
        )
    st.caption("All network, weather, sensor fault, and signal values are generated by the project simulator and labeled virtual.")


data = get_data()
junctions = sorted(data["junction"].unique().tolist())
configure_mlflow()
model_ready, champion = model_state()
virtual = current_virtual_readings()
if not virtual_feed_current():
    seed_virtual_history(data, model_ready)
    virtual = current_virtual_readings()
init_profile(junctions)
page = render_sidebar(junctions)
pop_flash()

if page == "Operator access":
    render_operator_access(data, junctions, model_ready, champion)
elif page == "Dashboard":
    render_dashboard(data, virtual, model_ready, champion)
elif page == "Live sensors":
    render_live(data, junctions)
elif page == "Weather monitor":
    render_weather(junctions)
elif page == "Alert centre":
    render_alerts(junctions)
elif page == "Junction map":
    render_map(virtual)
elif page == "Forecast & outlook":
    render_forecast(data, junctions, model_ready, champion)
elif page == "Smart control":
    render_control(data, junctions, virtual)
elif page == "Incident vision":
    render_incident_vision(data, junctions)
elif page == "Corridor advisory":
    render_corridor(virtual)
elif page == "AI assistant":
    render_assistant(data, junctions, champion)
elif page == "Edge & 5G":
    render_edge(data, junctions, model_ready, champion)
elif page == "Analytics & reports":
    render_analytics(data)
elif page == "MLOps & data":
    render_mlops(data, model_ready, champion)
else:
    render_simulation_lab(data, model_ready)

st.markdown(
    '<div class="footer-note">FLOW28 · AI-BASED TRAFFIC PREDICTION SYSTEM WITH MLOPS · PROJECT 28 · '
    'K.MAHIJA · SK.ANJUM · P.HASITHA · ALL TELEMETRY IS SIMULATED</div>',
    unsafe_allow_html=True,
)
