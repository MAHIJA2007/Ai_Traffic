"""Deterministic software-only virtual traffic, weather, network, and signal lab."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from traffic_mlops.data import normalize_data


def generate_virtual_tick(
    source: pd.DataFrame,
    tick: int,
    *,
    cloud_online: bool = True,
    failed_junction: str | None = None,
    model_ready: bool = True,
    now: datetime | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Synthesize one reproducible telemetry tick from each junction's history."""
    frame = normalize_data(source)
    now = now or datetime.now(timezone.utc)
    timestamp = now.astimezone(timezone.utc).isoformat()
    rng = np.random.default_rng(2800 + int(tick))
    readings: list[dict[str, object]] = []
    alerts: list[dict[str, object]] = []

    for junction, history in frame.groupby("junction", sort=True):
        history = history.sort_values("timestamp")
        hour_profile = history[history["timestamp"].dt.hour == now.hour]
        reference = hour_profile.tail(24) if not hour_profile.empty else history.tail(24)
        base_flow = float(reference["vehicles"].median())
        flow = max(0.0, float(rng.normal(base_flow, max(base_flow * 0.06, 2))))
        capacity = max(float(history["vehicles"].quantile(.98)), 1.0)
        occupancy = float(np.clip(flow / capacity * 100, 0, 100))
        speed = float(np.clip(72 - occupancy * .53 + rng.normal(0, 3), 5, 72))
        temperature = float(np.clip(history["temperature_c"].tail(48).mean() + rng.normal(0, 2.4), -15, 46))
        humidity = float(np.clip(rng.normal(58, 15), 18, 98))
        rain_probability = float(np.clip(rng.beta(1.7, 4.5) * 100, 0, 100))
        light = float(np.clip((22000 if 6 <= now.hour <= 18 else 120) * rng.uniform(.65, 1.3), 0, 100000))
        sensor_online = str(junction) != failed_junction
        signal_dbm = int(rng.integers(-105, -54)) if cloud_online else int(rng.integers(-120, -108))
        signal_green = int(np.clip(20 + occupancy * .42, 20, 70))
        reading: dict[str, object] = {
            "tick": int(tick), "recorded_at": timestamp, "junction": str(junction),
            "virtual": True, "sensor_status": "online" if sensor_online else "offline (fault injected)",
            "vehicles_per_hour": round(flow, 1), "average_speed_kmh": round(speed, 1),
            "road_occupancy_pct": round(occupancy, 1), "temperature_c": round(temperature, 1),
            "humidity_pct": round(humidity, 1), "rain_probability_pct": round(rain_probability, 1),
            "light_intensity_lux": round(light, 0), "water_on_road_pct": round(float(np.clip(rain_probability * .72 + rng.normal(0, 8), 0, 100)), 1),
            "signal_green_seconds_auto": signal_green,
            "edge_model_status": "champion ready" if model_ready else "model not trained",
            "edge_inference_latency_ms": round(float(rng.uniform(8, 55)), 1) if model_ready else None,
            "cloud_link": "online" if cloud_online else "offline (simulated)",
            "network_technology": "simulated 5G",
            "signal_strength_dbm": signal_dbm,
            "upload_mbps": round(float(rng.uniform(8, 145)), 1) if cloud_online else 0.0,
            "download_mbps": round(float(rng.uniform(35, 820)), 1) if cloud_online else 0.0,
            "network_latency_ms": round(float(rng.uniform(7, 45)), 1) if cloud_online else None,
        }
        readings.append(reading)
        if not sensor_online:
            alerts.append({"tick": tick, "created_at": timestamp, "junction": str(junction), "severity": "critical", "alert": "Virtual sensor offline (fault injection)"})
        if occupancy >= 82:
            alerts.append({"tick": tick, "created_at": timestamp, "junction": str(junction), "severity": "critical", "alert": f"Congestion risk: simulated occupancy {occupancy:.0f}%"})
        elif occupancy >= 68:
            alerts.append({"tick": tick, "created_at": timestamp, "junction": str(junction), "severity": "warning", "alert": f"Elevated traffic: simulated occupancy {occupancy:.0f}%"})
        if temperature >= 38:
            alerts.append({"tick": tick, "created_at": timestamp, "junction": str(junction), "severity": "warning", "alert": f"High simulated road temperature: {temperature:.1f} °C"})
        if rain_probability >= 80:
            alerts.append({"tick": tick, "created_at": timestamp, "junction": str(junction), "severity": "warning", "alert": f"Weather hazard risk: simulated rain probability {rain_probability:.0f}%"})
        if not cloud_online:
            alerts.append({"tick": tick, "created_at": timestamp, "junction": str(junction), "severity": "info", "alert": "Cloud link down; local edge mode remains available"})

    return readings, alerts
