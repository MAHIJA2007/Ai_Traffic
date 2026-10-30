"""Deterministic traffic advisories: alert taxonomy, weather guidance, signal plans, and scene triage.

Everything here is pure Python (no Streamlit, no model, no network). The simulator supplies the
readings; this module turns them into operator-facing guidance that the UI can render and test.
"""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from statistics import mean
from typing import Iterable, Sequence

from PIL import Image, ImageFilter, ImageStat


CATEGORIES = ("congestion", "weather", "safety", "sensor", "network", "other")

_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("weather", ("rain", "wet", "water", "wind", "fog", "visibility", "temperature", "weather", "storm", "frost", "ice")),
    ("network", ("cloud link", "cloud", "network", "latency", "5g", "signal strength", "upload", "download")),
    ("sensor", ("sensor", "telemetry", "fault injection", "offline")),
    ("safety", ("incident", "collision", "breakdown", "closure", "hazard", "debris")),
    ("congestion", ("congestion", "occupancy", "traffic", "queue", "flow", "demand")),
)


def alert_category(text: str, severity: str = "") -> str:
    """Bucket a free-text alert into one operational category."""
    lowered = f"{text} {severity}".lower()
    for category, keywords in _CATEGORY_RULES:
        if any(keyword in lowered for keyword in keywords):
            return category
    return "other"


def categorize_alerts(alerts: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Copy alerts and attach a category, leaving the source rows untouched."""
    enriched: list[dict[str, object]] = []
    for alert in alerts:
        row = dict(alert)
        row["category"] = alert_category(str(row.get("alert", "")), str(row.get("severity", "")))
        enriched.append(row)
    return enriched


def weather_index(reading: dict[str, object]) -> dict[str, object]:
    """Blend simulated weather readings into one 0-100 severity index."""
    rain = _number(reading.get("rain_probability_pct"))
    water = _number(reading.get("water_on_road_pct"))
    wind = _number(reading.get("wind_speed_kmph"))
    visibility = _number(reading.get("visibility_km"), fallback=16.0)
    value = 100 * (
        0.34 * rain / 100
        + 0.16 * water / 100
        + 0.14 * min(wind / 90, 1.0)
        + 0.16 * (1 - min(visibility / 16, 1.0))
        + 0.20 * min(max(_number(reading.get("humidity_pct"), 50.0) - 55, 0) / 45, 1.0)
    )
    value = max(0.0, min(value, 100.0))
    if value >= 70:
        band, tone = "Severe", "bad"
    elif value >= 45:
        band, tone = "Adverse", "warn"
    elif value >= 25:
        band, tone = "Unsettled", "info"
    else:
        band, tone = "Clear", "good"
    return {"value": round(value, 1), "band": band, "tone": tone}


def weather_suggestions(reading: dict[str, object]) -> list[dict[str, str]]:
    """Turn one simulated weather reading into operator suggestions."""
    rain = _number(reading.get("rain_probability_pct"))
    water = _number(reading.get("water_on_road_pct"))
    wind = _number(reading.get("wind_speed_kmph"))
    visibility = _number(reading.get("visibility_km"), fallback=16.0)
    temperature = _number(reading.get("temperature_c"), fallback=20.0)
    humidity = _number(reading.get("humidity_pct"), fallback=55.0)
    suggestions: list[dict[str, str]] = []

    if rain >= 65:
        suggestions.append({"label": "Wet-road protocol", "tone": "warn",
                            "detail": "Post the wet-weather message, widen the following-gap advisory, and expect longer stopping distances."})
    elif rain >= 35:
        suggestions.append({"label": "Carry a rain advisory", "tone": "info",
                            "detail": "Rain probability is moderate; keep a wet-weather advisory drafted for the next tick."})
    if water >= 45:
        suggestions.append({"label": "Standing-water watch", "tone": "warn",
                            "detail": "Inspect low-lying approaches and review the drainage escalation list."})
    if wind >= 55:
        suggestions.append({"label": "High-profile vehicle advisory", "tone": "bad",
                            "detail": "Advise high-sided vehicles and two-wheelers; hold lightweight signage deployment."})
    if visibility <= 4:
        suggestions.append({"label": "Low-visibility procedure", "tone": "bad",
                            "detail": "Activate fog messaging, reduce the advisory speed, and increase patrol spacing."})
    elif visibility <= 7:
        suggestions.append({"label": "Reduced visibility", "tone": "warn",
                            "detail": "Keep headline messaging concise and confirm the lighting circuit is on."})
    if temperature >= 38:
        suggestions.append({"label": "Heat load on pavement", "tone": "warn",
                            "detail": "Watch for speed variance and surface damage; schedule inspections outside the heat peak."})
    elif temperature <= 2 or (temperature <= 6 and humidity >= 85):
        suggestions.append({"label": "Frost risk", "tone": "warn",
                            "detail": "Flag shaded approaches and bridges for gritting and early-morning patrols."})

    if not suggestions:
        suggestions.append({"label": "Clear running conditions", "tone": "good",
                            "detail": "No weather advisory needed; continue routine corridor monitoring."})
    return suggestions[:4]


def signal_plan(occupancy: float, *, weather_value: float = 0.0, event: bool = False,
                override: int | None = None) -> dict[str, object]:
    """Derive a green-phase allocation from simulated occupancy and conditions."""
    occupancy = max(0.0, min(float(occupancy), 100.0))
    weather_value = max(0.0, min(float(weather_value), 100.0))
    derived = 20 + occupancy * 0.42
    if weather_value >= 45:
        derived += 4
    if event:
        derived += 8
    derived = int(round(max(20, min(derived, 75))))

    if override is not None:
        green = int(max(15, min(int(override), 90)))
        mode = "Manual"
    else:
        green = derived
        mode = "Automatic"

    if occupancy >= 78:
        phase, tone = "hold", "bad"
    elif occupancy >= 62:
        phase, tone = "balance", "warn"
    else:
        phase, tone = "release", "good"

    reasons = [f"simulated occupancy {occupancy:.0f}%"]
    if weather_value >= 45:
        reasons.append("adverse weather adds clearance time")
    if event:
        reasons.append("special event adds demand")
    return {
        "green_seconds": green,
        "cycle_seconds": green + 30,
        "phase": phase,
        "mode": mode,
        "derived_seconds": derived,
        "tone": tone,
        "reason": "; ".join(reasons),
    }


def corridor_plans(readings: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Rank adaptation plans 0-99 by how strongly the current readings support them."""
    rows = [row for row in readings if row]
    if not rows:
        return []
    occupancies = sorted((_number(row.get("road_occupancy_pct")) for row in rows), reverse=True)
    top_two = mean(occupancies[:2]) if len(occupancies) >= 2 else occupancies[0]
    peak = occupancies[0]
    mean_occupancy = mean(occupancies)
    speeds = [_number(row.get("average_speed_kmh"), 60.0) for row in rows]
    speed_deficit = max(0.0, (60 - mean(speeds)) / 60)
    mean_lux = mean(_number(row.get("light_intensity_lux"), 12000.0) for row in rows)
    mean_visibility = mean(_number(row.get("visibility_km"), 16.0) for row in rows)
    rain = mean(_number(row.get("rain_probability_pct")) for row in rows)
    water = max(_number(row.get("water_on_road_pct")) for row in rows)
    wind = max(_number(row.get("wind_speed_kmph")) for row in rows)
    weather = mean(float(weather_index(row)["value"]) for row in rows)
    offline = sum(1 for row in rows if not str(row.get("sensor_status", "")).lower().startswith("online"))
    cloud_down = any(str(row.get("cloud_link", "online")).lower() != "online" for row in rows)

    candidates = [
        ("Adaptive signal retiming", 34 + 0.58 * top_two + (6 if peak >= 78 else 0),
         f"Rebalance green time across the two busiest approaches (top occupancy {peak:.0f}%)."),
        ("Ramp metering", 20 + 0.55 * peak,
         "Release on-ramp demand in measured platoons while the corridor is loaded."),
        ("Traveller messaging", 16 + 0.62 * weather + (8 if rain >= 65 else 0) + (4 if cloud_down else 0),
         f"Weather index {weather:.0f}/100 — publish the matching variable-message-sign text."),
        ("Incident patrol", 14 + 0.30 * peak + 16 * offline + (10 if water >= 50 else 0) + (8 if wind >= 55 else 0),
         "Pre-position a patrol near the busiest zone; sensor faults or storm surge raise the priority."),
        ("Lighting response", 12 + (28 if mean_lux < 800 else 0) + 4 * max(0.0, 8 - mean_visibility)
         + (10 if mean_visibility <= 4 else 0),
         f"Mean ambient light {mean_lux:,.0f} lux, visibility {mean_visibility:.1f} km."),
        ("Demand broadcast", 15 + 0.35 * mean_occupancy + (6 if speed_deficit > 0.30 else 0),
         "Issue a suggested departure-time or alternate-corridor message when demand drains speed."),
    ]

    plans: list[dict[str, object]] = []
    for name, raw, detail in candidates:
        score = int(round(max(5, min(raw, 99))))
        if score >= 80:
            tone = "good"
        elif score >= 60:
            tone = "info"
        elif score >= 45:
            tone = "violet"
        else:
            tone = "warn"
        plans.append({"name": name, "score": score, "detail": detail, "tone": tone})
    return sorted(plans, key=lambda item: int(item["score"]), reverse=True)


def device_recommendation(devices: Sequence[dict[str, object]],
                          readings: Sequence[dict[str, object]]) -> dict[str, dict[str, str]]:
    """Suggest a state and mode per device from the current virtual readings."""
    rows = [row for row in readings if row]
    if not rows:
        return {}
    occupancies = sorted((_number(row.get("road_occupancy_pct")) for row in rows), reverse=True)
    peak = occupancies[0]
    mean_lux = mean(_number(row.get("light_intensity_lux"), 12000.0) for row in rows)
    mean_visibility = mean(_number(row.get("visibility_km"), 16.0) for row in rows)
    weather = max(float(weather_index(row)["value"]) for row in rows)
    patrol = next((item for item in corridor_plans(rows) if item["name"] == "Incident patrol"), None)
    patrol_score = int(patrol["score"]) if patrol else 0

    recommendations: dict[str, dict[str, str]] = {}
    for device in devices:
        slug = str(device.get("slug", ""))
        kind = str(device.get("kind", ""))
        if kind == "signal" or "signal" in slug:
            recommendations[slug] = {"state": "on", "mode": "auto",
                                     "reason": f"Adaptive timing holds the corridor at {peak:.0f}% peak occupancy."}
        elif kind == "flow" or "ramp" in slug:
            if peak >= 70:
                recommendations[slug] = {"state": "on", "mode": "auto",
                                         "reason": f"Peak occupancy {peak:.0f}% supports metered releases."}
            else:
                recommendations[slug] = {"state": "off", "mode": "auto",
                                         "reason": f"Peak occupancy {peak:.0f}% — metering not needed yet."}
        elif kind == "signage" or "vms" in slug:
            text = ("Wet road · reduce speed" if weather >= 45 else
                    "Heavy traffic ahead · consider alternate route" if peak >= 62 else
                    "Drive safely · corridor nominal")
            recommendations[slug] = {"state": "on", "mode": "auto",
                                     "reason": f'Display "{text}" (weather index {weather:.0f}/100).'}
        elif kind == "lighting" or "light" in slug:
            if mean_lux < 8000 or mean_visibility <= 6:
                recommendations[slug] = {"state": "on", "mode": "auto",
                                         "reason": f"Ambient {mean_lux:,.0f} lux, visibility {mean_visibility:.1f} km — keep the circuit energised."}
            else:
                recommendations[slug] = {"state": "off", "mode": "auto",
                                         "reason": f"Ambient {mean_lux:,.0f} lux — daylight circuit can rest."}
        elif kind == "access" or "barrier" in slug:
            if patrol_score >= 65:
                recommendations[slug] = {"state": "on", "mode": "manual",
                                         "reason": f"Patrol priority {patrol_score}/100 — keep depot access open for dispatch."}
            else:
                recommendations[slug] = {"state": "off", "mode": "manual",
                                         "reason": "No patrol dispatch recommended; keep the barrier closed."}
        else:
            recommendations[slug] = {"state": str(device.get("state", "off")), "mode": str(device.get("mode", "manual")),
                                     "reason": "No rule matched; hold the current state."}
    return recommendations


def triage_image(payload: bytes) -> dict[str, object]:
    """Heuristic scene triage for an uploaded frame. Not a trained classifier."""
    try:
        image = Image.open(BytesIO(payload))
        image.load()
    except Exception as error:  # Pillow raises several unrelated types for corrupt uploads
        raise ValueError("Unsupported or corrupt image file.") from error

    width, height = image.size
    grey = image.convert("L")
    grey.thumbnail((192, 192))
    bordered = grey.filter(ImageFilter.FIND_EDGES)
    # PIL treats out-of-frame pixels as black, so the outer ring always lights up; drop it.
    if bordered.size[0] > 8 and bordered.size[1] > 8:
        inner_width, inner_height = bordered.size
        bordered = bordered.crop((2, 2, inner_width - 2, inner_height - 2))
    stats = ImageStat.Stat(grey)
    brightness = stats.mean[0] / 255
    contrast = min(stats.stddev[0] / 96, 1.0)
    edge = min(ImageStat.Stat(bordered).mean[0] / 60, 1.0)
    histogram = grey.histogram()
    pixels = max(sum(histogram), 1)
    dark_share = sum(histogram[:60]) / pixels
    colourful = image.convert("RGB").resize((64, 64)).convert("HSV")
    saturation = ImageStat.Stat(colourful).mean[1] / 255

    exposure_gap = 1 - min(brightness / 0.42, 1.0)
    clutter = min(edge / 0.22, 1.0)
    flatness = 1 - min(contrast, 1.0)
    score = 100 * (0.44 * exposure_gap + 0.20 * flatness + 0.26 * clutter + 0.10 * dark_share)
    score = round(max(0.0, min(score, 100.0)), 1)
    if score >= 70:
        band, tone = "High", "bad"
    elif score >= 45:
        band, tone = "Moderate", "warn"
    else:
        band, tone = "Low", "good"
    confidence = round(55 + 45 * abs(score - 50) / 50, 1)

    findings: list[str] = []
    checklist: list[str] = ["Confirm the location and time against the operator log before filing the report."]
    if brightness < 0.18:
        findings.append("Low illumination — possible night, tunnel, or underexposed capture.")
        checklist.append("Request a re-capture with headlights or additional lighting.")
    if contrast < 0.18:
        findings.append("Very low contrast — possible fog, glare, or a rain-covered lens.")
        checklist.append("Clean the lens or wait for a clearer frame before classifying.")
    if edge > 0.55:
        findings.append("Dense edge detail — busy scene or sensor noise.")
        checklist.append("Zoom into the lane markings and stop line to verify the scene.")
    if dark_share > 0.55:
        findings.append("More than half the frame is dark — check the lens cover or capture time.")
    if saturation < 0.08:
        findings.append("Near-monochrome frame — dust, snow, or a low-light sensor.")
    if not findings:
        findings.append("Frame is well exposed with ordinary detail — no automated red flags.")

    checklist.append("Record the operator-observed category; image triage never files an incident on its own.")
    checklist.append("Keep the stored file free of faces and licence plates; this demo stores only the file name.")

    return {
        "risk": band,
        "tone": tone,
        "score": score,
        "confidence": confidence,
        "metrics": {
            "brightness": round(brightness, 3),
            "contrast": round(contrast, 3),
            "edge_density": round(edge, 3),
            "dark_share": round(dark_share, 3),
            "saturation": round(saturation, 3),
        },
        "findings": findings,
        "checklist": checklist,
        "signature": sha256(payload).hexdigest()[:12],
        "width": width,
        "height": height,
        "pixels": width * height,
    }


def _number(value: object, fallback: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
