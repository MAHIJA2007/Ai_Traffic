from io import BytesIO

import pytest
from PIL import Image

from traffic_mlops.advisory import (
    alert_category,
    categorize_alerts,
    corridor_plans,
    device_recommendation,
    signal_plan,
    triage_image,
    weather_index,
    weather_suggestions,
)


def _reading(**overrides):
    reading = {
        "junction": "A",
        "sensor_status": "online",
        "vehicles_per_hour": 420.0,
        "average_speed_kmh": 52.0,
        "road_occupancy_pct": 45.0,
        "temperature_c": 22.0,
        "humidity_pct": 55.0,
        "rain_probability_pct": 10.0,
        "light_intensity_lux": 18000.0,
        "water_on_road_pct": 5.0,
        "wind_speed_kmph": 14.0,
        "visibility_km": 12.0,
        "cloud_link": "online",
    }
    reading.update(overrides)
    return reading


def _png(color):
    buffer = BytesIO()
    Image.new("RGB", (96, 72), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_alert_category_covers_every_rule_branch():
    assert alert_category("Congestion risk: simulated occupancy 88%") == "congestion"
    assert alert_category("Weather hazard risk: simulated rain probability 90%") == "weather"
    assert alert_category("Virtual sensor offline (fault injection)") == "sensor"
    assert alert_category("Cloud link down; local edge mode remains available") == "network"
    assert alert_category("Road closure reported by the patrol") == "safety"
    assert alert_category("Something unexpected happened") == "other"


def test_categorize_alerts_does_not_mutate_source_rows():
    source = [{"alert": "High wind advisory: 60 km/h simulated gusts", "severity": "warning"}]
    enriched = categorize_alerts(source)

    assert enriched[0]["category"] == "weather"
    assert "category" not in source[0]


def test_weather_index_bands_and_suggestions_react_to_conditions():
    clear = _reading()
    storm = _reading(rain_probability_pct=92, water_on_road_pct=70, wind_speed_kmph=66,
                     visibility_km=1.2, humidity_pct=92)

    assert weather_index(clear)["band"] == "Clear"
    assert weather_index(storm)["value"] > weather_index(clear)["value"]
    assert weather_index(storm)["tone"] == "bad"

    clear_labels = [item["label"] for item in weather_suggestions(clear)]
    storm_labels = [item["label"] for item in weather_suggestions(storm)]
    assert "Clear running conditions" in clear_labels
    assert "Wet-road protocol" in storm_labels
    assert "High-profile vehicle advisory" in storm_labels
    assert "Low-visibility procedure" in storm_labels


def test_signal_plan_extends_green_time_and_respects_override():
    quiet = signal_plan(20.0)
    busy = signal_plan(88.0)
    wet = signal_plan(30.0, weather_value=60.0)
    manual = signal_plan(88.0, override=90)

    assert quiet["green_seconds"] < busy["green_seconds"]
    assert busy["phase"] == "hold"
    assert quiet["phase"] == "release"
    assert wet["green_seconds"] > signal_plan(30.0)["green_seconds"]
    assert manual["green_seconds"] == 90
    assert manual["mode"] == "Manual"


def test_corridor_plans_are_ranked_and_bounded():
    congested = [_reading(junction=str(index), road_occupancy_pct=91, average_speed_kmh=18) for index in range(3)]
    plans = corridor_plans(congested)

    scores = [plan["score"] for plan in plans]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= score <= 100 for score in scores)
    assert plans[0]["name"] == "Adaptive signal retiming"
    assert corridor_plans([]) == []


def test_device_recommendation_reacts_to_load_and_light():
    devices = [
        {"slug": "adaptive-signal", "kind": "signal", "state": "on", "mode": "auto"},
        {"slug": "ramp-meter", "kind": "flow", "state": "off", "mode": "auto"},
        {"slug": "street-lighting", "kind": "lighting", "state": "off", "mode": "auto"},
        {"slug": "vms", "kind": "signage", "state": "on", "mode": "auto"},
    ]
    busy = device_recommendation(devices, [_reading(road_occupancy_pct=86, light_intensity_lux=200)])
    calm = device_recommendation(devices, [_reading(road_occupancy_pct=30, light_intensity_lux=22000)])

    assert busy["ramp-meter"]["state"] == "on"
    assert calm["ramp-meter"]["state"] == "off"
    assert busy["street-lighting"]["state"] == "on"
    assert calm["street-lighting"]["state"] == "off"
    assert device_recommendation(devices, []) == {}


def test_triage_image_flags_dark_frames_over_bright_ones():
    dark = triage_image(_png((6, 6, 6)))
    bright = triage_image(_png((236, 236, 236)))

    assert dark["score"] > bright["score"]
    assert dark["risk"] == "High"
    assert bright["risk"] == "Low"
    assert 0 <= dark["confidence"] <= 100
    assert dark["signature"] != bright["signature"]
    assert dark["checklist"]


def test_triage_image_rejects_non_image_bytes():
    with pytest.raises(ValueError):
        triage_image(b"not an image")
