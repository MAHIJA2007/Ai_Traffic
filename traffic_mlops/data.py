from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"timestamp", "junction", "vehicles"}


def normalize_data(source: pd.DataFrame | str | Path) -> pd.DataFrame:
    """Validate and normalize the supported traffic CSV schema."""
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    else:
        frame = pd.read_csv(source)

    frame.columns = [str(column).strip().lower() for column in frame.columns]
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["junction"] = frame["junction"].astype("string").str.strip()
    frame["vehicles"] = pd.to_numeric(frame["vehicles"], errors="coerce")
    if "temperature_c" not in frame:
        frame["temperature_c"] = 20.0
    frame["temperature_c"] = pd.to_numeric(frame["temperature_c"], errors="coerce").fillna(20.0)
    if "is_event" not in frame:
        frame["is_event"] = 0
    frame["is_event"] = pd.to_numeric(frame["is_event"], errors="coerce").fillna(0).clip(0, 1)

    frame = frame.dropna(subset=["timestamp", "junction", "vehicles"])
    frame = frame[frame["junction"] != ""]
    frame["vehicles"] = frame["vehicles"].clip(lower=0)
    frame = frame.drop_duplicates(["timestamp", "junction"], keep="last")
    return frame.sort_values(["junction", "timestamp"]).reset_index(drop=True)


def generate_demo_data(
    days: int = 120,
    junctions: tuple[str, ...] = ("Central", "University", "Market", "Airport"),
    seed: int = 42,
) -> pd.DataFrame:
    """Create reproducible hourly traffic observations for a working demo."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-01", periods=days * 24, freq="h")
    junction_factors = {name: 0.8 + index * 0.16 for index, name in enumerate(junctions)}
    rows: list[dict[str, object]] = []

    for junction, factor in junction_factors.items():
        for timestamp in timestamps:
            hour = timestamp.hour
            weekday = timestamp.day_of_week < 5
            morning_peak = np.exp(-((hour - 8) / 2.2) ** 2)
            evening_peak = np.exp(-((hour - 17) / 2.8) ** 2)
            daytime = 0.25 if 0 <= hour < 5 else 0.65
            weekly = 0.82 if not weekday else 1.0
            event = int(rng.random() < 0.025)
            expected = factor * weekly * (30 + 105 * morning_peak + 125 * evening_peak + 65 * daytime)
            expected *= 1.0 + 0.22 * event
            vehicles = max(0, int(expected + rng.normal(0, 9)))
            temperature = 19 + 8 * np.sin(2 * np.pi * timestamp.day_of_year / 365) + rng.normal(0, 2)
            rows.append(
                {
                    "timestamp": timestamp,
                    "junction": junction,
                    "vehicles": vehicles,
                    "temperature_c": round(float(temperature), 1),
                    "is_event": event,
                }
            )

    return pd.DataFrame(rows)