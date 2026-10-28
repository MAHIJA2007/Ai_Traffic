import numpy as np
import pandas as pd

from traffic_mlops.data import normalize_data


FEATURE_COLUMNS = [
    "junction",
    "vehicles",
    "temperature_c",
    "is_event",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
]
NUMERIC_FEATURES = [column for column in FEATURE_COLUMNS if column != "junction"]


def build_features(source: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Build next-hour supervised rows using history within each junction."""
    frame = normalize_data(source)
    grouped = frame.groupby("junction", sort=False)

    frame["hour_sin"] = np.sin(2 * np.pi * frame["timestamp"].dt.hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["timestamp"].dt.hour / 24)
    frame["weekday_sin"] = np.sin(2 * np.pi * frame["timestamp"].dt.day_of_week / 7)
    frame["weekday_cos"] = np.cos(2 * np.pi * frame["timestamp"].dt.day_of_week / 7)
    frame["month_sin"] = np.sin(2 * np.pi * (frame["timestamp"].dt.month - 1) / 12)
    frame["month_cos"] = np.cos(2 * np.pi * (frame["timestamp"].dt.month - 1) / 12)

    for lag in (1, 2, 3, 24, 168):
        frame[f"lag_{lag}"] = grouped["vehicles"].shift(lag)
    frame["rolling_mean_24"] = grouped["vehicles"].transform(
        lambda values: values.shift(1).rolling(24, min_periods=6).mean()
    )
    if require_target:
        frame["target_next_hour"] = grouped["vehicles"].shift(-1)

    required = FEATURE_COLUMNS + (["target_next_hour"] if require_target else [])
    frame = frame.dropna(subset=required).sort_values(["timestamp", "junction"]).reset_index(drop=True)
    frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].astype("float64")
    return frame