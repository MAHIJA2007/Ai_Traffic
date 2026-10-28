from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd

from traffic_mlops.config import EXPERIMENT_NAME, configure_mlflow
from traffic_mlops.features import build_features


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Calculate PSI for a numeric feature using baseline-derived bins."""
    reference_values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    current_values = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if not len(reference_values) or not len(current_values):
        return 0.0
    edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(reference_values, bins=edges)[0].astype(float)
    cur_counts = np.histogram(current_values, bins=edges)[0].astype(float)
    ref_share = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_share = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def monitor_data(source: pd.DataFrame) -> dict[str, object]:
    """Compare recent traffic with its historical baseline and track drift."""
    configure_mlflow()
    features = build_features(source)
    timestamps = np.sort(features["timestamp"].unique())
    if len(timestamps) < 4:
        raise ValueError("At least four distinct feature timestamps are required for drift monitoring.")
    split_at = max(1, min(len(timestamps) - 1, int(len(timestamps) * 0.7)))
    baseline = features[features["timestamp"] < timestamps[split_at]]
    recent = features[features["timestamp"] >= timestamps[split_at]]
    monitored = ["vehicles", "temperature_c", "lag_1", "rolling_mean_24"]
    drift = {column: population_stability_index(baseline[column], recent[column]) for column in monitored}
    max_psi = max(drift.values(), default=0.0)
    status = "drift_detected" if max_psi >= 0.2 else "stable"

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="traffic-data-monitoring"):
        mlflow.log_metrics({f"psi_{name}": value for name, value in drift.items()})
        mlflow.log_metric("max_psi", max_psi)
        mlflow.set_tag("monitor_status", status)
        mlflow.set_tag("monitoring_window", "historical_70_percent_vs_recent_30_percent")

    return {
        "status": status,
        "threshold": 0.2,
        "max_psi": round(max_psi, 4),
        "features": {key: round(value, 4) for key, value in drift.items()},
        "baseline_rows": len(baseline),
        "recent_rows": len(recent),
        "recommendation": "Review the latest data and retrain if performance has also degraded." if status == "drift_detected" else "No material feature drift detected.",
    }