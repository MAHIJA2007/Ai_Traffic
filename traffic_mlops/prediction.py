from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import mlflow.pyfunc

from traffic_mlops.config import MODEL_NAME, PROJECT_ROOT, configure_mlflow
from traffic_mlops.data import normalize_data
from traffic_mlops.features import FEATURE_COLUMNS, build_features


EDGE_CACHE = PROJECT_ROOT / "edge_cache"


def cache_champion_model() -> str:
    """Copy the current champion artifact to durable host-local edge storage."""
    configure_mlflow()
    from mlflow.tracking import MlflowClient

    version = str(MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion").version)
    cache_path = EDGE_CACHE / f"version-{version}"
    if not cache_path.exists():
        downloaded = Path(mlflow.artifacts.download_artifacts(f"models:/{MODEL_NAME}@champion"))
        EDGE_CACHE.mkdir(parents=True, exist_ok=True)
        shutil.copytree(downloaded, cache_path)
    metadata = EDGE_CACHE / "champion.json"
    metadata.write_text(json.dumps({"version": version, "path": cache_path.name}, indent=2), encoding="utf-8")
    return str(cache_path)


def load_champion(offline: bool = False) -> mlflow.pyfunc.PyFuncModel:
    """Load the registry champion, or use the saved local copy in offline mode."""
    if offline:
        metadata_path = EDGE_CACHE / "champion.json"
        if not metadata_path.exists():
            raise RuntimeError("No local edge model is cached. Connect to MLflow and train/load the champion first.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        local_path = EDGE_CACHE / metadata["path"]
        if not local_path.exists():
            raise RuntimeError("The edge model cache is incomplete. Reconnect to MLflow and refresh the champion cache.")
        return mlflow.pyfunc.load_model(str(local_path))
    configure_mlflow()
    try:
        return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
    except Exception:
        return load_champion(offline=True)


def predict_next_hour(
    source: pd.DataFrame,
    junction: str,
    vehicles: float | None = None,
    temperature_c: float | None = None,
    is_event: int | None = None,
    offline: bool = False,
) -> dict[str, object]:
    """Forecast the selected junction one hour after its latest observation."""
    frame = normalize_data(source)
    history = frame[frame["junction"] == junction]
    if history.empty:
        raise ValueError(f"No observations found for junction '{junction}'.")

    latest = history.iloc[-1]
    forecast_time = latest["timestamp"] + pd.Timedelta(hours=1)
    next_row = {
        "timestamp": forecast_time,
        "junction": junction,
        "vehicles": float(latest["vehicles"] if vehicles is None else vehicles),
        "temperature_c": float(latest["temperature_c"] if temperature_c is None else temperature_c),
        "is_event": int(latest["is_event"] if is_event is None else is_event),
    }
    augmented = pd.concat([frame, pd.DataFrame([next_row])], ignore_index=True)
    feature_row = build_features(augmented, require_target=False)
    feature_row = feature_row[feature_row["timestamp"] == forecast_time]
    if feature_row.empty:
        raise ValueError("Insufficient junction history to build lag features; provide at least one week of hourly data.")

    model = load_champion(offline=offline)
    prediction = float(model.predict(feature_row[FEATURE_COLUMNS].tail(1))[0])
    return {
        "junction": junction,
        "forecast_for": forecast_time.isoformat(),
        "predicted_vehicles": max(0, round(prediction, 1)),
        "input_vehicles": next_row["vehicles"],
    }
