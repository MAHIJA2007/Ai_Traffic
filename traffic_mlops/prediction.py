from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn

from traffic_mlops.config import EDGE_CACHE_DIR, MODEL_NAME, configure_mlflow
from traffic_mlops.data import normalize_data
from traffic_mlops.features import FEATURE_COLUMNS, build_features


EDGE_CACHE = EDGE_CACHE_DIR

# Types used by the logged sklearn pipeline; skops refuses to load anything else.
TRUSTED_ARTIFACT_TYPES = [
    "sklearn.tree._tree.Tree",
    "sklearn.pipeline.Pipeline",
    "sklearn.compose._column_transformer.ColumnTransformer",
    "sklearn.preprocessing._encoders.OneHotEncoder",
    "sklearn.ensemble._forest.RandomForestRegressor",
    "sklearn.utils._bunch.Bunch",
    "numpy.ndarray",
    "numpy.dtype",
    "builtins.float",
    "builtins.int",
    "builtins.str",
    "builtins.list",
    "builtins.tuple",
    "builtins.dict",
    "builtins.bool",
    "builtins.bytes",
]


def _single_threaded(model: object) -> object:
    regressor = getattr(model, "named_steps", {}).get("regressor")
    if regressor is not None and hasattr(regressor, "set_params"):
        # Thread-pool startup dominates one-row inference; serving stays single-threaded.
        regressor.set_params(n_jobs=1)
    return model


@lru_cache(maxsize=4)
def _load_pipeline(path: str):
    target = Path(path)
    if target.is_dir() and (target / "model.skops").exists():
        target = target / "model.skops"
    try:
        import skops.io

        model = skops.io.load(str(target), trusted=TRUSTED_ARTIFACT_TYPES)
    except Exception:
        model = mlflow.sklearn.load_model(path)
    return _single_threaded(model)


def _edge_cache_entry() -> tuple[str, Path] | None:
    metadata_path = EDGE_CACHE / "champion.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        local_path = EDGE_CACHE / str(metadata["path"])
        return str(metadata["version"]), local_path
    except (OSError, KeyError, ValueError):
        return None


def _registry_version() -> str | None:
    try:
        configure_mlflow()
        from mlflow.tracking import MlflowClient

        return str(MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion").version)
    except Exception:
        return None


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
    _load_pipeline.cache_clear()
    return str(cache_path)


def load_champion(offline: bool = False):
    """Load the registry champion, or the saved local copy in offline mode."""
    edge = _edge_cache_entry()
    if offline:
        if edge is None:
            raise RuntimeError("No local edge model is cached. Connect to MLflow and train/load the champion first.")
        return _load_pipeline(str(edge[1]))

    version = _registry_version()
    if edge is not None and (version is None or edge[0] == version):
        return _load_pipeline(str(edge[1]))
    try:
        return _single_threaded(mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion"))
    except Exception:
        if edge is not None:
            return _load_pipeline(str(edge[1]))
        raise RuntimeError("No champion model is available. Train a model first.")


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


def predict_horizon(
    source: pd.DataFrame,
    junction: str,
    hours: int = 24,
    offline: bool = False,
    model: object | None = None,
) -> dict[str, object]:
    """Roll the champion model forward hour by hour for a 24-hour demand outlook."""
    if not 1 <= int(hours) <= 48:
        raise ValueError("The outlook horizon must be between 1 and 48 hours.")
    frame = normalize_data(source)
    history = frame[frame["junction"] == junction].sort_values("timestamp")
    if history.empty:
        raise ValueError(f"No observations found for junction '{junction}'.")
    if len(history) < 168:
        raise ValueError("At least one week (168 hours) of hourly history is required for the outlook.")

    window = history.tail(220).copy()
    latest = history.iloc[-1]
    temperature = float(latest["temperature_c"])
    is_event = int(latest["is_event"])
    state_time = pd.Timestamp(latest["timestamp"]) + pd.Timedelta(hours=1)
    last_volume = float(latest["vehicles"])
    active_model = model if model is not None else load_champion(offline=offline)

    points: list[dict[str, object]] = []
    for _ in range(int(hours)):
        window = pd.concat(
            [
                window,
                pd.DataFrame(
                    [
                        {
                            "timestamp": state_time,
                            "junction": junction,
                            "vehicles": last_volume,
                            "temperature_c": temperature,
                            "is_event": is_event,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        feature_row = build_features(window, require_target=False)
        feature_row = feature_row[feature_row["timestamp"] == state_time]
        if feature_row.empty:
            raise ValueError("Insufficient history to build lag features for the outlook.")
        prediction = max(0.0, float(active_model.predict(feature_row[FEATURE_COLUMNS].tail(1))[0]))
        points.append({"timestamp": state_time.isoformat(), "predicted_vehicles": round(prediction, 1)})
        last_volume = prediction
        state_time = state_time + pd.Timedelta(hours=1)

    peak = max(points, key=lambda point: point["predicted_vehicles"])
    mean_predicted = float(pd.Series([point["predicted_vehicles"] for point in points]).mean())
    return {
        "junction": junction,
        "hours": int(hours),
        "points": points,
        "peak": peak,
        "mean_predicted": round(mean_predicted, 1),
        "baseline_mean_24h": round(float(history.tail(24)["vehicles"].mean()), 1),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "offline": offline,
    }
