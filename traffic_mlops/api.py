from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from traffic_mlops.config import DEFAULT_DATASET, MODEL_NAME, configure_mlflow
from traffic_mlops.data import generate_demo_data
from traffic_mlops.monitoring import monitor_data
from traffic_mlops.history import recent_predictions, recent_virtual_alerts, recent_virtual_readings, save_prediction, save_virtual_tick
from traffic_mlops.prediction import predict_horizon, predict_next_hour
from traffic_mlops.training import champion_summary, train_model
from traffic_mlops.simulation import generate_virtual_tick


app = FastAPI(
    title="AI-Based Traffic Prediction API",
    version="1.0.0",
    description="Hourly junction-level traffic forecasts with MLflow model tracking and monitoring.",
)


class PredictionRequest(BaseModel):
    junction: str = Field(min_length=1)
    vehicles: float | None = Field(default=None, ge=0)
    temperature_c: float | None = None
    is_event: int | None = Field(default=None, ge=0, le=1)
    offline: bool = False


class SimulationTickRequest(BaseModel):
    cloud_online: bool = True
    failed_junction: str | None = None


class HorizonRequest(BaseModel):
    junction: str = Field(min_length=1)
    hours: int = Field(default=24, ge=1, le=48)
    offline: bool = False


def read_dataset() -> pd.DataFrame:
    if not Path(DEFAULT_DATASET).exists():
        DEFAULT_DATASET.parent.mkdir(parents=True, exist_ok=True)
        generate_demo_data().to_csv(DEFAULT_DATASET, index=False)
    return pd.read_csv(DEFAULT_DATASET)


@app.get("/health")
def health() -> dict[str, object]:
    configure_mlflow()
    try:
        from mlflow.tracking import MlflowClient

        MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
        model_ready = True
    except Exception:
        from traffic_mlops.prediction import EDGE_CACHE

        model_ready = (EDGE_CACHE / "champion.json").exists()
    return {"status": "ok", "model_ready": model_ready}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, object]:
    try:
        result = predict_next_hour(
            read_dataset(),
            request.junction,
            request.vehicles,
            request.temperature_c,
            request.is_event,
            offline=request.offline,
        )
        result["prediction_id"] = save_prediction(result, request.model_dump())
        return result
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail="No champion model is available. Train a model first.") from error


@app.post("/train")
def train() -> dict[str, object]:
    try:
        return train_model(read_dataset())
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/model")
def model_info() -> dict[str, object]:
    try:
        summary = champion_summary()
    except Exception as error:
        raise HTTPException(status_code=503, detail="The model registry is unavailable.") from error
    return {"ready": summary is not None, "champion": summary}


@app.post("/forecast/horizon")
def forecast_horizon(request: HorizonRequest) -> dict[str, object]:
    """Recursive next-N-hour outlook for one junction."""
    try:
        return predict_horizon(
            read_dataset(), request.junction, request.hours, offline=request.offline
        )
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail="No champion model is available. Train a model first.") from error


@app.get("/monitor")
def monitor() -> dict[str, object]:
    try:
        return monitor_data(read_dataset())
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/predictions")
def predictions(limit: int = 100) -> dict[str, object]:
    return {"items": recent_predictions(limit)}


@app.post("/simulation/tick")
def simulation_tick(request: SimulationTickRequest) -> dict[str, object]:
    """Generate a software-only virtual sensor/network/weather tick."""
    history = recent_virtual_readings(2000)
    tick = max((int(item.get("tick", 0)) for item in history), default=0) + 1
    readings, alerts = generate_virtual_tick(
        read_dataset(), tick, cloud_online=request.cloud_online,
        failed_junction=request.failed_junction,
        model_ready=bool(health()["model_ready"]),
    )
    save_virtual_tick(tick, readings, alerts)
    return {"tick": tick, "virtual": True, "readings": readings, "alerts": alerts}


@app.get("/simulation/telemetry")
def simulation_telemetry(limit: int = 100) -> dict[str, object]:
    return {"items": recent_virtual_readings(limit)}


@app.get("/simulation/alerts")
def simulation_alerts(limit: int = 100) -> dict[str, object]:
    return {"items": recent_virtual_alerts(limit)}
