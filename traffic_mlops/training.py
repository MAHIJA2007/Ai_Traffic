from __future__ import annotations

import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from traffic_mlops.config import EXPERIMENT_NAME, MODEL_NAME, configure_mlflow
from traffic_mlops.features import FEATURE_COLUMNS, NUMERIC_FEATURES, build_features


def train_model(source: pd.DataFrame, experiment_name: str = EXPERIMENT_NAME) -> dict[str, object]:
    """Train, evaluate, log, register, and conditionally promote a model."""
    configure_mlflow()
    features = build_features(source)
    timestamps = np.sort(features["timestamp"].unique())
    if len(timestamps) < 3:
        raise ValueError("At least three distinct feature timestamps are required for a chronological split.")

    split_at = max(1, min(len(timestamps) - 1, int(len(timestamps) * 0.8)))
    cutoff = timestamps[split_at]
    train_rows = features[features["timestamp"] < cutoff]
    test_rows = features[features["timestamp"] >= cutoff]
    if train_rows.empty or test_rows.empty:
        raise ValueError("Not enough history for a chronological train/test split.")

    x_train = train_rows[FEATURE_COLUMNS]
    y_train = train_rows["target_next_hour"]
    x_test = test_rows[FEATURE_COLUMNS]
    y_test = test_rows["target_next_hour"]
    transformer = ColumnTransformer(
        [
            ("junction", OneHotEncoder(handle_unknown="ignore"), ["junction"]),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = Pipeline(
        [
            ("features", transformer),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=160,
                    min_samples_leaf=2,
                    max_features=0.9,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    mlflow.set_experiment(experiment_name)
    client = MlflowClient()
    champion_rmse: float | None = None
    previous_champion_dataset: str | None = None
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
        champion_run = client.get_run(champion.run_id)
        previous_champion_dataset = champion_run.data.tags.get("dataset")
        if previous_champion_dataset == "Kaggle Traffic Prediction Dataset (4 junctions)":
            champion_rmse = champion_run.data.metrics.get("rmse")
    except Exception:
        pass

    with mlflow.start_run(run_name="random-forest-traffic") as run:
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "r2": float(r2_score(y_test, predictions)),
        }
        mlflow.log_params(
            {
                "model_type": "RandomForestRegressor",
                "n_estimators": 160,
                "min_samples_leaf": 2,
                "feature_count": len(FEATURE_COLUMNS),
                "junction_count": int(features["junction"].nunique()),
                "split": "chronological_80_20",
                "training_rows": len(train_rows),
                "test_rows": len(test_rows),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "project_number": "28",
                "project_title": "AI-Based Traffic Prediction System with MLOps",
                "team": "K.Mahija; Sk.Anjum; P.Hasitha",
                "dataset": "Kaggle Traffic Prediction Dataset (4 junctions)",
                "dataset_source": "fedesoriano/traffic-prediction-dataset",
                "target": "vehicles_next_hour",
            }
        )
        signature = infer_signature(x_train, model.predict(x_train.head(5)))
        logged_model = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=x_train.head(3),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        run_id = run.info.run_id

    registered = mlflow.register_model(logged_model.model_uri, MODEL_NAME)
    promoted = champion_rmse is None or metrics["rmse"] < champion_rmse
    edge_cache_ready = False
    if promoted:
        client.set_registered_model_alias(MODEL_NAME, "champion", registered.version)
        try:
            from traffic_mlops.prediction import cache_champion_model

            cache_champion_model()
            edge_cache_ready = True
        except Exception as error:
            warnings.warn(f"Champion registered, but local edge cache could not be refreshed: {error}", stacklevel=2)
    client.set_model_version_tag(
        MODEL_NAME,
        registered.version,
        "validation_rmse",
        f"{metrics['rmse']:.6f}",
    )
    client.set_model_version_tag(
        MODEL_NAME,
        registered.version,
        "deployment_status",
        "champion" if promoted else "candidate",
    )
    return {
        "run_id": run_id,
        "model_name": MODEL_NAME,
        "model_version": str(registered.version),
        "promoted": promoted,
        "edge_cache_ready": edge_cache_ready,
        "previous_champion_rmse": champion_rmse,
        "previous_champion_dataset": previous_champion_dataset,
        "metrics": metrics,
        "training_rows": len(train_rows),
        "test_rows": len(test_rows),
    }
