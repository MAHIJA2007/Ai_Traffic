import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from traffic_mlops.data import generate_demo_data, normalize_data
from traffic_mlops.features import FEATURE_COLUMNS, NUMERIC_FEATURES, build_features
from traffic_mlops.monitoring import population_stability_index
from traffic_mlops.prediction import predict_horizon


def test_demo_data_builds_time_series_features():
    data = generate_demo_data(days=10, junctions=("A", "B"))
    features = build_features(data)

    assert len(data) == 10 * 24 * 2
    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert "target_next_hour" in features.columns
    assert features["timestamp"].min() > data["timestamp"].min()
    assert features["junction"].nunique() == 2


def test_normalize_data_checks_schema_and_deduplicates():
    data = pd.DataFrame(
        {
            "timestamp": ["2025-01-01 00:00", "2025-01-01 00:00"],
            "junction": ["A", "A"],
            "vehicles": [20, 25],
        }
    )

    normalized = normalize_data(data)
    assert len(normalized) == 1
    assert normalized.iloc[0]["vehicles"] == 25
    assert normalized.iloc[0]["temperature_c"] == 20


def test_psi_is_near_zero_for_matching_data_and_rises_after_shift():
    baseline = pd.Series(range(100))
    same = population_stability_index(baseline, baseline)
    shifted = population_stability_index(baseline, baseline + 100)

    assert same < 0.001
    assert shifted > same


def _tiny_model(data: pd.DataFrame) -> Pipeline:
    features = build_features(data)
    transformer = ColumnTransformer(
        [
            ("junction", OneHotEncoder(handle_unknown="ignore"), ["junction"]),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = Pipeline(
        [("features", transformer), ("regressor", RandomForestRegressor(n_estimators=12, max_depth=8, random_state=0))]
    )
    model.fit(features[FEATURE_COLUMNS], features["target_next_hour"])
    return model


def test_predict_horizon_rolls_forward_with_ordered_timestamps():
    data = generate_demo_data(days=12, junctions=("A", "B"))
    result = predict_horizon(data, "A", hours=4, model=_tiny_model(data))

    stamps = pd.to_datetime([point["timestamp"] for point in result["points"]])
    last_observed = data[data["junction"] == "A"]["timestamp"].max()

    assert len(result["points"]) == 4
    assert list(stamps) == sorted(stamps)
    assert stamps.min() == last_observed + pd.Timedelta(hours=1)
    assert all(point["predicted_vehicles"] >= 0 for point in result["points"])
    assert result["peak"]["predicted_vehicles"] == max(point["predicted_vehicles"] for point in result["points"])


def test_predict_horizon_rejects_out_of_range_hours():
    data = generate_demo_data(days=8, junctions=("A",))
    with pytest.raises(ValueError):
        predict_horizon(data, "A", hours=0, model=object())