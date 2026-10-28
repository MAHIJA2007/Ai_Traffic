import pandas as pd

from traffic_mlops.data import generate_demo_data, normalize_data
from traffic_mlops.features import FEATURE_COLUMNS, build_features
from traffic_mlops.monitoring import population_stability_index


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