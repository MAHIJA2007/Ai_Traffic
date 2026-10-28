"""Prepare the downloaded four-junction Kaggle CSV for this project's schema."""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ARCHIVE = ROOT / "data" / "raw" / "kaggle_junction_traffic.zip"
OUTPUT = ROOT / "data" / "traffic.csv"


def main() -> None:
    if not SOURCE_ARCHIVE.exists():
        raise FileNotFoundError(f"Place the Kaggle traffic-prediction-dataset archive at {SOURCE_ARCHIVE}")
    with zipfile.ZipFile(SOURCE_ARCHIVE) as archive:
        member = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        with archive.open(member) as source:
            raw = pd.read_csv(source)

    required = {"DateTime", "Junction", "Vehicles"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Expected Kaggle fields {sorted(required)}; received {list(raw.columns)}")
    prepared = pd.DataFrame({
        "timestamp": pd.to_datetime(raw["DateTime"], errors="coerce"),
        "junction": "Junction " + raw["Junction"].astype("string").str.strip(),
        "vehicles": pd.to_numeric(raw["Vehicles"], errors="coerce"),
    }).dropna()
    prepared = prepared[prepared["junction"].ne("Junction <NA>")]
    prepared = prepared.drop_duplicates(["timestamp", "junction"], keep="last")
    prepared = prepared.sort_values(["junction", "timestamp"]).reset_index(drop=True)
    gaps = prepared.groupby("junction")["timestamp"].diff().dropna()
    if not gaps.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("The source has missing or irregular hourly observations; inspect before training.")
    prepared.to_csv(OUTPUT, index=False)
    print(f"Prepared {len(prepared):,} hourly observations for {prepared['junction'].nunique()} junctions: {OUTPUT}")


if __name__ == "__main__":
    main()
