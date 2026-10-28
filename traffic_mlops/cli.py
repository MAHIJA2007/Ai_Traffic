import argparse
import json
from pathlib import Path

import pandas as pd

from traffic_mlops.config import DEFAULT_DATASET
from traffic_mlops.data import generate_demo_data
from traffic_mlops.monitoring import monitor_data
from traffic_mlops.training import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Traffic prediction MLOps commands")
    parser.add_argument("command", choices=("bootstrap", "generate-data", "train", "monitor"))
    parser.add_argument("--data", help="Optional traffic CSV path")
    args = parser.parse_args()
    dataset = args.data or str(DEFAULT_DATASET)

    if args.command == "generate-data" or (args.command == "bootstrap" and not Path(dataset).exists()):
        Path(dataset).parent.mkdir(parents=True, exist_ok=True)
        frame = generate_demo_data()
        frame.to_csv(dataset, index=False)
        print(f"Generated {len(frame):,} rows at {dataset}")
        if args.command == "generate-data":
            return

    frame = pd.read_csv(dataset)

    result = train_model(frame) if args.command in ("bootstrap", "train") else monitor_data(frame)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
