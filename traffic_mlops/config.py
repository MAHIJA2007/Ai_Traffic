import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATASET = DATA_DIR / "traffic.csv"
MODEL_NAME = "TrafficFlowPredictor"
EXPERIMENT_NAME = "traffic-prediction"
MLFLOW_DB = PROJECT_ROOT / "mlflow_project28.db"
ARTIFACT_DIR = PROJECT_ROOT / "mlartifacts" / "project28"


def configure_mlflow() -> str:
    """Configure a persistent local tracking store unless one is provided."""
    import mlflow

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    # An explicitly supplied URI always wins. DagsHub can also be enabled with
    # DAGSHUB_REPO_OWNER / DAGSHUB_REPO_NAME and an optional access token.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri and os.getenv("DAGSHUB_REPO_OWNER") and os.getenv("DAGSHUB_REPO_NAME"):
        try:
            import dagshub
        except ImportError as error:
            raise RuntimeError("Install the optional DagsHub integration with: pip install -r requirements-dagshub.txt") from error
        dagshub.init(
            repo_owner=os.environ["DAGSHUB_REPO_OWNER"],
            repo_name=os.environ["DAGSHUB_REPO_NAME"],
            mlflow=True,
        )
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        tracking_uri = f"sqlite:///{MLFLOW_DB.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    if tracking_uri.startswith("sqlite:"):
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
            client.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACT_DIR.resolve().as_uri())
    return tracking_uri
