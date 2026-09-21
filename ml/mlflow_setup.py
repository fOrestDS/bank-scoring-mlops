import os

import mlflow

def setup_mlflow():
    """Send metadata and artifacts through the same server on Windows and Docker."""
    # MLflow's URL messages contain emoji that crash legacy Windows consoles.
    os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    if not tracking_uri.startswith(("http://", "https://")):
        raise ValueError("MLFLOW_TRACKING_URI must point to the HTTP MLflow server")
    mlflow.set_tracking_uri(tracking_uri)
    # Existing bank-scoring runs retain their original local artifact locations.
    experiment = mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "bank-scoring-http"))
    if not experiment.artifact_location.startswith("mlflow-artifacts:/"):
        raise ValueError(
            "Experiment uses local artifact storage. Select a new MLFLOW_EXPERIMENT_NAME "
            "and start the server with --serve-artifacts --artifacts-destination."
        )
    return experiment
