# register model

import json
import logging
import os

import mlflow

mlflow.set_tracking_uri(
    "http://ec2-3-107-8-28.ap-southeast-2.compute.amazonaws.com:5000/"
)


# logging configuration
logger = logging.getLogger("model_registration")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("model_registration_errors.log")
file_handler.setLevel("ERROR")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, "r") as file:
            model_info = json.load(file)
        logger.debug(f"Model info loaded from {file_path}")
        return model_info
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


def register_model(model_name: str, model_info: dict) -> None:
    """Register the model to the MLflow Registry"""
    try:
        # model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        ## i found the below model_uri from MLflow UI , steps below:
        # 1. go to MLflow UI
        # 2. click on the experiments
        # 3. click on the run name
        # 4. click on the artifacts
        # 5. click on the eg. conda.yaml
        # 6. You should see something like:
        # Path: s3://mlflow-bucket-ben-28/445780057568027865/models/m-327cca1283f14fdc95faef84bc5baa6b/artifacts/conda.yaml
        # The model_uri is: s3://mlflow-bucket-ben-28/445780057568027865/models/m-327cca1283f14fdc95faef84bc5baa6b/artifacts/

        model_uri = "s3://mlflow-bucket-ben-28/445780057568027865/models/m-d7514ed922c947c3937537430a05faed/artifacts/"
        model_version = mlflow.register_model(model_uri, model_name)

        # transition the model to Staging stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging",
        )

        logger.debug(
            f"Model {model_name} version {model_version.version} registered and transitioned to Staging stage"
        )

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


def main() -> None:
    try:
        model_info_path = "experiment_info.json"
        model_info = load_model_info(model_info_path)

        model_name = "yt_chrome_plugin_model"

        register_model(model_name, model_info)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
