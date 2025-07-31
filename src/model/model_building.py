"""
Model Building Script for YouTube Comment Analysis (MLOps Pipeline)

This script performs the model training stage of the MLOps pipeline. It is intended to be used as a DVC pipeline stage and includes the following steps:

1. Loads model hyperparameters and configuration from params.yaml.
2. Loads preprocessed training data from the data/interim directory.
3. Applies TF-IDF vectorization to the 'clean_comment' column using scikit-learn's TfidfVectorizer.
4. Trains a LightGBM classifier for multiclass classification using the vectorized features.
5. Saves the trained model (lgbm_model.pkl) and the fitted TF-IDF vectorizer (tfidf_vectorizer.pkl) to the project root.
6. Logs all operations and errors for debugging and reproducibility.

Logging:
- DEBUG-level logs for process tracing and successful operations.
- ERROR-level logs for exceptions and failures.

Typical usage:
    python src/model/model_building.py

This script is designed for modularity, reproducibility, and integration into automated ML pipelines.
"""

import logging
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

# logging configs
logger = logging.getLogger("model_building")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("model_building_errors.log")
file_handler.setLevel("ERROR")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logger.debug(f"Parameters retrived from {params_path}")
        return params
    except FileNotFoundError:
        logger.error(f"File not found: {params_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_path)
        df.fillna("", inplace=True)
        logger.debug(f"Data loaded and NaNs filled from {data_path}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse the CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occured while loading the data: {e}")
        raise


def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "../../"))


def apply_tfidf(
    train_data: pd.DataFrame, max_features: int, ngram_range: tuple
) -> tuple:
    """Apply TF-IDF vectorization to the training data."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

        X_train = train_data["clean_comment"].values
        y_train = train_data["category"].values

        # perform TF-IDF transformation
        X_train_tfidf = vectorizer.fit_transform(X_train)
        logger.debug(
            f"TF-IDF transformation complete. Train shape: {X_train_tfidf.shape}"
        )

        # save the vectorizer in the root directory
        with open(
            os.path.join(get_root_directory(), "tfidf_vectorizer.pkl"), "wb"
        ) as f:
            pickle.dump(vectorizer, f)

        logger.debug("TF-IDF aplpied with trigrams and data transformed.")
        return X_train_tfidf, y_train
    except Exception as e:
        logger.error(f"Unexpected error occured while applying TF-IDF: {e}")
        raise


def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    learning_rate: float,
    max_depth: int,
    n_estimators: int,
) -> lgb.LGBMClassifier:
    """Train a LightGBM classifier."""
    try:
        lgbm = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            metric="multi_logloss",
            is_unbalance=True,
            class_weight="balanced",
            reg_alpha=0.1,
            reg_lambda=0.1,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
        )
        lgbm.fit(X_train, y_train)
        logger.debug("LightGBM classifier trained successfully.")
        return lgbm
    except Exception as e:
        logger.error(f"Unexpected error occured while training LightGBM: {e}")
        raise


def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, "wb") as file:
            pickle.dump(model, file)
        logger.debug(f"Model saved to {file_path}")
    except Exception as e:
        logger.error(f"Unexpected error occured while saving the model: {e}")
        raise


def main() -> None:
    try:
        # get root directory and resolve the path for params.yaml
        root_dir = get_root_directory()

        # load params from the root directory
        params = load_params(os.path.join(root_dir, "params.yaml"))
        max_features = params["model_building"]["max_features"]
        ngram_range = tuple(params["model_building"]["ngram_range"])
        learning_rate = params["model_building"]["learning_rate"]
        max_depth = params["model_building"]["max_depth"]
        n_estimators = params["model_building"]["n_estimators"]

        # load the preprocessed training data from the interim directory
        train_data = load_data(
            os.path.join(root_dir, "data/interim/train_processed.csv")
        )

        # apply tfidf
        X_train_tfidf, y_train = apply_tfidf(train_data, max_features, ngram_range)

        # train the LightGBM model
        best_model = train_lgbm(
            X_train_tfidf, y_train, learning_rate, max_depth, n_estimators
        )

        # save the trained model in the root direcotry
        save_model(best_model, os.path.join(root_dir, "lgbm_model.pkl"))
    except Exception as e:
        logger.error(f"Unexpected error occured while running the model building: {e}")
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
