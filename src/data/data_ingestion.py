"""
Data Ingestion Script for YouTube Comment Analysis (MLOps Pipeline)

This script is responsible for the initial data ingestion step in the MLOps pipeline.
It performs the following tasks:

1. Loads configuration parameters from a YAML file (params.yaml).
2. Downloads raw data from a specified CSV URL.
3. Preprocesses the data by removing missing values, duplicates, and empty strings in the 'clean_comment' column.
4. Splits the cleaned data into training and test sets using scikit-learn.
5. Saves the resulting datasets to the 'data/raw/' directory, creating it if necessary.
6. Logs all operations and errors to dedicated log files for debugging and traceability.

Logging:
- DEBUG-level logs are used for successful operations and process tracing.
- ERROR-level logs are used for exceptions and unexpected failures.

Typical usage:
    python data_ingestion.py

This script is designed to be modular, robust, and reproducible for data science and machine learning workflows.
"""

import logging
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Logging configs
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.FileHandler("errors.log")
console_handler.setLevel(logging.ERROR)

file_handler = logging.FileHandler("error.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load params from a YAML file"""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logger.debug(f"Parameters retrieved from {params_path}")
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


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a csv file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug(f"Data loaded from {data_url}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse the CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occured while loading the data: {e}")
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by hanlding missing values, duplicates, and empty strings."""
    try:
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df = df[df["clean_comment"].str.strip() != ""]

        logger.debug(
            "Data preprocessing completed: Missing values, duplicates, and empty string removed."
        )
        return df
    except KeyError as e:
        logger.error(f"Missing column in the dataframe: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occured while preprocessing the data: {e}")
        raise


def save_data(
    train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str
) -> None:
    """Save the train and test datasets, creating the raw folder if it doesn't exist."""
    try:
        raw_data_path = os.path.join(data_path, "raw")

        os.makedirs(raw_data_path, exist_ok=True)

        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)

        logger.debug(f"Train and Test datasets saved to {raw_data_path}")
    except Exception as e:
        logger.error(f"Unexpected error occured while saving the data: {e}")
        raise


def main() -> None:
    try:
        # load params from the params.yaml in the root directory
        params = load_params(
            params_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "../../params.yaml"
            )
        )
        test_size = params["data_ingestion"]["test_size"]

        df = load_data(
            data_url="https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
        )
        final_df = preprocess_data(df)

        train_data, test_data = train_test_split(
            final_df, test_size=test_size, random_state=42
        )

        save_data(
            train_data,
            test_data,
            data_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "../../data"
            ),
        )
    except Exception as e:
        logger.error(f"Unexpected error occured while running the data ingestion: {e}")
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
