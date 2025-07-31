"""
Data Preprocessing Script for YouTube Comment Analysis (MLOps Pipeline)

This script performs text preprocessing on raw YouTube comment data as part of an MLOps pipeline.
It is designed to be used as a DVC pipeline stage and includes the following steps:

1. Loads raw train and test datasets from the data/raw directory.
2. Applies text normalization and cleaning to the 'clean_comment' column, including:
   - Lowercasing and stripping whitespace
   - Removing unwanted characters (preserving letters, numbers, whitespace, and select punctuation)
   - Removing stopwords (with exceptions for important negations)
   - Lemmatizing words using NLTK
3. Saves the processed datasets to the data/interim directory.
4. Logs all operations and errors for debugging and reproducibility.

Logging:
- DEBUG-level logs for process tracing and successful operations
- ERROR-level logs for exceptions and failures

Typical usage:
    python src/data/data_preprocessing.py

This script is intended to be modular, robust, and reproducible for downstream machine learning workflows.
"""

import logging
import os
import re
import string

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# logging configs
logger = logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("preprocessing_errors.log")
file_handler.setLevel("ERROR")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# download required NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")


# define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        comment = comment.lower()
        comment = comment.strip()
        comment = re.sub(r"\n", " ", comment)
        comment = re.sub(r"[^a-zA-Z0-9\s!?.,]", "", comment)

        # remove stopwords but retain important ones
        stop_words = set(stopwords.words("english")) - {
            "not",
            "but",
            "however",
            "no",
            "yet",
        }
        comment = " ".join([word for word in comment.split() if word not in stop_words])

        # lemmetize the words
        lemmatizer = WordNetLemmatizer()
        comment = " ".join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        logger.error(f"Unexpected error occured while preprocessing the comment: {e}")
        return comment


def normalize_text(df):
    """Apply preprocessing to the text data in the dataframe."""
    try:
        df["clean_comment"] = df["clean_comment"].apply(preprocess_comment)
        logger.debug("Text normalization completed")
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise


def save_data(
    train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str
) -> None:
    """Save the processed TRAIN and TESTt datasets."""
    try:
        interim_data_path = os.path.join(data_path, "interim")
        logger.debug(f"Creating directroy: {interim_data_path}")

        os.makedirs(interim_data_path, exist_ok=True)
        logger.debug(f"Directory {interim_data_path} created or already exists.")

        train_data.to_csv(
            os.path.join(interim_data_path, "train_processed.csv"), index=False
        )
        test_data.to_csv(
            os.path.join(interim_data_path, "test_processed.csv"), index=False
        )

        logger.debug(f"Processed datasets saved to {interim_data_path}")

    except Exception as e:
        logger.error(f"Unexpected error occured while saving the data: {e}")
        raise


def main() -> None:
    try:
        logger.debug("Starting data preprocessing ...")

        # fetch the data from data/raw
        train_data = pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")
        logger.debug("Data fetched successfully.")

        # preprocess the data
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        logger.debug("Data preprocessing completed.")

        # save the data
        save_data(train_processed_data, test_processed_data, data_path="./data")

    except Exception as e:
        logger.error(f"Failed to complete the data preprocessing process: {e}")
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
