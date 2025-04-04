"""
Data loading utilities for the Yelp Reviews Sentiment Analysis project.
"""

import logging
import os
from typing import Dict, Optional, Tuple, Union

import pandas as pd

from ..config import (
    HF_DATASET,
    HF_SPLITS,
    PROCESSED_TEST_PATH,
    PROCESSED_TRAIN_PATH,
    RATING_TO_SENTIMENT,
    RAW_TEST_PATH,
    RAW_TRAIN_PATH,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class YelpDataLoader:
    """Class for loading and handling the Yelp Reviews dataset."""

    def __init__(self):
        """Initialize the data loader."""
        self.train_df = None
        self.test_df = None
        self.train_with_sentiment_df = None
        self.test_with_sentiment_df = None

    def load_raw_data(
        self, force_download: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the raw Yelp dataset.

        Args:
            force_download: If True, force download from Hugging Face even if local files exist.

        Returns:
            Tuple containing the training and test DataFrames.
        """
        if (
            not force_download
            and os.path.exists(RAW_TRAIN_PATH)
            and os.path.exists(RAW_TEST_PATH)
        ):
            logger.info("Loading raw data from local files...")
            self.train_df = pd.read_parquet(RAW_TRAIN_PATH)
            self.test_df = pd.read_parquet(RAW_TEST_PATH)
        else:
            logger.info("Loading data from Hugging Face...")
            try:
                # Load from Hugging Face datasets
                train_path = f"hf://datasets/{HF_DATASET}/{HF_SPLITS['train']}"
                test_path = f"hf://datasets/{HF_DATASET}/{HF_SPLITS['test']}"

                self.train_df = pd.read_parquet(train_path)
                self.test_df = pd.read_parquet(test_path)

                # Save locally for future use
                logger.info("Saving raw data to local files...")
                self.train_df.to_parquet(RAW_TRAIN_PATH)
                self.test_df.to_parquet(RAW_TEST_PATH)
            except Exception as e:
                logger.error(f"Error loading dataset from Hugging Face: {e}")
                logger.info("If you're having issues with HF datasets, try:")
                logger.info("1. Install datasets: pip install datasets")
                logger.info("2. Login to HF: huggingface-cli login")
                raise

        logger.info(f"Loaded raw training data: {self.train_df.shape}")
        logger.info(f"Loaded raw test data: {self.test_df.shape}")

        return self.train_df, self.test_df

    def create_sentiment_labels(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Map original star ratings to sentiment categories (negative, neutral, positive).

        Returns:
            Tuple containing training and test DataFrames with sentiment labels.
        """
        if self.train_df is None or self.test_df is None:
            self.load_raw_data()

        logger.info("Mapping star ratings to sentiment categories...")
        # Create copies to avoid modifying original data
        self.train_with_sentiment_df = self.train_df.copy()
        self.test_with_sentiment_df = self.test_df.copy()

        # Map ratings to sentiment categories
        self.train_with_sentiment_df["sentiment"] = self.train_with_sentiment_df[
            "label"
        ].map(RATING_TO_SENTIMENT)
        self.test_with_sentiment_df["sentiment"] = self.test_with_sentiment_df[
            "label"
        ].map(RATING_TO_SENTIMENT)

        # Save processed data
        logger.info("Saving processed data with sentiment labels...")
        self.train_with_sentiment_df.to_parquet(PROCESSED_TRAIN_PATH)
        self.test_with_sentiment_df.to_parquet(PROCESSED_TEST_PATH)

        logger.info(
            f"Processed training data with sentiment labels: {self.train_with_sentiment_df.shape}"
        )
        logger.info(
            f"Processed test data with sentiment labels: {self.test_with_sentiment_df.shape}"
        )

        return self.train_with_sentiment_df, self.test_with_sentiment_df

    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the processed data with sentiment labels.

        Returns:
            Tuple containing processed training and test DataFrames.
        """
        if os.path.exists(PROCESSED_TRAIN_PATH) and os.path.exists(PROCESSED_TEST_PATH):
            logger.info("Loading processed data from local files...")
            self.train_with_sentiment_df = pd.read_parquet(PROCESSED_TRAIN_PATH)
            self.test_with_sentiment_df = pd.read_parquet(PROCESSED_TEST_PATH)
        else:
            logger.info("Processed data not found. Creating sentiment labels...")
            self.create_sentiment_labels()

        return self.train_with_sentiment_df, self.test_with_sentiment_df

    def get_data_statistics(self) -> Dict:
        """
        Get basic statistics about the dataset.

        Returns:
            Dictionary with dataset statistics.
        """
        if self.train_with_sentiment_df is None:
            self.load_processed_data()

        train_sentiment_counts = (
            self.train_with_sentiment_df["sentiment"].value_counts().to_dict()
        )
        test_sentiment_counts = (
            self.test_with_sentiment_df["sentiment"].value_counts().to_dict()
        )

        # Calculate word counts
        self.train_with_sentiment_df["word_count"] = self.train_with_sentiment_df[
            "text"
        ].apply(lambda x: len(str(x).split()))

        stats = {
            "train_samples": len(self.train_with_sentiment_df),
            "test_samples": len(self.test_with_sentiment_df),
            "train_sentiment_distribution": train_sentiment_counts,
            "test_sentiment_distribution": test_sentiment_counts,
            "avg_words_per_review": self.train_with_sentiment_df["word_count"].mean(),
            "max_words": self.train_with_sentiment_df["word_count"].max(),
            "min_words": self.train_with_sentiment_df["word_count"].min(),
        }

        return stats


# Simple usage example
if __name__ == "__main__":
    data_loader = YelpDataLoader()
    train_df, test_df = data_loader.load_processed_data()
    stats = data_loader.get_data_statistics()
    print(f"Dataset Statistics: {stats}")
