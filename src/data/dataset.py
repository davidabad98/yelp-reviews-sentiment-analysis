"""
PyTorch dataset classes for sentiment analysis.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer

from src.data.preprocessor import DistilBERTPreprocessor, LSTMPreprocessor

from ..config import DISTILBERT_CONFIG, LSTM_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YelpReviewDataset(Dataset):
    """Base Yelp review dataset class."""

    def __init__(self, texts: List[str], labels: List[int], transform=None):
        """
        Initialize the dataset.

        Args:
            texts: List of review texts
            labels: List of sentiment labels
            transform: Optional transform to apply to the data
        """
        assert len(texts) == len(labels), "Reviews and labels must have the same length"
        self.texts = texts
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """Return the dataset size."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing the review and label
        """
        review = self.texts[idx]
        label = self.labels[idx]

        if self.transform:
            review = self.transform(review)

        return {"review": review, "label": label}


class LSTMYelpDataset(YelpReviewDataset):
    """Dataset for LSTM models."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        preprocessor: Optional[LSTMPreprocessor] = None,
        fit_preprocessor: bool = False,
    ):
        """
        Initialize the LSTM dataset.

        Args:
            texts: List of review texts.
            labels: List of sentiment labels.
            preprocessor: LSTM preprocessor instance.
            fit_preprocessor: Whether to fit the preprocessor on this data.
        """
        super().__init__(texts, labels)

        # Initialize preprocessor if not provided
        if preprocessor is None:
            self.preprocessor = LSTMPreprocessor(
                max_vocab_size=LSTM_CONFIG["max_vocab_size"],
                max_sequence_length=LSTM_CONFIG["max_sequence_length"],
            )
        else:
            self.preprocessor = preprocessor

        # Preprocess texts
        logger.info("Preprocessing texts for LSTM...")
        self.processed_texts = self.preprocessor.preprocess_for_lstm(
            texts, fit=fit_preprocessor
        )
        logger.info(f"Processed {len(self.processed_texts)} texts for LSTM")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            Tuple of (preprocessed_text, label).
        """
        text = torch.tensor(self.processed_texts[idx], dtype=torch.long)
        # Calculate length (non-zero tokens, or use actual length if needed)
        length = torch.tensor(
            sum(1 for token in self.processed_texts[idx] if token != 0),
            dtype=torch.long,
        )
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return {"text": text, "lengths": length, "labels": label}


class DistilBERTYelpDataset(YelpReviewDataset):
    """Dataset for BERT-based models."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        preprocessor: Optional[DistilBERTPreprocessor] = None,
    ):
        """
        Initialize the DistilBERT dataset.

        Args:
            texts: List of review texts.
            labels: List of sentiment labels.
            preprocessor: DistilBERT preprocessor instance.
        """
        super().__init__(texts, labels)

        # Initialize preprocessor if not provided
        if preprocessor is None:
            self.preprocessor = DistilBERTPreprocessor(
                pretrained_model_name=DISTILBERT_CONFIG["pretrained_model_name"],
                max_sequence_length=DISTILBERT_CONFIG["max_sequence_length"],
            )
        else:
            self.preprocessor = preprocessor

        # For DistilBERT, we preprocess on the fly to save memory

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            Dictionary with preprocessed text and label.
        """
        text = self.texts[idx]
        encoded = self.preprocessor.preprocess_for_distilbert([text])

        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def create_data_loaders(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_size: float = 0.1,
    model_type: str = "lstm",
    batch_size: Optional[int] = None,
    random_seed: int = 42,
    tokenizer: Optional[DistilBertTokenizer] = None,
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for training, validation, and testing.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        val_size: Validation set size as fraction of training data.
        model_type: Model type ('lstm' or 'distilbert').
        batch_size: Batch size. If None, use model-specific default.
        random_seed: Random seed for reproducibility.

    Returns:
        Dictionary with train, val, and test DataLoaders.
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Determine batch size
    if batch_size is None:
        batch_size = (
            LSTM_CONFIG["batch_size"]
            if model_type == "lstm"
            else DISTILBERT_CONFIG["batch_size"]
        )

    # Extract texts and labels
    train_texts = train_df["text"].tolist()
    train_labels = train_df["sentiment"].tolist()
    test_texts = test_df["text"].tolist()
    test_labels = test_df["sentiment"].tolist()

    # Split training data into train and validation
    indices = np.random.permutation(len(train_texts))
    val_size_abs = int(len(train_texts) * val_size)
    train_indices = indices[val_size_abs:]
    val_indices = indices[:val_size_abs]

    train_texts_split = [train_texts[i] for i in train_indices]
    train_labels_split = [train_labels[i] for i in train_indices]
    val_texts = [train_texts[i] for i in val_indices]
    val_labels = [train_labels[i] for i in val_indices]

    logger.info(
        f"Created data split: {len(train_texts_split)} train, {len(val_texts)} val, {len(test_texts)} test"
    )

    # Create datasets based on model type
    if model_type.lower() == "lstm":
        # Initialize preprocessor for LSTM
        preprocessor = LSTMPreprocessor()

        # Create datasets
        train_dataset = LSTMYelpDataset(
            train_texts_split, train_labels_split, preprocessor, fit_preprocessor=True
        )
        val_dataset = LSTMYelpDataset(val_texts, val_labels, preprocessor)
        test_dataset = LSTMYelpDataset(test_texts, test_labels, preprocessor)

    elif model_type.lower() == "distilbert":
        # Initialize preprocessor for DistilBERT
        preprocessor = DistilBERTPreprocessor(tokenizer)

        # Create datasets
        train_dataset = DistilBERTYelpDataset(
            train_texts_split, train_labels_split, preprocessor
        )
        val_dataset = DistilBERTYelpDataset(val_texts, val_labels, preprocessor)
        test_dataset = DistilBERTYelpDataset(test_texts, test_labels, preprocessor)

    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Expected 'lstm' or 'distilbert'"
        )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    logger.info(f"Created DataLoaders with batch size {batch_size}")

    return {"train": train_loader, "val": val_loader, "test": test_loader}
