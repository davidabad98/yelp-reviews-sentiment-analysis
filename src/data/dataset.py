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


class LazyLSTMYelpDataset(YelpReviewDataset):
    """Dataset for LSTM models with lazy loading."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        preprocessor: LSTMPreprocessor,
    ):
        """
        Initialize the lazy-loading LSTM dataset.

        Args:
            texts: List of raw review texts
            labels: List of sentiment labels
            preprocessor: Pre-fitted LSTM preprocessor
        """
        super().__init__(texts, labels)
        self.preprocessor = preprocessor
        self.raw_texts = texts

        # Validate that preprocessor is already fitted
        if self.preprocessor.word_to_idx is None:
            raise ValueError(
                "Preprocessor must be fitted before using with LazyLSTMYelpDataset"
            )

        logger.info(f"Created lazy-loading dataset with {len(texts)} samples")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset, preprocessing on-demand.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary with text tensor, length, and label
        """
        # Get raw text and preprocess just this single example
        raw_text = self.raw_texts[idx]

        # Tokenize the text (using the existing preprocessor)
        tokenized_text = self.preprocessor.preprocess([raw_text])[0]

        # Convert to sequence
        sequence = self.preprocessor.texts_to_sequences([tokenized_text])[0]

        # Pad sequence
        padded_sequence = self.preprocessor.pad_sequences([sequence])[0]

        # Convert to tensor
        text = torch.tensor(padded_sequence, dtype=torch.long)

        # Calculate length (non-zero tokens)
        length = torch.tensor(
            max(1, sum(1 for token in padded_sequence if token != 0)),
            dtype=torch.long,
        )

        # Get label
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
    num_workers: int = 4,  # Add this parameter for parallel loading
    pin_memory: bool = True,  # Add this parameter for faster data transfer to GPU
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for training, validation, and testing with lazy loading.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        val_size: Validation set size as fraction of training data
        model_type: Model type ('lstm' or 'distilbert')
        batch_size: Batch size. If None, use model-specific default
        random_seed: Random seed for reproducibility
        tokenizer: Tokenizer for DistilBERT
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to use pin_memory for faster data transfer to GPU

    Returns:
        Dictionary with train, val, and test DataLoaders
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
        # Initialize and fit preprocessor on a small subset for efficiency
        preprocessor = LSTMPreprocessor()
        # Fit on a subset of training data (this is more efficient)
        sample_size = min(
            100000, len(train_texts_split)
        )  # Use at most 100K samples for vocab
        sample_indices = np.random.choice(
            len(train_texts_split), sample_size, replace=False
        )
        sample_texts = [train_texts_split[i] for i in sample_indices]

        # Preprocess and fit on the sample
        tokenized_sample = preprocessor.preprocess(sample_texts)
        preprocessor.fit(tokenized_sample)

        logger.info(
            f"Fitted preprocessor on {sample_size} samples with vocab size {preprocessor.vocab_size}"
        )

        # Create lazy-loading datasets
        train_dataset = LazyLSTMYelpDataset(
            train_texts_split, train_labels_split, preprocessor
        )
        val_dataset = LazyLSTMYelpDataset(val_texts, val_labels, preprocessor)
        test_dataset = LazyLSTMYelpDataset(test_texts, test_labels, preprocessor)

    elif model_type.lower() == "distilbert":
        # For DistilBERT, keep the existing implementation
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

    # Create DataLoaders with parallel loading
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # Parallel loading
        pin_memory=pin_memory,  # Faster data transfer to GPU
        persistent_workers=(
            True if num_workers > 0 else False
        ),  # Keep workers alive between epochs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )

    logger.info(
        f"Created DataLoaders with batch size {batch_size} and {num_workers} workers"
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
