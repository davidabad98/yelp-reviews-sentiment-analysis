"""
Evaluation metrics for sentiment analysis models.
"""

import logging
from typing import Dict, List, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_metrics(
    predictions: Union[List[int], np.ndarray],
    labels: Union[List[int], np.ndarray],
    average: str = "weighted",
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        predictions: Predicted classes
        labels: True labels
        average: Averaging method for multi-class metrics

    Returns:
        Dictionary with metrics
    """
    # Convert to numpy arrays if needed
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average=average, zero_division=0)
    recall = recall_score(labels, predictions, average=average, zero_division=0)
    f1 = f1_score(labels, predictions, average=average, zero_division=0)

    # Return as dictionary
    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    return metrics


def compute_confusion_matrix(
    predictions: Union[List[int], np.ndarray],
    labels: Union[List[int], np.ndarray],
    normalize: str = None,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        predictions: Predicted classes
        labels: True labels
        normalize: Normalization method ('true', 'pred', 'all', or None)

    Returns:
        Confusion matrix
    """
    cm = confusion_matrix(labels, predictions)

    if normalize is not None:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    return cm
