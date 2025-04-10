"""
Visualization utilities for the Yelp sentiment analysis project.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_training_history(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training and validation metrics.

    Args:
        history: Dictionary with metrics history
        figsize: Figure size
        save_path: Path to save the plot
    """
    plt.figure(figsize=figsize)

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history["train_accuracy"], label="Training Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Training history plot saved to {save_path}")

    plt.show()


def plot_confusion_matrix(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    classes: List[str],
    normalize: bool = False,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names
        normalize: Whether to normalize values
        figsize: Figure size
        cmap: Colormap
        save_path: Path to save the plot
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        square=True,
        cbar=True,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Confusion matrix plot saved to {save_path}")

    plt.show()


def plot_attention_weights(
    text: str,
    attention_weights: np.ndarray,
    tokenizer,
    layer_idx: int = -1,
    head_idx: int = 0,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot attention weights for a given text.

    Args:
        text: Input text
        attention_weights: Attention weights tensor
        tokenizer: Tokenizer used
        layer_idx: Index of layer to visualize
        head_idx: Index of attention head to visualize
        figsize: Figure size
        save_path: Path to save the plot
    """
    # Tokenize text
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))

    # Get attention weights from specified layer and head
    attn = attention_weights[layer_idx][head_idx].numpy()

    # Create figure
    plt.figure(figsize=figsize)

    # Plot heatmap
    sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens, cmap="YlOrRd", vmin=0.0)
    plt.title(f"Attention Weights (Layer {layer_idx}, Head {head_idx})")
    plt.ylabel("Queries")
    plt.xlabel("Keys")
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Attention weights plot saved to {save_path}")

    plt.show()
