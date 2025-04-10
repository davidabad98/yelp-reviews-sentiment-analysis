import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    normalize: bool = False,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues",
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix
        classes: Class names
        normalize: Whether to normalize values
        figsize: Figure size
        cmap: Colormap
        title: Plot title
        save_path: Path to save the plot
    """
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
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Confusion matrix plot saved to {save_path}")

    plt.show()


def plot_prediction_examples(
    examples: List[Dict],
    title: str = "Prediction Examples",
    figsize: Tuple[int, int] = (16, 12),
    max_text_length: int = 200,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot examples of model predictions with their confidence scores.

    Args:
        examples: List of prediction dictionaries containing text, true_label,
                 sentiment_id (predicted), and confidence
        title: Plot title
        figsize: Figure size
        max_text_length: Maximum text length to display
        save_path: Path to save the plot
    """
    n_examples = len(examples)

    # Create figure and axes
    fig, axes = plt.subplots(n_examples, 1, figsize=figsize)
    fig.suptitle(title, fontsize=16)

    # If only one example, wrap axes in a list
    if n_examples == 1:
        axes = [axes]

    # Configure subplot spacing
    plt.subplots_adjust(hspace=0.5)

    for i, example in enumerate(axes):
        if i < len(examples):
            pred = examples[i]

            # Truncate text if too long
            text = pred["text"]
            if len(text) > max_text_length:
                text = text[:max_text_length] + "..."

            # Determine if prediction is correct
            correct = pred["sentiment_id"] == pred["true_label"]

            # Set up colors based on correctness
            color = "green" if correct else "red"

            # Create background color for the text box
            bbox_props = dict(
                boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.5
            )

            # Create title with prediction information
            title_text = f"{'✓' if correct else '✗'} "
            title_text += f"True: {pred.get('true_label_name', pred['true_label'])}, "
            title_text += f"Pred: {pred.get('sentiment_name', pred['sentiment_id'])}, "
            title_text += f"Conf: {pred['confidence']:.2f}"

            # Display title with colored box
            example.text(
                0.5,
                1.02,
                title_text,
                horizontalalignment="center",
                transform=example.transAxes,
                fontsize=12,
                color=color,
                bbox=bbox_props,
            )

            # Display the review text
            example.text(
                0.03, 0.5, text, wrap=True, verticalalignment="center", fontsize=10
            )

            # Remove axis ticks and labels
            example.set_xticks([])
            example.set_yticks([])

            # Set border color based on correctness
            for spine in example.spines.values():
                spine.set_color(color)
                spine.set_linewidth(2)

    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Prediction examples plot saved to {save_path}")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for title
    plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Training History",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training history metrics.

    Args:
        history: Dictionary of training metrics (e.g., loss, accuracy) by epoch
        figsize: Figure size
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=figsize)

    # Plot each metric
    for metric, values in history.items():
        plt.plot(values, label=metric)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Training history plot saved to {save_path}")

    plt.show()


def plot_model_comparison(
    metrics: Dict[str, Dict[str, Union[float, List[float]]]],
    metric_names: List[str] = ["accuracy", "precision", "recall", "f1"],
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot comparison of multiple models across different metrics.

    Args:
        metrics: Dictionary of model metrics {model_name: {metric_name: value}}
        metric_names: List of metrics to include in comparison
        figsize: Figure size
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=figsize)

    # Get available models
    models = list(metrics.keys())
    x = np.arange(len(metric_names))
    width = 0.8 / len(models)  # Width of bars

    # Plot bars for each model
    for i, model in enumerate(models):
        values = [metrics[model].get(metric, 0) for metric in metric_names]
        plt.bar(
            x + i * width - width * (len(models) - 1) / 2, values, width, label=model
        )

    plt.title(title)
    plt.ylabel("Score")
    plt.xticks(x, metric_names)
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.3, axis="y")

    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Model comparison plot saved to {save_path}")

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
