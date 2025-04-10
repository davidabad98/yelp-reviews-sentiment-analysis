#!/usr/bin/env python
"""
Script for evaluating and comparing trained sentiment analysis models.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import NUM_CLASSES
from src.data.data_loader import YelpDataLoader
from src.inference.predictor import (
    DistilBERTPredictor,
    LSTMPredictor,
    SentimentPredictor,
)
from src.training.metrics import compute_confusion_matrix, compute_metrics
from src.utils.logger import setup_logger
from src.utils.visualization import (  # plot_prediction_examples,; plot_roc_curves,
    plot_confusion_matrix,
)

# Initialize basic logger at module level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate and compare sentiment analysis models."
    )

    parser.add_argument(
        "--lstm-model-path",
        type=str,
        help="Path to trained LSTM model",
    )

    parser.add_argument(
        "--distilbert-model-path",
        type=str,
        help="Path to trained DistilBERT model",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/test.csv",
        help="Path to test data CSV",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluation",
        help="Directory to save evaluation results",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of samples to evaluate (None for all)",
    )

    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Calculate and print evaluation metrics",
    )

    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate and save evaluation plots",
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=10,
        help="Number of prediction examples to show",
    )

    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save all predictions to CSV",
    )

    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Compare models side by side",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="CUDA device index to use",
    )

    return parser.parse_args()


def load_test_data(data_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load test data from file.

    Args:
        data_path: Path to test data CSV
        sample_size: Number of samples to load (None for all)

    Returns:
        DataFrame with test data
    """
    logger.info(f"Loading test data from {data_path}")

    if os.path.exists(data_path):
        # Load directly from file
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} samples from CSV")
    else:
        # Load using data loader
        logger.info("Data file not found, loading from data loader")
        data_loader = YelpDataLoader()
        _, df = data_loader.load_processed_data()
        logger.info(f"Loaded {len(df)} samples from data loader")

    # Take sample if specified
    if sample_size is not None and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
        logger.info(f"Using random sample of {sample_size} reviews")

    return df


def evaluate_model(
    model_predictor: SentimentPredictor,
    test_df: pd.DataFrame,
    batch_size: int = 32,
) -> Tuple[List[Dict], Dict]:
    """
    Evaluate a model on test data.

    Args:
        model_predictor: Initialized model predictor
        test_df: Test data
        batch_size: Batch size for evaluation

    Returns:
        Tuple of (predictions, metrics)
    """
    # Load model if not already loaded
    if model_predictor.model is None:
        model_predictor.load_model()

    # Get predictions in batches
    texts = test_df["text"].tolist()
    true_labels = test_df["label"].tolist()

    # Get predictions
    logger.info(
        f"Getting predictions for {len(texts)} samples with batch size {batch_size}"
    )
    start_time = time.time()
    all_predictions = []

    # Process in batches for progress tracking
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch_texts = texts[i : i + batch_size]
        batch_predictions = model_predictor.batch_predict(batch_texts)
        all_predictions.extend(batch_predictions)

    elapsed = time.time() - start_time

    # Add true labels to predictions
    for i, pred in enumerate(all_predictions):
        pred["true_label"] = int(true_labels[i])
        pred["true_sentiment"] = model_predictor.sentiment_map[true_labels[i]]

    # Calculate metrics using our existing metrics module
    predicted_labels = [pred["sentiment_id"] for pred in all_predictions]

    # Use the existing compute_metrics function
    basic_metrics = compute_metrics(predicted_labels, true_labels)

    # Create confusion matrix using existing function
    cm = compute_confusion_matrix(predicted_labels, true_labels)

    # Generate classification report
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=list(model_predictor.sentiment_map.values()),
        output_dict=True,
    )

    # Calculate timing stats
    samples_per_second = len(texts) / elapsed

    # Combine all metrics
    metrics = {
        **basic_metrics,  # Include accuracy, precision, recall, f1
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "timing": {
            "total_seconds": elapsed,
            "samples_per_second": samples_per_second,
            "ms_per_sample": (elapsed * 1000) / len(texts),
        },
    }

    return all_predictions, metrics


def print_metrics(metrics: Dict, model_name: str) -> None:
    """
    Print evaluation metrics.

    Args:
        metrics: Metrics dictionary
        model_name: Name of the model
    """
    print(f"\n=== {model_name} Evaluation Metrics ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

    # Print timing info
    timing = metrics["timing"]
    print(f"\nTiming Information:")
    print(f"  Total time: {timing['total_seconds']:.2f} seconds")
    print(f"  Samples per second: {timing['samples_per_second']:.2f}")
    print(f"  Average time per sample: {timing['ms_per_sample']:.2f} ms")

    # Print classification report
    print("\nClassification Report:")
    report = metrics["classification_report"]

    # Format report
    headers = ["", "precision", "recall", "f1-score", "support"]
    print(
        f"{headers[0]:<10} {headers[1]:<10} {headers[2]:<10} {headers[3]:<10} {headers[4]:<10}"
    )
    print("-" * 50)

    for label, metrics in report.items():
        if label in ["accuracy", "macro avg", "weighted avg", "samples avg"]:
            continue
        print(
            f"{label:<10} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1-score']:<10.4f} {metrics['support']:<10}"
        )

    print("-" * 50)
    print(
        f"{'accuracy':<10} {'':<10} {'':<10} {report['accuracy']:<10.4f} {sum(metrics['support'] for _, metrics in report.items() if isinstance(metrics, dict) and 'support' in metrics):<10}"
    )
    print(
        f"{'macro avg':<10} {report['macro avg']['precision']:<10.4f} {report['macro avg']['recall']:<10.4f} {report['macro avg']['f1-score']:<10.4f} {report['macro avg']['support']:<10}"
    )
    print(
        f"{'weighted avg':<10} {report['weighted avg']['precision']:<10.4f} {report['weighted avg']['recall']:<10.4f} {report['weighted avg']['f1-score']:<10.4f} {report['weighted avg']['support']:<10}"
    )


def save_predictions(predictions: List[Dict], output_path: str) -> None:
    """
    Save predictions to CSV.

    Args:
        predictions: List of prediction dictionaries
        output_path: Path to save CSV
    """
    # Convert to DataFrame
    df = pd.DataFrame(predictions)

    # Flatten probability dictionary
    prob_columns = predictions[0]["probabilities"].keys()
    for col in prob_columns:
        df[f"prob_{col}"] = df["probabilities"].apply(lambda x: x[col])

    # Drop probabilities column (now flattened)
    df = df.drop(columns=["probabilities"])

    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} predictions to {output_path}")


def find_prediction_examples(
    predictions: List[Dict],
    num_examples: int = 5,
    include_correct: bool = True,
    include_incorrect: bool = True,
) -> List[Dict]:
    """
    Find interesting prediction examples.

    Args:
        predictions: List of prediction dictionaries
        num_examples: Number of examples to find
        include_correct: Include correctly classified examples
        include_incorrect: Include incorrectly classified examples

    Returns:
        List of example predictions
    """
    # Split into correct and incorrect predictions
    correct = [p for p in predictions if p["sentiment_id"] == p["true_label"]]
    incorrect = [p for p in predictions if p["sentiment_id"] != p["true_label"]]

    # Set how many of each type to include
    total_examples = num_examples

    # If both types requested, split examples between them
    if include_correct and include_incorrect:
        # Try to balance, but adjust based on availability
        correct_count = min(total_examples // 2, len(correct))
        incorrect_count = min(total_examples - correct_count, len(incorrect))
        # If not enough incorrect examples, add more correct ones
        if incorrect_count < total_examples - correct_count:
            correct_count = min(total_examples - incorrect_count, len(correct))
    elif include_correct:
        correct_count = min(total_examples, len(correct))
        incorrect_count = 0
    elif include_incorrect:
        correct_count = 0
        incorrect_count = min(total_examples, len(incorrect))
    else:
        # No examples requested
        return []

    # Sort by confidence
    sorted_correct = sorted(correct, key=lambda x: x["confidence"], reverse=True)
    sorted_incorrect = sorted(incorrect, key=lambda x: x["confidence"], reverse=True)

    # Get high confidence correct examples
    high_conf_correct = sorted_correct[: correct_count // 2]
    # Get low confidence correct examples
    low_conf_correct = sorted(sorted_correct, key=lambda x: x["confidence"])[
        : correct_count - len(high_conf_correct)
    ]

    # Get high confidence incorrect examples (most interesting mistakes)
    high_conf_incorrect = sorted_incorrect[:incorrect_count]

    # Combine examples
    examples = high_conf_correct + low_conf_correct + high_conf_incorrect

    # Shuffle to mix correct and incorrect
    np.random.shuffle(examples)

    return examples[:num_examples]


def compare_models(
    lstm_preds: List[Dict], distilbert_preds: List[Dict], num_examples: int = 5
) -> List[Dict]:
    """
    Find examples where models disagree.

    Args:
        lstm_preds: LSTM model predictions
        distilbert_preds: DistilBERT model predictions
        num_examples: Number of examples to find

    Returns:
        List of comparison examples
    """
    # Create dictionaries for easier lookup
    lstm_dict = {p["text"]: p for p in lstm_preds}
    distilbert_dict = {p["text"]: p for p in distilbert_preds}

    # Find texts where predictions disagree
    common_texts = set(lstm_dict.keys()) & set(distilbert_dict.keys())
    disagreements = []

    for text in common_texts:
        lstm_pred = lstm_dict[text]
        distilbert_pred = distilbert_dict[text]

        if lstm_pred["sentiment_id"] != distilbert_pred["sentiment_id"]:
            # Keep track of which model was correct
            true_label = lstm_pred["true_label"]
            lstm_correct = lstm_pred["sentiment_id"] == true_label
            distilbert_correct = distilbert_pred["sentiment_id"] == true_label

            disagreements.append(
                {
                    "text": text,
                    "true_label": true_label,
                    "true_sentiment": lstm_pred["true_sentiment"],
                    "lstm_prediction": lstm_pred["sentiment"],
                    "lstm_confidence": lstm_pred["confidence"],
                    "lstm_correct": lstm_correct,
                    "distilbert_prediction": distilbert_pred["sentiment"],
                    "distilbert_confidence": distilbert_pred["confidence"],
                    "distilbert_correct": distilbert_correct,
                }
            )

    # Sort by combined confidence (interesting cases are high confidence disagreements)
    disagreements.sort(
        key=lambda x: x["lstm_confidence"] + x["distilbert_confidence"], reverse=True
    )

    return disagreements[:num_examples]


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Set device
    device = torch.device(
        f"cuda:{args.cuda_device}"
        if torch.cuda.is_available() and args.cuda_device >= 0
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Set up file logger - enhance the existing logger with file logging
    log_file = os.path.join(output_dir, "evaluation.log")
    file_logger = setup_logger("model_evaluation", log_file)

    # Log both to console and file
    logger.info(f"Starting model evaluation with arguments: {args}")
    file_logger.info(f"Starting model evaluation with arguments: {args}")

    # Load test data
    test_df = load_test_data(args.data_path, args.sample_size)
    logger.info(f"Test data shape: {test_df.shape}")

    results = {}

    # Evaluate DistilBERT if path provided
    if args.distilbert_model_path:
        logger.info(f"Evaluating DistilBERT model from {args.distilbert_model_path}")
        distilbert_predictor = DistilBERTPredictor(
            args.distilbert_model_path, device=device
        )

        # Get predictions and metrics
        distilbert_preds, distilbert_metrics = evaluate_model(
            distilbert_predictor, test_df, args.batch_size
        )

        # Print metrics if requested
        if args.metrics:
            print_metrics(distilbert_metrics, "DistilBERT")

        # Save predictions if requested
        if args.save_predictions:
            save_predictions(
                distilbert_preds, os.path.join(output_dir, "distilbert_predictions.csv")
            )

        # Save metrics
        with open(os.path.join(output_dir, "distilbert_metrics.json"), "w") as f:
            json.dump(distilbert_metrics, f, indent=4)

        # Generate confusion matrix plot
        if args.plots:
            cm = np.array(distilbert_metrics["confusion_matrix"])
            cm_path = os.path.join(output_dir, "distilbert_confusion_matrix.png")
            plot_confusion_matrix(
                cm,
                classes=list(distilbert_predictor.sentiment_map.values()),
                normalize=True,
                title="DistilBERT Confusion Matrix",
                save_path=cm_path,
            )

            # Generate example plots
            examples = find_prediction_examples(distilbert_preds, args.examples)
            examples_path = os.path.join(output_dir, "distilbert_examples.png")
            plot_prediction_examples(
                examples,
                title="DistilBERT Prediction Examples",
                save_path=examples_path,
            )

        results["distilbert"] = {
            "predictions": distilbert_preds,
            "metrics": distilbert_metrics,
        }

    # Evaluate LSTM if path provided
    if args.lstm_model_path:
        logger.info(f"Evaluating LSTM model from {args.lstm_model_path}")
        lstm_predictor = LSTMPredictor(args.lstm_model_path, device=device)

        try:
            # Get predictions and metrics
            lstm_preds, lstm_metrics = evaluate_model(
                lstm_predictor, test_df, args.batch_size
            )

            # Print metrics if requested
            if args.metrics:
                print_metrics(lstm_metrics, "LSTM")

            # Save predictions if requested
            if args.save_predictions:
                save_predictions(
                    lstm_preds, os.path.join(output_dir, "lstm_predictions.csv")
                )

            # Save metrics
            with open(os.path.join(output_dir, "lstm_metrics.json"), "w") as f:
                json.dump(lstm_metrics, f, indent=4)

            # Generate confusion matrix plot
            if args.plots:
                cm = np.array(lstm_metrics["confusion_matrix"])
                cm_path = os.path.join(output_dir, "lstm_confusion_matrix.png")
                plot_confusion_matrix(
                    cm,
                    classes=list(lstm_predictor.sentiment_map.values()),
                    normalize=True,
                    title="LSTM Confusion Matrix",
                    save_path=cm_path,
                )

                # Generate example plots
                examples = find_prediction_examples(lstm_preds, args.examples)
                examples_path = os.path.join(output_dir, "lstm_examples.png")
                plot_prediction_examples(
                    examples, title="LSTM Prediction Examples", save_path=examples_path
                )

            results["lstm"] = {"predictions": lstm_preds, "metrics": lstm_metrics}
        except NotImplementedError:
            logger.warning("LSTM prediction not implemented yet, skipping evaluation")

    # Compare models if requested and both models were evaluated
    if args.comparison and "lstm" in results and "distilbert" in results:
        logger.info("Comparing LSTM and DistilBERT models")

        # Find examples where models disagree
        disagreements = compare_models(
            results["lstm"]["predictions"],
            results["distilbert"]["predictions"],
            args.examples,
        )

        # Save disagreements to file
        if disagreements:
            with open(os.path.join(output_dir, "model_disagreements.json"), "w") as f:
                json.dump(disagreements, f, indent=4)

            # Print some disagreement examples
            print("\n=== Model Disagreement Examples ===")
            for i, example in enumerate(disagreements[: min(5, len(disagreements))]):
                print(f"\nExample {i+1}:")
                print(
                    f"Text: {example['text'][:100]}..."
                    if len(example["text"]) > 100
                    else f"Text: {example['text']}"
                )
                print(f"True sentiment: {example['true_sentiment']}")
                print(
                    f"LSTM: {example['lstm_prediction']} (confidence: {example['lstm_confidence']:.4f}) - {'✓' if example['lstm_correct'] else '✗'}"
                )
                print(
                    f"DistilBERT: {example['distilbert_prediction']} (confidence: {example['distilbert_confidence']:.4f}) - {'✓' if example['distilbert_correct'] else '✗'}"
                )
        else:
            logger.info("No disagreements found between models")

        # Create comparative bar chart of metrics
        if args.plots:
            # Extract metrics for comparison
            metrics_to_compare = ["accuracy", "precision", "recall", "f1"]
            lstm_values = [results["lstm"]["metrics"][m] for m in metrics_to_compare]
            distilbert_values = [
                results["distilbert"]["metrics"][m] for m in metrics_to_compare
            ]

            # Create plot
            plt.figure(figsize=(10, 6))
            x = np.arange(len(metrics_to_compare))
            width = 0.35

            plt.bar(x - width / 2, lstm_values, width, label="LSTM")
            plt.bar(x + width / 2, distilbert_values, width, label="DistilBERT")

            plt.xlabel("Metrics")
            plt.ylabel("Scores")
            plt.title("Model Performance Comparison")
            plt.xticks(x, metrics_to_compare)
            plt.ylim(0, 1.0)
            plt.legend()
            plt.grid(axis="y", linestyle="--", alpha=0.7)

            # Add value labels
            for i, v in enumerate(lstm_values):
                plt.text(i - width / 2, v + 0.01, f"{v:.3f}", ha="center")
            for i, v in enumerate(distilbert_values):
                plt.text(i + width / 2, v + 0.01, f"{v:.3f}", ha="center")

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "model_comparison.png"))
            plt.close()

            # Create timing comparison
            lstm_timing = results["lstm"]["metrics"]["timing"]
            distilbert_timing = results["distilbert"]["metrics"]["timing"]

            timing_metrics = ["samples_per_second", "ms_per_sample"]
            lstm_timing_values = [lstm_timing[m] for m in timing_metrics]
            distilbert_timing_values = [distilbert_timing[m] for m in timing_metrics]

            # Create plot (note: log scale may be appropriate for speed differences)
            plt.figure(figsize=(10, 6))
            x = np.arange(len(timing_metrics))

            plt.bar(x - width / 2, lstm_timing_values, width, label="LSTM")
            plt.bar(x + width / 2, distilbert_timing_values, width, label="DistilBERT")

            plt.xlabel("Metrics")
            plt.ylabel("Values")
            plt.title("Model Timing Comparison")
            plt.xticks(x, ["Samples/second", "Ms/sample"])
            plt.legend()
            plt.grid(axis="y", linestyle="--", alpha=0.7)

            # Add value labels
            for i, v in enumerate(lstm_timing_values):
                plt.text(i - width / 2, v + (v * 0.02), f"{v:.2f}", ha="center")
            for i, v in enumerate(distilbert_timing_values):
                plt.text(i + width / 2, v + (v * 0.02), f"{v:.2f}", ha="center")

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "timing_comparison.png"))
            plt.close()

    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Evaluation results saved to: {output_dir}")

    if "distilbert" in results:
        print(
            f"DistilBERT Accuracy: {results['distilbert']['metrics']['accuracy']:.4f}"
        )
        print(f"DistilBERT F1 Score: {results['distilbert']['metrics']['f1']:.4f}")

    if "lstm" in results:
        print(f"LSTM Accuracy: {results['lstm']['metrics']['accuracy']:.4f}")
        print(f"LSTM F1 Score: {results['lstm']['metrics']['f1']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
