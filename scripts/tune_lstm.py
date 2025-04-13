#!/usr/bin/env python
"""
Hyperparameter tuning script for LSTM sentiment analysis model.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import LSTM_CONFIG, NUM_CLASSES, RANDOM_SEED
from src.data.data_loader import YelpDataLoader
from src.data.dataset import create_data_loaders
from src.models.lstm_model import LSTMSentimentModel
from src.training.hyperparameter_tuner import HyperparameterTuner
from src.utils.logger import setup_logger

# Set up argument parser
parser = argparse.ArgumentParser(
    description="Tune hyperparameters for LSTM sentiment analysis"
)
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--epochs", type=int, default=5, help="Number of epochs per fold")
parser.add_argument(
    "--k_folds", type=int, default=3, help="Number of cross-validation folds"
)
parser.add_argument(
    "--num_workers", type=int, default=4, help="Number of workers for data loading"
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="models/lstm_tuning",
    help="Directory to save results",
)
parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
parser.add_argument(
    "--early_stopping", type=int, default=2, help="Early stopping patience"
)
parser.add_argument(
    "--max_combinations",
    type=int,
    default=None,
    help="Max hyperparameter combinations to try",
)


def main():
    """Main function for hyperparameter tuning."""
    # Parse arguments
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("logs", f"hp_tuning_{timestamp}.log")
    logger = setup_logger("hp_tuning", log_file)

    logger.info(f"Starting LSTM hyperparameter tuning with arguments: {args}")

    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading and processing data...")
    data_loader = YelpDataLoader()
    train_df, _ = data_loader.load_processed_data()

    logger.info(f"Train set shape: {train_df.shape}")

    # Create a dataset for hyperparameter tuning
    # We'll use just the training data since we'll do k-fold cross-validation
    loaders = create_data_loaders(
        train_df,
        None,  # No separate test data needed
        0.0,  # No validation split needed, k-fold will handle this
        "lstm",
        batch_size=args.batch_size,
    )

    # Get the training dataset
    train_dataset = loaders["dataset"]
    vocab = loaders.get("vocab")
    vocab_size = LSTM_CONFIG["max_vocab_size"]

    # Define the model factory function
    def model_factory(**kwargs):
        return LSTMSentimentModel(
            vocab_size=vocab_size,
            embedding_dim=kwargs.get(
                "embedding_dim", LSTM_CONFIG.get("embedding_dim", 300)
            ),
            hidden_dim=kwargs.get("hidden_dim", LSTM_CONFIG.get("hidden_dim", 256)),
            num_layers=kwargs.get("num_layers", LSTM_CONFIG.get("num_layers", 2)),
            bidirectional=kwargs.get(
                "bidirectional", LSTM_CONFIG.get("bidirectional", False)
            ),
            num_classes=NUM_CLASSES,
            dropout=kwargs.get("dropout", 0.3),
            padding_idx=0,
            pretrained_embeddings=None,
        )

    # Define optimizer factory function
    def optimizer_factory(params, **kwargs):
        optimizer_type = kwargs.pop("optimizer_type", "adam")

        if optimizer_type == "adam":
            return torch.optim.Adam(params, **kwargs)
        elif optimizer_type == "adamw":
            return torch.optim.AdamW(params, **kwargs)
        elif optimizer_type == "sgd":
            return torch.optim.SGD(params, **kwargs)
        elif optimizer_type == "rmsprop":
            return torch.optim.RMSprop(params, **kwargs)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    # Define hyperparameter grid
    hyperparam_grid = {
        "optimizer_type": ["adam", "adamw"],
        "lr": [0.0001, 0.0005, 0.001],
        "weight_decay": [1e-5, 1e-4, 0.0],
    }

    # Model hyperparameter grid (these will be passed to model_factory)
    model_params = {
        "embedding_dim": [100, 200, 300],
        "hidden_dim": [128, 256],
        "num_layers": [1, 2],
        "bidirectional": [False, True],
        "dropout": [0.2, 0.3, 0.5],
    }

    # If max_combinations is set, sample a subset of model parameter combinations
    if args.max_combinations:
        # We'll just limit the number of combinations
        hyperparam_grid["lr"] = hyperparam_grid["lr"][:2]
        hyperparam_grid["weight_decay"] = hyperparam_grid["weight_decay"][:2]
        model_params["embedding_dim"] = model_params["embedding_dim"][:2]
        model_params["hidden_dim"] = model_params["hidden_dim"][:1]
        model_params["num_layers"] = model_params["num_layers"][:1]
        model_params["bidirectional"] = model_params["bidirectional"][:1]
        model_params["dropout"] = model_params["dropout"][:2]

    # Create model parameter combinations
    model_param_keys, model_param_values = zip(*model_params.items())
    model_param_combinations = [
        dict(zip(model_param_keys, v)) for v in itertools.product(*model_param_values)
    ]

    if args.max_combinations and len(model_param_combinations) > args.max_combinations:
        import random

        random.seed(args.seed)
        model_param_combinations = random.sample(
            model_param_combinations, args.max_combinations
        )

    # Initialize loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize hyperparameter tuner
    tuner = HyperparameterTuner(
        model_factory=model_factory,
        optimizer_factory=optimizer_factory,
        criterion=criterion,
        device=device,
        logger=logger,
    )

    # Keep track of best configurations
    all_results = []

    # Tune for each model configuration
    for i, model_config in enumerate(model_param_combinations):
        logger.info(
            f"Model configuration {i+1}/{len(model_param_combinations)}: {model_config}"
        )

        start_time = time.time()
        best_params, best_score = tuner.tune_hyperparameters(
            dataset=train_dataset,
            param_grid=hyperparam_grid,
            k_folds=args.k_folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            early_stopping_patience=args.early_stopping,
            model_params=model_config,
        )
        elapsed_time = time.time() - start_time

        # Record results
        result = {
            "model_config": model_config,
            "best_optimizer_params": best_params,
            "score": best_score,
            "time": elapsed_time,
        }
        all_results.append(result)

        # Save intermediate results
        with open(
            os.path.join(args.output_dir, f"tuning_results_{timestamp}.json"), "w"
        ) as f:
            json.dump(all_results, f, indent=2)

    # Find overall best configuration
    best_result = max(all_results, key=lambda x: x["score"])

    logger.info(f"Overall best configuration:")
    logger.info(f"Model params: {best_result['model_config']}")
    logger.info(f"Optimizer params: {best_result['best_optimizer_params']}")
    logger.info(f"Score: {best_result['score']:.4f}")

    # Save final results
    with open(os.path.join(args.output_dir, f"best_config_{timestamp}.json"), "w") as f:
        json.dump(best_result, f, indent=2)

    logger.info("Hyperparameter tuning completed successfully!")
    return 0


if __name__ == "__main__":
    import itertools  # Required for the model parameter combinations

    sys.exit(main())
