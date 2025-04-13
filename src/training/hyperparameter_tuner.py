import itertools
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.utils.cross_validation import create_fold_datasets, get_k_folds


class HyperparameterTuner:
    """
    A general hyperparameter tuning class that uses k-fold cross-validation
    to find optimal hyperparameters for a given model.
    """

    def __init__(
        self,
        model_factory: Callable,
        optimizer_factory: Callable,
        criterion: nn.Module,
        device: torch.device,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the hyperparameter tuner.

        Args:
            model_factory: Function that creates a model instance with given parameters
            optimizer_factory: Function that creates an optimizer with given parameters
            criterion: Loss function
            device: Device to run the model on
            logger: Logger instance
        """
        self.model_factory = model_factory
        self.optimizer_factory = optimizer_factory
        self.criterion = criterion
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

    def log(self, message: str) -> None:
        """Log a message using the configured logger."""
        self.logger.info(message)

    def tune_hyperparameters(
        self,
        dataset: Dataset,
        param_grid: Dict[str, List[Any]],
        k_folds: int = 5,
        epochs: int = 3,
        batch_size: int = 64,
        num_workers: int = 4,
        early_stopping_patience: int = 2,
        model_params: Dict[str, Any] = None,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Tune hyperparameters using k-fold cross-validation.

        Args:
            dataset: Dataset to use
            param_grid: Dictionary mapping parameter names to possible values
            k_folds: Number of folds for cross-validation
            epochs: Number of epochs to train for each fold
            batch_size: Batch size
            num_workers: Number of workers for data loading
            early_stopping_patience: Number of epochs to wait before early stopping
            model_params: Additional parameters for model initialization

        Returns:
            Tuple: (best_params, best_score)
        """
        model_params = model_params or {}
        best_config, best_score = None, 0

        # Generate all combinations of hyperparameters
        param_keys, param_values = zip(*param_grid.items())
        param_combinations = [
            dict(zip(param_keys, v)) for v in itertools.product(*param_values)
        ]

        self.log(
            f"Starting hyperparameter tuning with {len(param_combinations)} combinations"
        )

        for i, params in enumerate(param_combinations):
            self.log(f"Combination {i+1}/{len(param_combinations)}: {params}")
            total_val_metrics = {"accuracy": 0, "f1": 0, "loss": 0}
            start_time = time.time()

            # Perform k-fold cross-validation
            for fold, (train_idx, val_idx) in enumerate(
                get_k_folds(dataset, k=k_folds)
            ):
                self.log(f"  Fold {fold+1}/{k_folds}")

                # Create data loaders for this fold
                train_loader, val_loader = create_fold_datasets(
                    dataset, train_idx, val_idx, batch_size, num_workers
                )

                # Create model and optimizer
                model = self.model_factory(**model_params).to(self.device)
                optimizer = self.optimizer_factory(model.parameters(), **params)
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=2, gamma=0.5
                )

                # Train model
                best_val_metrics = self._train_fold(
                    model,
                    train_loader,
                    val_loader,
                    optimizer,
                    scheduler,
                    epochs,
                    early_stopping_patience,
                )

                # Accumulate metrics
                for key, value in best_val_metrics.items():
                    total_val_metrics[key] += value

            # Calculate average metrics across folds
            avg_val_metrics = {k: v / k_folds for k, v in total_val_metrics.items()}
            elapsed_time = time.time() - start_time

            # Use a combined metric (you can adjust this based on importance)
            combined_score = (
                avg_val_metrics["accuracy"] * 0.5 + avg_val_metrics["f1"] * 0.5
            )

            self.log(f"  Average metrics: {avg_val_metrics}")
            self.log(f"  Combined score: {combined_score:.4f}")
            self.log(f"  Time taken: {elapsed_time:.2f}s")

            if combined_score > best_score:
                best_score = combined_score
                best_config = params
                self.log(f"  New best configuration found!")

        self.log(f"Best configuration: {best_config} with score: {best_score:.4f}")
        return best_config, best_score

    def _train_fold(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        num_epochs: int,
        early_stopping_patience: int,
    ) -> Dict[str, float]:
        """
        Train and evaluate model for one fold.

        Args:
            model: Model to train
            train_loader: DataLoader with training data
            val_loader: DataLoader with validation data
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            num_epochs: Number of epochs to train for
            early_stopping_patience: Number of epochs to wait before early stopping

        Returns:
            Dict: Best validation metrics
        """
        from src.training.trainer import LSTMTrainer

        trainer = LSTMTrainer(model, self.device)

        best_val_metrics = None
        no_improve_count = 0
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            # Train
            train_metrics = trainer.train_epoch(
                train_loader,
                self.criterion,
                optimizer,
                current_epoch=epoch,
                accumulation_steps=1,  # Simplified for hyperparameter tuning
            )

            # Validate
            val_metrics, _, _ = trainer.evaluate(val_loader, self.criterion)

            # Update scheduler
            scheduler.step()

            # Check for improvement
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_val_metrics = val_metrics
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Early stopping
            if no_improve_count >= early_stopping_patience:
                self.log(f"    Early stopping at epoch {epoch+1}")
                break

        return best_val_metrics
