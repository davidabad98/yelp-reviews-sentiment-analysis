"""
Training implementations for different models.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from ..config import RANDOM_SEED
from .metrics import compute_metrics

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """Base trainer class for sentiment analysis models."""

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize the trainer.

        Args:
            model: Model to train
            device: Device to train on
        """
        self.model = model
        self.device = device
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        self.best_val_accuracy = 0
        self.best_model_path = None

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        num_epochs: int = 5,
        output_dir: str = "models",
        model_name: str = "model",
        early_stopping_patience: int = None,
    ) -> Dict:
        """
        Train the model.

        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            num_epochs: Number of epochs to train for
            output_dir: Directory to save model checkpoints
            model_name: Name for saved model files
            early_stopping_patience: Number of epochs to wait before early stopping

        Returns:
            Dict containing training history
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        self.best_model_path = os.path.join(output_dir, f"best_{model_name}.pt")

        # Initialize early stopping counter
        no_improve_epochs = 0

        # Set random seeds for reproducibility
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RANDOM_SEED)

        logger.info(
            f"Starting training with {num_epochs} epochs on device {self.device}"
        )

        # Track total training time
        start_time = time.time()

        for epoch in range(num_epochs):
            # Train for one epoch
            train_metrics = self.train_epoch(
                train_dataloader, criterion, optimizer, scheduler
            )

            # Evaluate on validation set
            val_metrics, _, _ = self.evaluate(val_dataloader, criterion)

            # Update training history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_accuracy"].append(train_metrics["accuracy"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])

            # Log epoch results
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train loss: {train_metrics['loss']:.4f}, "
                f"accuracy: {train_metrics['accuracy']:.4f} | "
                f"Val loss: {val_metrics['loss']:.4f}, "
                f"accuracy: {val_metrics['accuracy']:.4f}"
            )

            # Save best model
            if val_metrics["accuracy"] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics["accuracy"]
                # Save model
                if hasattr(self.model, "save"):
                    self.model.save(self.best_model_path)
                else:
                    torch.save(self.model.state_dict(), self.best_model_path)
                logger.info(
                    f"New best model saved with validation accuracy: {self.best_val_accuracy:.4f}"
                )
                # Reset early stopping counter
                no_improve_epochs = 0
            else:
                # Increment early stopping counter
                no_improve_epochs += 1

            # Check early stopping
            if early_stopping_patience and no_improve_epochs >= early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch+1} epochs "
                    f"with no improvement"
                )
                break

        # Calculate training time
        training_time = time.time() - start_time
        logger.info(
            f"Training completed in {training_time:.2f} seconds "
            f"({training_time/60:.2f} minutes)"
        )

        return self.history

    def train_epoch(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> Dict:
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader for training data
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler

        Returns:
            Dict containing epoch metrics
        """
        self.model.train()
        epoch_loss = 0
        all_preds = []
        all_labels = []

        # Progress bar
        progress_bar = tqdm(dataloader, desc="Training", leave=False)

        for batch in progress_bar:
            # This must be implemented by specific trainers
            loss, batch_preds, batch_labels = self._process_batch(batch, criterion)

            # Update running loss
            epoch_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Update parameters
            optimizer.step()

            # Update learning rate schedule
            if scheduler is not None:
                scheduler.step()

            # Store predictions and labels
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())

        # Calculate metrics
        metrics = compute_metrics(all_preds, all_labels)
        metrics["loss"] = epoch_loss / len(dataloader)

        return metrics

    def evaluate(
        self, dataloader: DataLoader, criterion: nn.Module
    ) -> Tuple[Dict, List, List]:
        """
        Evaluate the model on a dataset.

        Args:
            dataloader: DataLoader for evaluation data
            criterion: Loss function

        Returns:
            Tuple containing metrics, predictions and labels
        """
        self.model.eval()
        epoch_loss = 0
        all_preds = []
        all_labels = []

        # Progress bar
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

        with torch.no_grad():
            for batch in progress_bar:
                # This must be implemented by specific trainers
                loss, batch_preds, batch_labels = self._process_batch(batch, criterion)

                # Update running loss
                epoch_loss += loss.item()

                # Store predictions and labels
                all_preds.extend(batch_preds)
                all_labels.extend(batch_labels)

                # Update progress bar
                progress_bar.set_postfix(loss=loss.item())

        # Calculate metrics
        metrics = compute_metrics(all_preds, all_labels)
        metrics["loss"] = epoch_loss / len(dataloader)

        # Print detailed classification report
        logger.debug(
            f"\nClassification Report:\n {classification_report(all_labels, all_preds, digits=4)}"
        )

        return metrics, all_preds, all_labels

    def _process_batch(
        self, batch: Dict, criterion: nn.Module
    ) -> Tuple[torch.Tensor, List, List]:
        """
        Process a batch of data.

        Args:
            batch: Batch of data
            criterion: Loss function

        Returns:
            Tuple containing loss, predictions and labels

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement _process_batch")


class DistilBERTTrainer(Trainer):
    """Trainer for DistilBERT-based sentiment analysis models."""

    def _process_batch(
        self, batch: Dict, criterion: nn.Module
    ) -> Tuple[torch.Tensor, List, List]:
        """
        Process a batch of DistilBERT data.

        Args:
            batch: Batch of data with input_ids, attention_mask, and labels
            criterion: Loss function

        Returns:
            Tuple containing loss, predictions and labels
        """
        # Move data to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Forward pass
        outputs = self.model(input_ids, attention_mask)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Get predictions
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()

        # Get labels
        labels = labels.detach().cpu().numpy()

        return loss, preds, labels
