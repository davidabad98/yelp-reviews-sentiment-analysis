"""
Training implementations for different models.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import RANDOM_SEED
from .metrics import compute_metrics

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """Base trainer class for sentiment analysis models."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        accumulation_steps: int = 1,  # parameter with default=1 (no accumulation)
    ):
        """
        Initialize the trainer.

        Args:
            model: Model to train
            device: Device to train on
        Attributes:
            accumulation_steps: Number of batches to accumulate gradients before
                performing a parameter update. This enables effective batch size
                enlargement while keeping memory usage manageable.
                - Value = 1: Normal training (immediate update after each batch)
                - Value > 1: Accumulates gradients over N batches before updating.
                Typical usage: Set to 4-8 when facing GPU memory limitations with
                large batches. Requires corresponding logic in the training loop
                to accumulate gradients and only step optimizer every N batches.
                For LSTM: 2-4 accumulation steps with batch size ~64-128
                For DistilBERT: 4-8 accumulation steps with batch size ~16-32
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
        self.best_model_path = (None,)
        self.accumulation_steps = accumulation_steps
        logger.info(
            f"Trainer initialized with gradient accumulation over {accumulation_steps} steps"
        )

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
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
            logger.debug(f"Starting epoch {epoch+1}")
            # Train for one epoch
            train_metrics = self.train_epoch(
                train_dataloader, criterion, optimizer, scheduler
            )

            # Update learning rate schedule if it's an epoch-based scheduler
            if scheduler is not None and not (
                hasattr(scheduler, "is_batch_level_scheduler")
                and scheduler.is_batch_level_scheduler
            ):
                scheduler.step()

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
                self.save_model(self.best_model_path)
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
        scheduler=None,
    ) -> Dict:
        """
        Train for one epoch with gradient accumulation.

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

        # Clear gradients at the beginning of epoch
        optimizer.zero_grad()

        # Progress bar
        progress_bar = tqdm(dataloader, desc="Training", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            # This must be implemented by specific trainers
            loss, batch_preds, batch_labels = self._process_batch(batch, criterion)

            # Scale loss for accumulation
            loss = loss / self.accumulation_steps

            # Backward pass
            loss.backward()

            # Update running metrics
            epoch_loss += (
                loss.item() * self.accumulation_steps
            )  # Rescale to get actual loss
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)

            # Only update parameters after accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0 or (
                batch_idx + 1 == len(dataloader)
            ):
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters
                optimizer.step()

                # Zero gradients
                optimizer.zero_grad()

                # Update learning rate schedule if it's a step-based scheduler
                if (
                    scheduler is not None
                    and hasattr(scheduler, "is_batch_level_scheduler")
                    and scheduler.is_batch_level_scheduler
                ):
                    scheduler.step()

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item() * self.accumulation_steps)

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

    def save_model(self, path):
        """Save model with explicit verification."""
        if hasattr(self.model, "save"):
            self.model.save(path)
        else:
            torch.save(self.model.state_dict(), path)

        # Verify file was created
        if os.path.exists(path):
            file_size_mb = os.path.getsize(path) / (1024 * 1024)
            logger.info(f"✓ Model saved successfully to {path} ({file_size_mb:.2f} MB)")
        else:
            logger.error(f"✗ Failed to save model to {path}")

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

    def monitor_system_resources(self, device):
        if torch.cuda.is_available():
            # GPU monitoring
            current_mem = torch.cuda.memory_allocated(device) / (1024**3)  # GB
            max_mem = torch.cuda.get_device_properties(device).total_memory / (
                1024**3
            )  # GB
            percent_used = (current_mem / max_mem) * 100
            logger.debug(
                f"GPU Memory: {current_mem:.2f}GB / {max_mem:.2f}GB ({percent_used:.1f}%)"
            )
            if percent_used > 90:
                logger.warning("⚠️ APPROACHING GPU MEMORY LIMIT!")

            # CPU/RAM monitoring
            cpu_percent = psutil.cpu_percent()
            ram_percent = psutil.virtual_memory().percent
            logger.debug(f"CPU: {cpu_percent}%, RAM: {ram_percent}%")
            if ram_percent > 90:
                logger.warning("⚠️ APPROACHING SYSTEM RAM LIMIT!")

            return current_mem, max_mem
        return 0, 0


class LSTMTrainer(Trainer):
    """Trainer for LSTM-based sentiment analysis models."""

    def _process_batch(
        self, batch: Dict, criterion: nn.Module
    ) -> Tuple[torch.Tensor, List, List]:
        """
        Process a batch of LSTM data.

        Args:
            batch: Batch of data with text, lengths, and labels
            criterion: Loss function

        Returns:
            Tuple containing loss, predictions and labels
        """
        # Move data to device
        text = batch["text"].to(self.device)
        lengths = batch["lengths"].to(self.device) if "lengths" in batch else None
        labels = batch["labels"].to(self.device)

        # Ensure all lengths are at least 1
        if lengths is not None:
            lengths = torch.clamp(lengths, min=1)

        # Forward pass
        outputs = self.model(text, lengths)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Get predictions
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()

        # Get labels
        labels = labels.detach().cpu().numpy()

        return loss, preds, labels

    def benchmark_training(
        self, model, train_loader, epochs=1, batches_per_epoch=100, device=None
    ):
        """Benchmark the model's training time per epoch.

        This function trains the model for a specified number of epochs and batches per epoch,
        measures the training time, and estimates the time required for a full training epoch
        using all batches in the training data loader.

        Args:
            model (torch.nn.Module): The neural network model to benchmark.
            train_loader (torch.utils.data.DataLoader): DataLoader providing the training data batches.
            epochs (int, optional): Number of epochs to run during benchmarking. Defaults to 1.
            batches_per_epoch (int, optional): Maximum number of batches processed per epoch during benchmarking.
                If the data loader has more batches, only this number is used each epoch. Defaults to 100.
            device (torch.device, optional): Target device for training (e.g., 'cuda', 'cpu'). If None, uses
                CUDA if available, otherwise CPU. Defaults to None.

        Returns:
            float: Estimated time in seconds for one full epoch using all batches in the training data loader.

        Notes:
            - Uses Adam optimizer with learning rate 0.001 and CrossEntropyLoss for training
            - Estimates epoch time by extrapolating average batch time to total number of batches
              in `train_loader` (actual epoch duration may vary due to system factors)
            - Model is set to training mode (`model.train()`) during benchmarking
            - Prints average time per batch and estimated epoch time in minutes
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        start_time = time.time()
        model.train()

        for epoch in range(epochs):
            for i, batch in enumerate(train_loader):
                if i >= batches_per_epoch:
                    break

                # Move data to device
                text = batch["text"].to(device)
                lengths = batch["lengths"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass
                outputs = model(text, lengths)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        elapsed = time.time() - start_time
        time_per_batch = elapsed / (epochs * batches_per_epoch)
        estimated_epoch_time = time_per_batch * (len(train_loader))

        print(f"Avg. time per batch: {time_per_batch:.4f}s")
        print(f"Estimated time per epoch: {estimated_epoch_time/60:.2f} minutes")

        return estimated_epoch_time


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

    def benchmark_training(
        self,
        train_dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        batches: int = 50,
        warmup_batches: int = 5,
        profile_memory: bool = True,
    ) -> Dict:
        """
        Benchmark training performance and memory usage.

        Args:
            train_dataloader: DataLoader for training data
            criterion: Loss function
            optimizer: Optimizer
            batches: Number of batches to process for benchmark
            warmup_batches: Number of warmup batches before timing
            profile_memory: Whether to profile GPU memory usage

        Returns:
            Dict containing benchmark results
        """
        self.model.train()
        device = self.device

        # ====== SAFEGUARD ======
        if torch.cuda.is_available():
            # Get batch dimensions without consuming the iterator
            first_batch = next(iter(train_dataloader))
            batch_size = first_batch["input_ids"].size(0)
            seq_len = first_batch["input_ids"].size(1)

            # Memory estimation formula for transformer models
            estimated_gb = (batch_size * seq_len**2 * 16) / (
                8e9
            )  # 16 bytes/param for mixed precision
            logger.info(
                f"Estimated memory requirement: ~{estimated_gb:.1f}GB (approximation)"
            )

            # Memory safety check
            max_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            if estimated_gb > max_mem * 0.8:
                logger.warning(
                    f"⚠️ Configuration might exceed GPU memory ({max_mem:.1f}GB)!"
                )
                response = input("Continue anyway? (y/n): ")
                if response.lower() != "y":
                    return {"error": "Aborted due to high memory requirements"}
        # ====== End of safeguard ======

        # Track memory usage if profiling enabled and using CUDA
        if profile_memory and torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated(device) / (1024**2)  # MB
            torch.cuda.reset_peak_memory_stats(device)

        # Warm-up phase
        logger.info(f"Starting warm-up with {warmup_batches} batches")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx % 5 == 0:  # Check every 5 batches
                current_mem, max_mem = self.monitor_system_resources(device)

            # Process batch (forward + backward)
            loss, _, _ = self._process_batch(batch, criterion)
            loss = loss / self.accumulation_steps
            loss.backward()

            # Step if accumulation complete
            if (batch_idx + 1) % self.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Reset gradients after warmup
        optimizer.zero_grad()

        # Benchmarking phase
        logger.info(f"Starting benchmark with {batches} batches")
        total_samples = 0
        batch_times = []

        # Start timing
        start_time = time.time()

        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx % 5 == 0:  # Check every 5 batches
                current_mem, max_mem = self.monitor_system_resources(device)
            if batch_idx >= batches:
                break

            batch_start = time.time()

            # Process batch
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Count samples
            batch_size = input_ids.size(0)
            total_samples += batch_size

            # Forward pass
            outputs = self.model(input_ids, attention_mask)

            # Calculate loss
            loss = criterion(outputs, labels)
            loss = loss / self.accumulation_steps

            # Backward pass
            loss.backward()

            # Update weights if accumulation complete
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Step optimizer
                optimizer.step()
                optimizer.zero_grad()

            # Record batch time
            batch_end = time.time()
            batch_times.append(batch_end - batch_start)

        # Ensure final update is applied if needed
        if batches % self.accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        # End timing
        end_time = time.time()
        total_time = end_time - start_time

        # Calculate memory usage
        memory_stats = {}
        if profile_memory and torch.cuda.is_available():
            memory_stats = {
                "peak_memory_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
                "final_memory_mb": torch.cuda.memory_allocated(device) / (1024**2),
                "memory_increase_mb": (torch.cuda.memory_allocated(device) / (1024**2))
                - initial_memory,
            }

        # Calculate statistics
        avg_time_per_batch = sum(batch_times) / len(batch_times)
        samples_per_second = total_samples / total_time

        # Prepare benchmark results
        results = {
            "total_batches": batches,
            "total_samples": total_samples,
            "total_time_seconds": total_time,
            "avg_time_per_batch_seconds": avg_time_per_batch,
            "samples_per_second": samples_per_second,
            "effective_batch_size": batch_size * self.accumulation_steps,
            "batch_times": batch_times,  # Detailed timing for each batch
            **memory_stats,
        }

        # Log results
        logger.info(f"Benchmark results for DistilBERT:")
        logger.info(f"  Samples/second: {samples_per_second:.2f}")
        logger.info(f"  Avg. batch time: {avg_time_per_batch*1000:.2f} ms")
        logger.info(f"  Effective batch size: {batch_size * self.accumulation_steps}")

        if profile_memory and torch.cuda.is_available():
            logger.info(f"  Peak GPU memory: {memory_stats['peak_memory_mb']:.2f} MB")
            logger.info(
                f"  Memory increase: {memory_stats['memory_increase_mb']:.2f} MB"
            )

        return results
