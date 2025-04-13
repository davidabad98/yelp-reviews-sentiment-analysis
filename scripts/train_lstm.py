#!/usr/bin/env python
"""
Training script for LSTM sentiment analysis model.
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import LSTM_CONFIG, NUM_CLASSES, RANDOM_SEED
from src.data.data_loader import YelpDataLoader
from src.data.dataset import create_data_loaders
from src.models.lstm_model import LSTMSentimentModel
from src.training.trainer import LSTMTrainer
from src.utils.logger import setup_logger
from src.utils.visualization import plot_confusion_matrix, plot_training_history

# Set up argument parser
parser = argparse.ArgumentParser(description="Train LSTM for sentiment analysis")
parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument(
    "--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer"
)
parser.add_argument(
    "--embedding_dim",
    type=int,
    default=None,
    help="Dimension of word embeddings (defaults to value in config)",
)
parser.add_argument(
    "--hidden_dim",
    type=int,
    default=None,
    help="Dimension of LSTM hidden state (defaults to value in config)",
)
parser.add_argument(
    "--num_layers",
    type=int,
    default=None,
    help="Number of LSTM layers (defaults to value in config)",
)
parser.add_argument(
    "--bidirectional", action="store_true", help="Use bidirectional LSTM"
)
parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
parser.add_argument(
    "--max_vocab_size",
    type=int,
    default=None,
    help="Maximum vocabulary size (defaults to value in config)",
)
parser.add_argument(
    "--max_seq_length",
    type=int,
    default=None,
    help="Maximum sequence length (defaults to value in config)",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="models/lstm",
    help="Directory to save model and results",
)
parser.add_argument(
    "--validation_split",
    type=float,
    default=0.1,
    help="Fraction of training data to use for validation",
)
parser.add_argument(
    "--early_stopping",
    type=int,
    default=3,
    help="Number of epochs to wait for improvement before early stopping",
)
parser.add_argument(
    "--use_pretrained_embeddings",
    action="store_true",
    help="Use pretrained word embeddings",
)
parser.add_argument(
    "--freeze_embeddings",
    action="store_true",
    help="Freeze pretrained word embeddings during training",
)
parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
parser.add_argument(
    "--num_workers", type=int, default=4, help="Number of workers for data loading"
)
parser.add_argument(
    "--save_every", type=int, default=None, help="Save model every N epochs"
)
parser.add_argument(
    "--plot_results", action="store_true", default=True, help="Generate and save plots"
)


def main():
    """Main training function."""
    # Parse arguments
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging
    log_file = os.path.join(
        "logs", f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logger = setup_logger("lstm_training", log_file)

    logger.info(f"Starting LSTM training with arguments: {args}")

    if args.plot_results:
        logger.info("Generating training plots...")

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set maximum sequence length and other hyperparameters
    max_seq_length = args.max_seq_length or LSTM_CONFIG.get("max_sequence_length", 256)
    embedding_dim = args.embedding_dim or LSTM_CONFIG.get("embedding_dim", 300)
    hidden_dim = args.hidden_dim or LSTM_CONFIG.get("hidden_dim", 256)
    num_layers = args.num_layers or LSTM_CONFIG.get("num_layers", 2)
    bidirectional = args.bidirectional or LSTM_CONFIG.get("bidirectional", False)
    max_vocab_size = args.max_vocab_size or LSTM_CONFIG.get("max_vocab_size", 30000)

    logger.info(f"Using maximum sequence length: {max_seq_length}")
    logger.info(f"Using embedding dimension: {embedding_dim}")
    logger.info(f"Using hidden dimension: {hidden_dim}")
    logger.info(f"Using LSTM layers: {num_layers}")
    logger.info(f"Using bidirectional LSTM: {bidirectional}")
    logger.info(f"Using maximum vocabulary size: {max_vocab_size}")

    # Load data
    logger.info("Loading and processing data...")
    data_loader = YelpDataLoader()
    train_df, test_df = data_loader.load_processed_data()

    logger.info(f"Train set shape: {train_df.shape}")
    logger.info(f"Test set shape: {test_df.shape}")

    # Create data loaders
    logger.info(f"Creating data loaders with batch size {args.batch_size}...")

    loaders = create_data_loaders(
        train_df,
        test_df,
        args.validation_split,
        "lstm",
        # max_seq_length=max_seq_length,
        # max_vocab_size=max_vocab_size,
        batch_size=args.batch_size,
        # num_workers=args.num_workers,
    )

    train_dataloader = loaders["train"]
    val_dataloader = loaders["val"]
    test_dataloader = loaders["test"]
    vocab = loaders.get("vocab")  # Get vocabulary for model initialization

    # Get vocabulary size
    vocab_size = LSTM_CONFIG["max_vocab_size"]
    logger.info(f"Vocabulary size: {vocab_size}")

    # Initialize model
    logger.info("Initializing LSTM model...")
    model = LSTMSentimentModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        bidirectional=bidirectional,
        num_classes=NUM_CLASSES,
        dropout=args.dropout,
        # padding_idx=vocab.get("<PAD>", 0),
        padding_idx=0,
        pretrained_embeddings=None,  # TODO: Add support for pretrained embeddings if needed
    )

    # Move model to device
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(
        f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.1%})"
    )

    # Initialize optimizer
    logger.info(
        f"Initializing Adam optimizer with lr={args.lr}, weight_decay={args.weight_decay}"
    )
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Initialize scheduler
    logger.info("Initializing learning rate scheduler (StepLR)")
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
    scheduler.is_batch_level_scheduler = False

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize trainer
    trainer = LSTMTrainer(model, device, accumulation_steps=4)

    # Train model
    logger.info(f"Starting training for {args.epochs} epochs...")
    history = trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        output_dir=args.output_dir,
        model_name="lstm",
        early_stopping_patience=args.early_stopping,
    )

    # Save vocabulary for inference
    vocab_path = os.path.join(args.output_dir, "vocab.pt")
    torch.save(vocab, vocab_path)
    logger.info(f"Vocabulary saved to {vocab_path}")

    # Plot training history
    if args.plot_results:
        logger.info("Generating training plots...")
        history_plot_path = os.path.join(args.output_dir, "training_history.png")
        plot_training_history(history, figsize=(12, 5), save_path=history_plot_path)

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics, test_preds, test_labels = trainer.evaluate(test_dataloader, criterion)

    logger.info(f"Test Results:")
    logger.info(f"  Loss: {test_metrics['loss']:.4f}")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  F1 Score: {test_metrics['f1']:.4f}")

    # # Plot confusion matrix
    # if args.plot_results:
    #     logger.info("Generating confusion matrix...")
    #     cm_plot_path = os.path.join(args.output_dir, "confusion_matrix.png")
    #     plot_confusion_matrix(
    #         test_labels,
    #         test_preds,
    #         classes=["Negative", "Neutral", "Positive"],
    #         save_path=cm_plot_path,
    #     )

    logger.info("Training completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
