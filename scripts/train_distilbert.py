#!/usr/bin/env python
"""
Training script for DistilBERT sentiment analysis model.
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from transformers import AdamW, DistilBertTokenizer, get_linear_schedule_with_warmup

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DISTILBERT_CONFIG, NUM_CLASSES, RANDOM_SEED
from src.data.data_loader import YelpDataLoader
from src.data.dataset import create_data_loaders
from src.models.distilbert_model import DistilBERTSentimentModel
from src.training.trainer import DistilBERTTrainer
from src.utils.logger import setup_logger
from src.utils.visualization import plot_confusion_matrix, plot_training_history

# Set up argument parser
parser = argparse.ArgumentParser(description="Train DistilBERT for sentiment analysis")
parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
parser.add_argument(
    "--weight_decay", type=float, default=0.01, help="Weight decay for AdamW"
)
parser.add_argument(
    "--max_seq_length",
    type=int,
    default=None,
    help="Maximum sequence length (defaults to value in config)",
)
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
parser.add_argument(
    "--freeze_layers",
    type=int,
    default=None,
    help="Number of DistilBERT layers to freeze (None = fine-tune all)",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="models/distilbert",
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
    "--warmup_ratio",
    type=float,
    default=0.1,
    help="Fraction of steps for learning rate warmup",
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

    # Validate path is writable by saving and reading a test file
    try:
        test_file_path = os.path.join(args.output_dir, "path_test.txt")
        with open(test_file_path, "w") as f:
            f.write("Path validation test")

        # Verify we can read it back
        with open(test_file_path, "r") as f:
            content = f.read()

        # Clean up test file
        os.remove(test_file_path)

        print(f"✓ Output directory {args.output_dir} is writable.")
    except Exception as e:
        print(f"ERROR: Cannot write to output directory {args.output_dir}")
        print(f"Error details: {str(e)}")

    # Set up logging
    log_file = os.path.join(
        "logs", f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logger = setup_logger("distilbert_training", log_file)

    logger.info(f"Starting DistilBERT training with arguments: {args}")

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set maximum sequence length
    max_seq_length = args.max_seq_length or DISTILBERT_CONFIG["max_sequence_length"]
    logger.info(f"Using maximum sequence length: {max_seq_length}")

    # Initialize tokenizer
    logger.info(
        f"Initializing DistilBERT tokenizer: {DISTILBERT_CONFIG['pretrained_model_name']}"
    )
    tokenizer = DistilBertTokenizer.from_pretrained(
        DISTILBERT_CONFIG["pretrained_model_name"]
    )

    # Load data
    logger.info("Loading and processing data...")
    data_loader = YelpDataLoader()
    train_df, test_df = data_loader.load_processed_data()

    logger.info(f"Train set shape: {train_df.shape}")
    logger.info(f"Test set shape: {test_df.shape}")

    # Create data loaders
    logger.info(
        f"Creating data loaders with batch size {DISTILBERT_CONFIG['batch_size']}..."
    )

    loaders = create_data_loaders(
        train_df, test_df, 0.1, "distilbert", tokenizer=tokenizer
    )

    train_dataloader = loaders["train"]
    val_dataloader = loaders["val"]
    test_dataloader = loaders["test"]

    # Initialize model
    logger.info("Initializing DistilBERT model...")
    model = DistilBERTSentimentModel(
        pretrained_model_name=DISTILBERT_CONFIG["pretrained_model_name"],
        num_classes=NUM_CLASSES,
        dropout=args.dropout,
        freeze_bert_layers=args.freeze_layers,
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

    # Initialize optimizer with weight decay
    logger.info(
        f"Initializing AdamW optimizer with lr={args.lr}, weight_decay={args.weight_decay}"
    )
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    # Calculate total steps and warmup steps
    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)

    # Create scheduler
    logger.info(f"Creating learning rate scheduler with {warmup_steps} warmup steps")
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scheduler.is_batch_level_scheduler = True

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize trainer
    trainer = DistilBERTTrainer(
        model, device, accumulation_steps=4, use_amp=True  # Enable mixed precision
    )

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
        model_name="distilbert",
        early_stopping_patience=args.early_stopping,
        checkpoint_interval=15,  # Minutes between checkpoints
    )

    # Save tokenizer for inference
    tokenizer_path = os.path.join(args.output_dir, "tokenizer")
    os.makedirs(tokenizer_path, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"Tokenizer saved to {tokenizer_path}")

    # Plot training history
    if args.plot_results:
        logger.info("Generating training plots...")
        history_plot_path = os.path.join(args.output_dir, "training_history.png")
        plot_training_history(history, figsize=(12, 5), save_path=history_plot_path)

    # Load best model for evaluation
    # best_model_path = trainer.best_model_path
    # logger.info(f"Loading best model from {best_model_path}")
    # best_model = DistilBERTSentimentModel.load(best_model_path, device)

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics, test_preds, test_labels = trainer.evaluate(test_dataloader, criterion)

    logger.info(f"Test Results:")
    logger.info(f"  Loss: {test_metrics['loss']:.4f}")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  F1 Score: {test_metrics['f1']:.4f}")

    # Update config with test metrics
    # config = torch.load(config_path)
    # config["test_metrics"] = test_metrics
    # torch.save(config, config_path)

    # Plot confusion matrix
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
