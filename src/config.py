"""
Configuration file for the Yelp Reviews Sentiment Analysis project.
"""

import os
from pathlib import Path

# Base directories
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Create directories if they don't exist
for dir_path in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    EMBEDDINGS_DIR,
    MODELS_DIR,
    LOGS_DIR,
]:
    os.makedirs(dir_path, exist_ok=True)

# Data file paths
RAW_TRAIN_PATH = os.path.join(RAW_DATA_DIR, "train.parquet")
RAW_TEST_PATH = os.path.join(RAW_DATA_DIR, "test.parquet")
PROCESSED_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "train_with_sentiment.parquet")
PROCESSED_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "test_with_sentiment.parquet")

# Hugging Face dataset info
HF_DATASET = "Yelp/yelp_review_full"
HF_SPLITS = {
    "train": "train-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}

# Model parameters
# LSTM parameters
LSTM_CONFIG = {
    "embedding_dim": 300,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.3,
    "bidirectional": True,
    "max_vocab_size": 50000,
    "max_sequence_length": 512,
    "batch_size": 64,
    "learning_rate": 0.001,
    "epochs": 10,
}

# DistilBERT parameters
DISTILBERT_CONFIG = {
    "pretrained_model_name": "distilbert-base-uncased",
    "max_sequence_length": 256,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "epochs": 3,
}

# Hyperparameter tuning
TUNING_CONFIG = {
    "lstm": {
        "learning_rate": [1e-3, 1e-4],
        "hidden_units": [64, 128],
        "dropout": [0.3, 0.5],
    },
    "distilbert": {"learning_rate": [2e-5, 5e-5], "batch_size": [16, 32]},
}

# General training parameters
RANDOM_SEED = 42
NUM_CLASSES = 3  # Negative, Neutral, Positive
VALIDATION_SPLIT = 0.1
USE_CUDA = True

# Mapping from original star ratings to sentiment classes
RATING_TO_SENTIMENT = {
    0: 0,  # Negative
    1: 0,  # Negative
    2: 1,  # Neutral
    3: 2,  # Positive
    4: 2,  # Positive
}
# Sentiment map used for Prediction Inference
SENTIMENT_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Class names for visualization and reporting
CLASS_NAMES = ["Negative", "Neutral", "Positive"]
