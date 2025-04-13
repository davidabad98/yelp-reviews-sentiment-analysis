"""
Inference utilities for sentiment analysis models.
"""

import logging
import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from transformers import DistilBertTokenizer

from src.config import SENTIMENT_MAP
from src.data.preprocessor import LSTMPreprocessor
from src.models.lstm_model import LSTMSentimentModel

from ..models.distilbert_model import DistilBERTSentimentModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentPredictor:
    """Base class for sentiment prediction."""

    def __init__(self, model_path: str, device: Optional[torch.device] = None):
        """
        Initialize predictor.

        Args:
            model_path: Path to model file
            device: Device to use
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.max_sequence_length = None
        self.sentiment_map = SENTIMENT_MAP

        logger.info(f"Initialized SentimentPredictor with device: {self.device}")

    def load_model(self):
        """Load model from disk."""
        raise NotImplementedError("Subclasses must implement load_model")

    def predict(self, text: str) -> Dict:
        """
        Predict sentiment for a text.

        Args:
            text: Input text

        Returns:
            Dictionary with prediction results
        """
        raise NotImplementedError("Subclasses must implement predict")

    def batch_predict(self, texts: List[str]) -> List[Dict]:
        """
        Predict sentiment for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of dictionaries with prediction results
        """
        return [self.predict(text) for text in texts]


class DistilBERTPredictor(SentimentPredictor):
    """Predictor for DistilBERT-based sentiment analysis."""

    def load_model(self) -> None:
        """Load pretrained DistilBERT model and tokenizer."""
        try:
            # Check if model path exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model path not found: {self.model_path}")

            # Load model
            logger.info(f"Loading DistilBERT model from {self.model_path}")
            self.model = DistilBERTSentimentModel.load(self.model_path)
            self.model.to(self.device)
            self.model.eval()

            # Load tokenizer
            tokenizer_path = os.path.join(os.path.dirname(self.model_path), "tokenizer")
            logger.info(f"Loading tokenizer from {tokenizer_path}")
            self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)

            # Set max sequence length from config
            self.max_sequence_length = (
                self.model.distilbert.config.max_position_embeddings
            )
            logger.info(f"Using max sequence length: {self.max_sequence_length}")

            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def preprocess(self, text: str) -> Dict:
        """
        Preprocess text for DistilBERT model.

        Args:
            text: Input text to preprocess

        Returns:
            Dictionary with input tensors
        """
        encoded_dict = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_sequence_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Move tensors to device
        input_ids = encoded_dict["input_ids"].to(self.device)
        attention_mask = encoded_dict["attention_mask"].to(self.device)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def predict(self, text: str) -> Dict:
        """
        Predict sentiment for a text using DistilBERT.

        Args:
            text: Input text

        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            self.load_model()

        # Ensure model is in evaluation mode
        self.model.eval()

        # Preprocess text
        inputs = self.preprocess(text)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            probs = torch.softmax(logits, dim=1).squeeze()
            predicted_class = torch.argmax(probs).item()

        # Convert to numpy for easier handling
        probabilities = probs.cpu().numpy()

        # Create result dictionary
        result = {
            "text": text,
            "sentiment": self.sentiment_map[predicted_class],
            "sentiment_id": predicted_class,
            "confidence": float(probabilities[predicted_class]),
            "probabilities": {
                self.sentiment_map[i]: float(prob)
                for i, prob in enumerate(probabilities)
            },
        }

        return result

    def batch_predict(self, texts: List[str], batch_size: int = 8) -> List[Dict]:
        """
        Predict sentiment for multiple texts in batches.

        Args:
            texts: List of input texts
            batch_size: Batch size for processing

        Returns:
            List of dictionaries with prediction results
        """
        if self.model is None:
            self.load_model()

        results = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Encode batch
            encoded_dict = self.tokenizer.batch_encode_plus(
                batch_texts,
                add_special_tokens=True,
                max_length=self.max_sequence_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            # Move tensors to device
            input_ids = encoded_dict["input_ids"].to(self.device)
            attention_mask = encoded_dict["attention_mask"].to(self.device)

            # Make predictions
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                probs = torch.softmax(logits, dim=1)
                predicted_classes = torch.argmax(probs, dim=1).cpu().numpy()

            # Convert probabilities to numpy
            probabilities = probs.cpu().numpy()

            # Create result dictionaries
            for j, text in enumerate(batch_texts):
                pred_class = predicted_classes[j]
                result = {
                    "text": text,
                    "sentiment": self.sentiment_map[pred_class],
                    "sentiment_id": int(pred_class),
                    "confidence": float(probabilities[j][pred_class]),
                    "probabilities": {
                        self.sentiment_map[k]: float(prob)
                        for k, prob in enumerate(probabilities[j])
                    },
                }
                results.append(result)

        return results


class LSTMPredictor(SentimentPredictor):
    """Predictor for LSTM-based sentiment analysis."""

    def load_model(self) -> None:
        """Load pretrained LSTM model."""
        try:
            # Check if model path exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model path not found: {self.model_path}")

            # Load model
            logger.info(f"Loading LSTM model from {self.model_path}")
            self.model = LSTMSentimentModel.load(self.model_path)
            self.model.to(self.device)
            self.model.eval()

            # Load vocabulary (word_to_idx dictionary)
            vocab_path = os.path.join(os.path.dirname(self.model_path), "vocab.pt")
            if os.path.exists(vocab_path):
                try:
                    self.word_to_idx = torch.load(vocab_path)
                    logger.info(
                        f"Loaded vocabulary with {len(self.word_to_idx)} tokens"
                    )

                    # Create a preprocessor with the loaded vocabulary
                    self.preprocessor = LSTMPreprocessor()
                    self.preprocessor.word_to_idx = self.word_to_idx
                    self.preprocessor.idx_to_word = {
                        idx: word for word, idx in self.word_to_idx.items()
                    }
                    self.preprocessor.vocab_size = len(self.word_to_idx)

                except Exception as e:
                    logger.error(f"Error loading vocabulary: {e}")
                    raise ValueError(f"Failed to load vocabulary: {e}")
            else:
                logger.warning(f"Vocabulary not found at {vocab_path}")
                raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")

            # Set max sequence length from config or default
            self.max_sequence_length = getattr(self.model, "max_seq_length", 256)
            logger.info(f"Using max sequence length: {self.max_sequence_length}")

            self.pad_idx = self.word_to_idx.get(
                "<PAD>", 0
            )  # Get from vocab or default to 0

            logger.info("LSTM model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def preprocess(self, text: str) -> Dict:
        """
        Preprocess text for LSTM model.

        Args:
            text: Input text to preprocess

        Returns:
            Dictionary with input tensor
        """
        try:
            # Clean the text using the LSTM preprocessor
            tokenized_sample = self.preprocessor.preprocess([text])
            # Tokenize into tokens
            tokens = tokenized_sample[0]

            # Convert tokens to indices, handling unknown tokens with <UNK>
            unk_idx = self.word_to_idx.get(
                "<UNK>", 0
            )  # Fallback to 0 if <UNK> not present
            sequence = [self.word_to_idx.get(token, unk_idx) for token in tokens]

            # Truncate or pad sequence to max_sequence_length
            if len(sequence) > self.max_sequence_length:
                sequence = sequence[: self.max_sequence_length]
            else:
                # Pad with zeros (assuming padding index is 0)
                sequence += [0] * (self.max_sequence_length - len(sequence))

            # Convert to tensor and add batch dimension
            input_tensor = (
                torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(self.device)
            )

            return {"input_ids": input_tensor}

        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            raise

    def predict(self, text: str) -> Dict:
        """
        Predict sentiment for a text using LSTM.

        Args:
            text: Input text

        Returns:
            Dictionary with prediction results
        """
        try:
            inputs = self.preprocess(text)
            input_ids = inputs["input_ids"]

            # Model inference
            with torch.no_grad():
                logits = self.model(input_ids)

            # Convert logits to probabilities
            probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            predicted_class = np.argmax(probabilities)

            return {
                "text": text,
                "sentiment": self.sentiment_map[predicted_class],
                "sentiment_id": predicted_class,
                "confidence": float(probabilities[predicted_class]),
                "probabilities": {
                    self.sentiment_map[i]: float(prob)
                    for i, prob in enumerate(probabilities)
                },
                "logits": logits.squeeze().cpu().numpy().tolist(),
            }

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def batch_predict(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Batch predict sentiments using LSTM for efficiency.

        Args:
            texts: List of input texts
            batch_size: Batch size for processing

        Returns:
            List of prediction dictionaries with original text and formatted probabilities
        """
        try:
            # Validate inputs
            if not texts:
                return []

            # Process all texts first (could be batched for memory optimization)
            processed = [self.preprocess(text) for text in texts]
            input_ids = torch.cat([p["input_ids"] for p in processed], dim=0)

            # Split into batches for memory efficiency
            results = []
            for i in range(0, len(input_ids), batch_size):
                batch_inputs = input_ids[i : i + batch_size].to(self.device)

                # Batch inference
                with torch.no_grad():
                    batch_logits = self.model(batch_inputs)

                batch_probs = torch.softmax(batch_logits, dim=1).cpu().numpy()
                batch_preds = np.argmax(batch_probs, axis=1)

                # Create results for this batch
                for j in range(len(batch_inputs)):
                    original_idx = i + j
                    if original_idx >= len(texts):  # Handle edge case
                        continue

                    pred_class = batch_preds[j]
                    results.append(
                        {
                            "text": texts[original_idx],  # Include original text
                            "sentiment": self.sentiment_map.get(pred_class, "Neutral"),
                            "sentiment_id": int(pred_class),
                            "confidence": float(batch_probs[j][pred_class]),
                            "probabilities": {
                                label: float(prob)
                                for label, prob in zip(
                                    self.sentiment_map.values(), batch_probs[j]
                                )
                            },
                            "logits": batch_logits[j].cpu().numpy().tolist(),
                        }
                    )

            return results

        except Exception as e:
            logger.error(f"Error during batch prediction: {str(e)}")
            raise
