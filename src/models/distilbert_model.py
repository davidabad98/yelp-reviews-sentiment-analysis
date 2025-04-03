"""
DistilBERT model for sentiment analysis.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import DistilBertConfig, DistilBertModel

from ..config import DISTILBERT_CONFIG, NUM_CLASSES

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistilBERTSentimentModel(nn.Module):
    """DistilBERT-based model for sentiment analysis."""

    def __init__(
        self,
        pretrained_model_name: str = DISTILBERT_CONFIG["pretrained_model_name"],
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.1,
        freeze_bert_layers: Optional[int] = None,
    ):
        """
        Initialize DistilBERT model for sentiment classification.

        Args:
            pretrained_model_name: Name of the pretrained model
            num_classes: Number of sentiment classes
            dropout: Dropout probability
            freeze_bert_layers: Number of BERT layers to freeze (None = no freezing)
        """
        super(DistilBERTSentimentModel, self).__init__()

        # Load pre-trained DistilBERT model
        self.distilbert = DistilBertModel.from_pretrained(pretrained_model_name)

        # Get the hidden size from model config
        self.hidden_size = self.distilbert.config.hidden_size

        # Freeze BERT layers if specified
        if freeze_bert_layers is not None:
            self.freeze_bert_layers(freeze_bert_layers)

        # Attention pooling layer
        # self.attention_pool = BERTAttentionPool(self.hidden_size)

        # Classifier layers
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.hidden_size // 2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(self.hidden_size // 2, num_classes),
        # )

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_classes)

        logger.info(
            f"Initialized DistilBERT model with pretrained_model={pretrained_model_name}, "
            f"hidden_size={self.hidden_size}, num_classes={num_classes}, dropout={dropout}"
        )

    def freeze_bert_layers(self, num_layers: int):
        """
        Freeze specified number of BERT layers.

        Args:
            num_layers: Number of layers to freeze
        """
        # Freeze embeddings
        for param in self.distilbert.embeddings.parameters():
            param.requires_grad = False

        # Freeze transformer layers
        for layer_idx in range(min(num_layers, len(self.distilbert.transformer.layer))):
            for param in self.distilbert.transformer.layer[layer_idx].parameters():
                param.requires_grad = False

        logger.info(f"Froze {num_layers} DistilBERT layers and embeddings")

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask for padding [batch_size, seq_len]

        Returns:
            logits: Classification logits
            attention_weights: Attention weights for interpretability
        """
        # Get BERT outputs
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)

        # Get the hidden states
        # sequence_output = (
        #     outputs.last_hidden_state
        # )  # [batch_size, seq_len, hidden_size]

        # Apply attention pooling
        # pooled_output, attention_weights = self.attention_pool(
        #     sequence_output, attention_mask
        # )

        # # Apply classifier
        # logits = self.classifier(pooled_output)

        # return logits, attention_weights

        # Get the [CLS] token representation (first token)
        sequence_output = outputs.last_hidden_state[:, 0, :]

        # Apply dropout and classify
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return logits

    def predict(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Make predictions.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.

        Returns:
            Predicted class indices.
        """
        with torch.no_grad():
            logits = self(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1)
        return predictions

    def save(self, path: str):
        """
        Save the model.

        Args:
            path: Path to save the model.
        """
        # Save model state dict and configuration
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "pretrained_model_name": DISTILBERT_CONFIG["pretrained_model_name"],
                "num_classes": self.classifier.out_features,
                "hidden_size": self.hidden_size,
                "dropout": self.dropout.p,
            },
            path,
        )

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(
        cls, path: str, device: torch.device = None
    ) -> "DistilBERTSentimentClassifier":
        """
        Load the model.

        Args:
            path: Path to load the model from.
            device: Device to load the model to.

        Returns:
            Loaded model.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(path, map_location=device)

        # Create model with saved configuration
        model = cls(
            pretrained_model_name=checkpoint["pretrained_model_name"],
            num_classes=checkpoint["num_classes"],
            dropout=checkpoint["dropout"],
        )

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        logger.info(f"Model loaded from {path}")
        return model


class BERTAttentionPool(nn.Module):
    """Attention pooling for BERT sequence outputs."""

    def __init__(self, hidden_size: int):
        """
        Initialize attention pooling module.

        Args:
            hidden_size: Hidden size of BERT representations
        """
        super(BERTAttentionPool, self).__init__()

        # Attention query vector
        self.attention = nn.Linear(hidden_size, 1)

    def forward(
        self,
        sequence_output: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention pooling over sequence outputs.

        Args:
            sequence_output: BERT output sequences [batch_size, seq_len, hidden_size]
            attention_mask: Padding mask [batch_size, seq_len]

        Returns:
            pooled_output: Pooled vector after attention
            attention_weights: Attention weights for interpretability
        """
        # Compute attention scores
        attention_scores = self.attention(sequence_output).squeeze(
            -1
        )  # [batch_size, seq_len]

        # Apply mask to ignore padding
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(
            attention_scores, dim=-1
        )  # [batch_size, seq_len]

        # Apply attention weights to sequence
        pooled_output = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch_size, 1, seq_len]
            sequence_output,  # [batch_size, seq_len, hidden_size]
        ).squeeze(
            1
        )  # [batch_size, hidden_size]

        return pooled_output, attention_weights
