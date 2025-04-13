"""
LSTM model for sentiment analysis.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import LSTM_CONFIG, NUM_CLASSES

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMSentimentModel(nn.Module):
    """LSTM model for sentiment analysis."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = LSTM_CONFIG["embedding_dim"],
        hidden_dim: int = LSTM_CONFIG["hidden_dim"],
        num_layers: int = LSTM_CONFIG["num_layers"],
        dropout: float = LSTM_CONFIG["dropout"],
        bidirectional: bool = LSTM_CONFIG["bidirectional"],
        num_classes: int = NUM_CLASSES,
        pretrained_embeddings: Optional[np.ndarray] = None,
        freeze_embeddings: bool = False,
        padding_idx: int = 0,
    ):
        """
        Initialize LSTM model for sentiment classification.

        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of LSTM units
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            num_classes: Number of sentiment classes
            pretrained_embeddings: Optional pretrained word embeddings
            freeze_embeddings: Whether to freeze embedding weights
        """
        super(LSTMSentimentModel, self).__init__()

        # Embedding layer
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_embeddings),
                padding_idx=padding_idx,
                freeze=freeze_embeddings,
            )
            embedding_dim = pretrained_embeddings.shape[1]
            logger.info(
                f"Loaded pretrained embeddings with shape {pretrained_embeddings.shape}"
            )

        else:
            self.embedding = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=padding_idx
            )

        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Attention mechanism
        # self.attention = SelfAttention(lstm_output_dim)

        # Dropout layer
        # self.dropout = nn.Dropout(dropout)

        # Classification layer
        # self.classifier = nn.Sequential(
        #     nn.Linear(lstm_output_dim, lstm_output_dim // 2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(lstm_output_dim // 2, num_classes),
        # )

        # Fully connected layers
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        # Initialize weights
        self._init_weights()

        logger.info(
            f"Initialized LSTM model with vocab_size={vocab_size}, "
            f"embedding_dim={embedding_dim}, hidden_dim={hidden_dim}, "
            f"num_layers={num_layers}, dropout={dropout}, "
            f"bidirectional={bidirectional}, num_classes={num_classes}"
        )

    def _init_weights(self):
        """Initialize model weights."""
        # for name, param in self.lstm.named_parameters():
        #     if "weight" in name:
        #         nn.init.orthogonal_(param)
        #     elif "bias" in name:
        #         nn.init.constant_(param, 0.0)

        # for name, param in self.classifier.named_parameters():
        #     if "weight" in name:
        #         nn.init.xavier_uniform_(param)
        #     elif "bias" in name:
        #         nn.init.constant_(param, 0.0)

        for name, param in self.named_parameters():
            if "embedding" not in name:
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def forward(
        self, input_ids: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            lengths: Optional lengths of sequences for packing

        Returns:
            logits: Classification logits
            attention_weights: Attention weights for interpretability
        """
        # Input shape: [batch_size, seq_len]

        # Get embeddings
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]

        # Create a padding mask where 1 indicates non-padding tokens
        # mask = (input_ids != 0).float()  # [batch_size, seq_len]

        # Pack sequences if lengths are provided
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        # LSTM: [batch_size, seq_len, hidden_dim * (2 if bidirectional else 1)]
        lstm_out, (hidden, _) = self.lstm(embedded)

        # Get final hidden state
        if self.lstm.bidirectional:
            # Concatenate the last hidden state from both directions
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        # Unpack if packed
        # if lengths is not None:
        #     lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
        #         lstm_output, batch_first=True
        #     )

        # Apply attention
        # context, attention_weights = self.attention(lstm_output, mask)

        # Apply dropout
        # context = self.dropout(context)

        # Classification
        # logits = self.classifier(context)

        # return logits, attention_weights

        # FC layers
        out = F.relu(self.fc1(hidden))
        out = self.dropout(out)
        logits = self.fc2(out)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions.

        Args:
            x: Input tensor.

        Returns:
            Predicted class indices.
        """
        with torch.no_grad():
            logits = self(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions

    def save(self, path: str):
        """
        Save the model.

        Args:
            path: Path to save the model.
        """
        # Save model parameters and configuration
        state_dict = {
            "vocab_size": self.embedding.weight.size(0),
            "embedding_dim": self.embedding.weight.size(1),
            "hidden_dim": self.lstm.hidden_size,
            "num_layers": self.lstm.num_layers,
            "dropout": self.dropout.p,
            "bidirectional": self.lstm.bidirectional,
            "num_classes": self.fc2.out_features,
            "model_state_dict": self.state_dict(),
        }
        torch.save(state_dict, path)

    @classmethod
    def load(cls, path: str, device: torch.device = None) -> "LSTMSentimentClassifier":
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

        state_dict = torch.load(path, map_location=device)

        # Create model with saved configuration
        model = cls(
            vocab_size=state_dict["vocab_size"],
            embedding_dim=state_dict["embedding_dim"],
            hidden_dim=state_dict["hidden_dim"],
            num_layers=state_dict["num_layers"],
            dropout=state_dict["dropout"],
            bidirectional=state_dict["bidirectional"],
            num_classes=state_dict["num_classes"],
        )

        # Load weights
        model.load_state_dict(state_dict["model_state_dict"])
        model.to(device)

        logger.info(f"Model loaded from {path}")
        return model


class SelfAttention(nn.Module):
    """Self-attention mechanism for sequence modeling."""

    def __init__(self, hidden_dim: int):
        """
        Initialize self-attention module.

        Args:
            hidden_dim: Hidden dimension size
        """
        super(SelfAttention, self).__init__()

        # Attention projection
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(
        self, sequence_output: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply self-attention over sequence outputs.

        Args:
            sequence_output: LSTM output sequences [batch_size, seq_len, hidden_dim]
            mask: Padding mask [batch_size, seq_len]

        Returns:
            context: Context vector after attention
            attention_weights: Attention weights for interpretability
        """
        # Compute attention scores
        attention_scores = self.attention(sequence_output).squeeze(
            -1
        )  # [batch_size, seq_len]

        # Apply mask to ignore padding
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, seq_len]

        # Apply attention weights to sequence
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch_size, 1, seq_len]
            sequence_output,  # [batch_size, seq_len, hidden_dim]
        ).squeeze(
            1
        )  # [batch_size, hidden_dim]

        return context, attention_weights
