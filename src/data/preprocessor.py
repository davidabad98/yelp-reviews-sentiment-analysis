"""
Text preprocessing utilities for sentiment analysis.
"""

import logging
import re
import string
from collections import Counter
from typing import Dict, List, Optional, Union

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import DistilBertTokenizer

from ..config import DISTILBERT_CONFIG, LSTM_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download necessary NLTK resources
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
    nltk.data.find("corpora/wordnet")
except LookupError:
    logger.info("Downloading necessary NLTK resources...")
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")


class TextPreprocessor:
    """Text preprocessing class for cleaning and normalizing text data."""

    def __init__(
        self,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_stopwords: bool = False,
        lemmatize: bool = True,
        lowercase: bool = True,
        min_word_length: int = 2,
        custom_stopwords: Optional[List[str]] = None,
    ):
        """
        Initialize the text preprocessor with specified options.

        Args:
            remove_punctuation: Whether to remove punctuation
            remove_numbers: Whether to remove numeric characters
            remove_stopwords: Whether to remove common stopwords
            lemmatize: Whether to apply lemmatization
            lowercase: Whether to convert text to lowercase
            min_word_length: Minimum word length to keep
            custom_stopwords: Additional stopwords to remove
        """
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self.min_word_length = min_word_length

        # Set up stopwords
        self.stop_words = set(stopwords.words("english"))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)

        # Initialize lemmatizer
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        """
        Clean the text by removing special characters, URLs, and normalizing.

        Args:
            text: Raw text to clean.

        Returns:
            Cleaned text.
        """
        # Handle empty or None text
        if not text or not isinstance(text, str):
            return ""

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)

        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        # Remove special characters and numbers
        if self.remove_numbers:
            text = re.sub(r"\d+", "", text)
            text = re.sub(r"[^\w\s]", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _process_token(self, token: str) -> str:
        """Internal token-level processing"""
        if len(token) < self.min_word_length:
            return None

        # Remove stopwords if configured
        if self.remove_stopwords and token in self.stop_words:
            return None

        # Apply lemmatization if configured
        if self.lemmatize:
            token = self.lemmatizer.lemmatize(token)

        return token

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the text.

        Args:
            text: Text to tokenize.

        Returns:
            List of tokens.
        """
        # Clean the text first
        text = self.clean_text(text)

        # Tokenize
        tokens = nltk.word_tokenize(text)

        return [t for t in (self._process_token(t) for t in tokens) if t is not None]

    def preprocess(self, texts: List[str]) -> List[List[str]]:
        """
        Preprocess a list of texts.

        Args:
            texts: List of raw texts.

        Returns:
            List of tokenized texts.
        """
        return [self.tokenize(text) for text in texts]


class LSTMPreprocessor(TextPreprocessor):
    """Preprocessor specifically for LSTM models."""

    def __init__(
        self,
        max_vocab_size: int = LSTM_CONFIG["max_vocab_size"],
        max_sequence_length: int = LSTM_CONFIG["max_sequence_length"],
        remove_stopwords: bool = True,
        lemmatize: bool = True,
    ):
        """
        Initialize the LSTM preprocessor.

        Args:
            max_vocab_size: Maximum vocabulary size.
            max_sequence_length: Maximum sequence length.
            remove_stopwords: Whether to remove stopwords.
            lemmatize: Whether to apply lemmatization.
        """
        super().__init__(remove_stopwords, lemmatize)
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.word_to_idx = None
        self.idx_to_word = None
        self.vocab_size = None

    def fit(self, tokenized_texts: List[List[str]]) -> Dict[str, int]:
        """
        Build vocabulary from tokenized texts.

        Args:
            tokenized_texts: List of tokenized texts.

        Returns:
            Dictionary mapping words to indices.
        """
        logger.info("Building vocabulary...")

        # Count word frequencies
        word_counts = Counter([word for text in tokenized_texts for word in text])

        # Sort by frequency and take top words
        most_common = word_counts.most_common(
            self.max_vocab_size - 2
        )  # -2 for <PAD> and <UNK>

        # Create word to index mapping
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.word_to_idx.update(
            {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
        )

        # Create index to word mapping
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

        # Update vocab size
        self.vocab_size = len(self.word_to_idx)

        logger.info(f"Vocabulary built with {self.vocab_size} words")

        return self.word_to_idx

    def texts_to_sequences(self, tokenized_texts: List[List[str]]) -> List[List[int]]:
        """
        Convert tokenized texts to sequences of indices.

        Args:
            tokenized_texts: List of tokenized texts.

        Returns:
            List of sequences of indices.
        """
        if self.word_to_idx is None:
            raise ValueError("Vocabulary not built. Call build_vocabulary first.")

        sequences = []
        for text in tokenized_texts:
            sequence = [
                self.word_to_idx.get(word, self.word_to_idx["<UNK>"]) for word in text
            ]
            sequences.append(sequence)

        return sequences

    def pad_sequences(self, sequences: List[List[int]]) -> np.ndarray:
        """
        Pad sequences to the same length.

        Args:
            sequences: List of sequences.

        Returns:
            Padded sequences as numpy array.
        """
        padded_sequences = []
        for sequence in sequences:
            # Truncate or pad as needed
            if len(sequence) > self.max_sequence_length:
                padded_sequence = sequence[: self.max_sequence_length]
            else:
                # return sequence + [self.word_to_idx["<PAD>"]] * (
                #     self.max_sequence_length - len(sequence)
                # )
                padded_sequence = sequence + [0] * (
                    self.max_sequence_length - len(sequence)
                )
            padded_sequences.append(padded_sequence)

        return np.array(padded_sequences)

    def preprocess_for_lstm(self, texts: List[str], fit: bool = False) -> np.ndarray:
        """
        Preprocess texts for LSTM model.

        Args:
            texts: List of raw texts.
            fit: Whether to build vocabulary (for training data).

        Returns:
            Preprocessed texts as numpy array.
        """
        # Tokenize texts
        tokenized_texts = self.preprocess(texts)

        # Build vocabulary if needed
        if fit or self.word_to_idx is None:
            self.fit(tokenized_texts)

        # Convert to sequences
        sequences = self.texts_to_sequences(tokenized_texts)

        # Pad sequences
        padded_sequences = self.pad_sequences(sequences)

        return padded_sequences


class DistilBERTPreprocessor:
    """Preprocessor for BERT-based models."""

    def __init__(
        self,
        tokenizer: DistilBertTokenizer,
        pretrained_model_name: str = DISTILBERT_CONFIG["pretrained_model_name"],
        max_sequence_length: int = DISTILBERT_CONFIG["max_sequence_length"],
    ):
        """
        Initialize the DistilBERT preprocessor.

        Args:
            pretrained_model_name: Pretrained model name.
            max_sequence_length: Maximum sequence length.
        """
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        logger.info(
            f"Initialized DistilBERT preprocessor with {pretrained_model_name} tokenizer"
        )

    def preprocess_for_distilbert(
        self,
        texts: Union[str, List[str]],
        padding: bool = True,
        truncation: bool = True,
    ):
        """
        Preprocess texts for BERT models.

        Args:
            texts: Input text or list of texts
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences

        Returns:
            Dictionary with input_ids, attention_mask.

        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize and encode
        encoded = self.tokenizer(
            texts,
            # padding=padding,
            padding="max_length",
            truncation=truncation,
            max_length=self.max_sequence_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
