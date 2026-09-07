"""Small untrained Transformer encoder-decoder architecture for Phase 5."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from src.tokenizer_utils import SentencePieceTokenizer


class ModelConfigurationError(ValueError):
    """Raised when model settings cannot safely construct a Transformer."""


def validate_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the deliberately small Transformer settings before construction."""
    model = config.get("model")
    if not isinstance(model, dict):
        raise ModelConfigurationError("Missing model configuration.")
    integer_keys = ("d_model", "attention_heads", "encoder_layers", "decoder_layers", "feedforward_dimension", "max_sequence_length")
    for key in integer_keys:
        if isinstance(model.get(key), bool) or not isinstance(model.get(key), int) or model[key] <= 0:
            raise ModelConfigurationError(f"model.{key} must be a positive integer.")
    if model["d_model"] % model["attention_heads"]:
        raise ModelConfigurationError("model.d_model must be divisible by model.attention_heads.")
    if not isinstance(model.get("dropout"), (int, float)) or not 0 <= model["dropout"] < 1:
        raise ModelConfigurationError("model.dropout must be at least 0 and less than 1.")
    return model


class PositionalEncoding(nn.Module):
    """Sinusoidal position information for batch-first embedding tensors."""

    def __init__(self, d_model: int, max_sequence_length: int, dropout: float) -> None:
        super().__init__()
        positions = torch.arange(max_sequence_length, dtype=torch.float).unsqueeze(1)
        divisors = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_sequence_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * divisors)
        encoding[:, 1::2] = torch.cos(positions * divisors)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(dropout)
        self.max_sequence_length = max_sequence_length

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError("PositionalEncoding expects (batch, sequence, d_model) tensors.")
        length = embeddings.size(1)
        if length > self.max_sequence_length:
            raise ValueError(f"Sequence length {length} exceeds configured model maximum of {self.max_sequence_length}.")
        return self.dropout(embeddings + self.encoding[:, :length].to(dtype=embeddings.dtype))


class TransformerTranslator(nn.Module):
    """Shared-vocabulary Transformer that returns raw vocabulary logits only."""

    def __init__(self, vocab_size: int, pad_id: int, d_model: int, attention_heads: int, encoder_layers: int, decoder_layers: int, feedforward_dimension: int, dropout: float, max_sequence_length: int) -> None:
        super().__init__()
        if vocab_size <= 0 or not 0 <= pad_id < vocab_size:
            raise ModelConfigurationError("Tokenizer vocabulary size and PAD ID are incompatible.")
        self.vocab_size, self.pad_id, self.d_model = vocab_size, pad_id, d_model
        self.max_sequence_length = max_sequence_length
        # One embedding is shared because English and Tagalog share the SentencePiece vocabulary.
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.positional_encoding = PositionalEncoding(d_model, max_sequence_length, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=attention_heads, dim_feedforward=feedforward_dimension, dropout=dropout, batch_first=True)
        encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers, enable_nested_tensor=False)
        self.transformer = nn.Transformer(d_model=d_model, nhead=attention_heads, num_encoder_layers=encoder_layers, num_decoder_layers=decoder_layers, dim_feedforward=feedforward_dimension, dropout=dropout, batch_first=True, custom_encoder=encoder)
        self.output_projection = nn.Linear(d_model, vocab_size)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Use standard Xavier initialization while retaining a zero padding vector."""
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        with torch.no_grad():
            self.token_embedding.weight[self.pad_id].zero_()

    def key_padding_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return True at padding positions, as required by PyTorch attention."""
        return tokens.eq(self.pad_id)

    @staticmethod
    def causal_mask(length: int, device: torch.device) -> torch.Tensor:
        """Return a boolean mask where True blocks decoder access to future tokens."""
        return torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)

    def _validate_tokens(self, tokens: torch.Tensor, name: str) -> None:
        if tokens.ndim != 2:
            raise ValueError(f"{name} tokens must have shape (batch, sequence).")
        if tokens.dtype != torch.long:
            raise TypeError(f"{name} tokens must use torch.long dtype.")
        if tokens.size(1) == 0:
            raise ValueError(f"{name} sequence length must be greater than zero.")
        if tokens.size(1) > self.max_sequence_length:
            raise ValueError(f"Sequence length {tokens.size(1)} exceeds configured model maximum of {self.max_sequence_length}.")
        if tokens.numel() and (tokens.min().item() < 0 or tokens.max().item() >= self.vocab_size):
            raise ValueError(f"{name} tokens contain IDs outside the tokenizer vocabulary.")

    def forward(self, src_tokens: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        """Return raw logits shaped (batch, target_length, vocabulary_size); no softmax."""
        self._validate_tokens(src_tokens, "Source")
        self._validate_tokens(tgt_tokens, "Target")
        src_padding = self.key_padding_mask(src_tokens)
        tgt_padding = self.key_padding_mask(tgt_tokens)
        src = self.positional_encoding(self.token_embedding(src_tokens) * math.sqrt(self.d_model))
        tgt = self.positional_encoding(self.token_embedding(tgt_tokens) * math.sqrt(self.d_model))
        hidden = self.transformer(src, tgt, tgt_mask=self.causal_mask(tgt_tokens.size(1), tgt_tokens.device), src_key_padding_mask=src_padding, tgt_key_padding_mask=tgt_padding, memory_key_padding_mask=src_padding)
        return self.output_projection(hidden)


def build_model(config: dict[str, Any], tokenizer: SentencePieceTokenizer) -> TransformerTranslator:
    """Construct the architecture from config and the actual trained tokenizer."""
    settings = validate_model_config(config)
    special = tokenizer.get_special_token_ids()
    if any(token_id < 0 for token_id in special.values()):
        raise ModelConfigurationError("Trained tokenizer does not provide all special token IDs.")
    return TransformerTranslator(vocab_size=tokenizer.get_vocab_size(), pad_id=special["pad"], d_model=settings["d_model"], attention_heads=settings["attention_heads"], encoder_layers=settings["encoder_layers"], decoder_layers=settings["decoder_layers"], feedforward_dimension=settings["feedforward_dimension"], dropout=settings["dropout"], max_sequence_length=settings["max_sequence_length"])


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return parameter counts and float32 parameter memory only."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable, "parameter_memory_bytes": total * 4}


def architecture_summary(model: TransformerTranslator, config: dict[str, Any]) -> dict[str, Any]:
    """Build dynamic inspection metadata without saving untrained weights."""
    settings = validate_model_config(config)
    return {"architecture": "Transformer Encoder-Decoder", "vocab_size": model.vocab_size, "shared_embedding": True, **settings, **count_parameters(model)}
