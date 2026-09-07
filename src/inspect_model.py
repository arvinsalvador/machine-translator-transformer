"""CLI inspection and synthetic forward-pass validation for the untrained model."""

from __future__ import annotations

import argparse

import torch

from src.model import architecture_summary, build_model
from src.tokenizer_utils import load_tokenizer
from src.train_tokenizer import get_paths, load_config, validate_config


def synthetic_validation(model, tokenizer) -> tuple[int, int, int]:
    """Run one tiny no-grad forward pass with valid synthetic IDs only."""
    special = tokenizer.get_special_token_ids()
    value = 4 if model.vocab_size > 4 else special["unk"]
    src = torch.tensor([[special["bos"], value, special["eos"], special["pad"], special["pad"]], [special["bos"], value, value, special["eos"], special["pad"]]], dtype=torch.long)
    tgt = torch.tensor([[special["bos"], value, special["eos"], special["pad"]], [special["bos"], value, value, special["eos"]]], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        logits = model(src, tgt)
    expected = (2, tgt.size(1), model.vocab_size)
    if tuple(logits.shape) != expected:
        raise RuntimeError(f"Unexpected synthetic output shape {tuple(logits.shape)}; expected {expected}.")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the untrained Phase 5 Transformer.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    config = load_config()
    dataset, preprocessing, tokenizer_settings = validate_config(config)
    _, _, prefix, _ = get_paths(dataset, preprocessing, tokenizer_settings)
    tokenizer = load_tokenizer(prefix.with_suffix(".model"))
    model = build_model(config, tokenizer)
    summary = architecture_summary(model, config)
    shape = synthetic_validation(model, tokenizer)
    print("=" * 60)
    print("PHASE 5 – TRANSFORMER MODEL")
    print("=" * 60)
    for key in ("vocab_size", "d_model", "attention_heads", "encoder_layers", "decoder_layers", "feedforward_dimension", "dropout", "max_sequence_length", "total", "trainable"):
        print(f"{key.replace('_', ' ').title():24}: {summary[key]:,}" if isinstance(summary[key], int) else f"{key.replace('_', ' ').title():24}: {summary[key]}")
    print(f"Synthetic output         : {shape}")
    print("Status                   : MODEL ARCHITECTURE VALID")
    print("No training performed.")
    if args.verbose:
        print(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
