"""Safe local English-to-Tagalog inference using the Phase 6 best checkpoint."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from src.model import build_model
from src.tokenizer_utils import SentencePieceTokenizer, load_tokenizer
from src.train import PROJECT_ROOT, TrainingError, check_preconditions
from src.train_tokenizer import load_config


class InferenceError(Exception): pass


def load_inference_components(config: dict[str, Any]) -> tuple[torch.nn.Module, SentencePieceTokenizer, torch.device]:
    """Load only a strict compatible trained best checkpoint; never a random fallback."""
    try: paths = check_preconditions(config)
    except TrainingError as error: raise InferenceError(str(error)) from error
    if not paths["best_file"].is_file(): raise InferenceError("The trained model is unavailable. Complete Phase 6 training first.")
    tokenizer = load_tokenizer(paths["tokenizer"])
    device = torch.device("cuda" if config["resources"].get("prefer_gpu") and torch.cuda.is_available() else "cpu")
    model = build_model(config, tokenizer).to(device)
    try: checkpoint = torch.load(paths["best_file"], map_location=device, weights_only=False)
    except Exception as error: raise InferenceError("The best checkpoint could not be loaded.") from error
    expected = {"vocab_size": model.vocab_size, "pad_id": tokenizer.pad_id, "model_config": config["model"]}
    if any(checkpoint.get(key) != value for key, value in expected.items()): raise InferenceError("Checkpoint configuration does not match the current model configuration.")
    try: model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (KeyError, RuntimeError) as error: raise InferenceError("Checkpoint weights are incompatible with the current model.") from error
    model.eval(); return model, tokenizer, device


def prepare_source(text: str, tokenizer: SentencePieceTokenizer, max_length: int, max_characters: int = 500) -> list[int]:
    """Validate one short caption-like input and explicitly account for BOS/EOS."""
    if not isinstance(text, str) or not text.strip(): raise InferenceError("Enter a non-empty English sentence.")
    if len(text) > max_characters: raise InferenceError(f"Input exceeds the {max_characters}-character limit.")
    try: return tokenizer.encode(text.strip(), add_bos=True, add_eos=True, max_length=max_length)
    except ValueError as error: raise InferenceError("Input is too long for the current model. Please shorten the sentence.") from error


def greedy_decode(model, tokenizer: SentencePieceTokenizer, source_ids: list[int], device: torch.device, max_length: int) -> tuple[list[int], str]:
    """Bounded argmax decoding with no softmax and no gradient allocation."""
    generated = [tokenizer.bos_id]; source = torch.tensor([source_ids], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(max_length - 1):
            target = torch.tensor([generated], dtype=torch.long, device=device)
            next_id = int(model(source, target)[0, -1].argmax().item()); generated.append(next_id)
            if next_id == tokenizer.eos_id: return generated, "eos"
    return generated, "max_length"


def translate_text(text: str, config: dict[str, Any] | None = None, components: tuple[torch.nn.Module, SentencePieceTokenizer, torch.device] | None = None) -> dict[str, Any]:
    """Translate one bounded request locally through the trained custom Transformer."""
    config = load_config() if config is None else config
    model, tokenizer, device = components or load_inference_components(config)
    source_ids = prepare_source(text, tokenizer, config["model"]["max_sequence_length"], config["preprocessing"]["max_characters"])
    started = time.perf_counter(); output_ids, stopped_by = greedy_decode(model, tokenizer, source_ids, device, config["model"]["max_sequence_length"])
    return {"source_text": text.strip(), "translation": tokenizer.decode(output_ids), "source_token_count": len(source_ids), "generated_token_count": len(output_ids), "device": str(device), "duration_ms": (time.perf_counter() - started) * 1000, "stopped_by": stopped_by}


def main() -> int:
    parser = argparse.ArgumentParser(description="Translate one short English sentence with the trained local Transformer."); parser.add_argument("--text", required=True); args = parser.parse_args()
    try:
        result = translate_text(args.text)
        print(f"English: {result['source_text']}\nTagalog: {result['translation']}\nSource Tokens: {result['source_token_count']}\nGenerated Tokens: {result['generated_token_count']}\nDevice: {result['device']}\nInference Time: {result['duration_ms'] / 1000:.2f} seconds")
        return 0
    except InferenceError as error: print(f"TRANSLATION FAILED: {error}"); return 1


if __name__ == "__main__": raise SystemExit(main())
