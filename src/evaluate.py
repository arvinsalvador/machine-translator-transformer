"""Bounded test-only greedy evaluation with BLEU and chrF for Phase 6."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sacrebleu
import torch

from src.model import build_model
from src.tokenizer_utils import SentencePieceTokenizer, load_tokenizer
from src.train import PROJECT_ROOT, TrainingError, check_preconditions, paths
from src.train_tokenizer import load_config


def greedy_decode(model, tokenizer: SentencePieceTokenizer, source: str, device: torch.device, max_length: int) -> str:
    """Generate one bounded greedy sequence; this supports evaluation, not a public translator."""
    source_ids = tokenizer.encode(source, add_bos=True, add_eos=True, max_length=max_length)
    generated = [tokenizer.bos_id]
    model.eval()
    with torch.no_grad():
        for _ in range(max_length - 1):
            logits = model(torch.tensor([source_ids], dtype=torch.long, device=device), torch.tensor([generated], dtype=torch.long, device=device))
            next_id = int(logits[0, -1].argmax().item()); generated.append(next_id)
            if next_id == tokenizer.eos_id: break
    return tokenizer.decode(generated)


def evaluate_model(config: dict[str, Any], limit: int | None = None) -> dict[str, Any]:
    """Load best checkpoint and evaluate only the configured prepared test split."""
    result = check_preconditions(config); evaluation = config["evaluation"]
    maximum = evaluation["maximum_samples"]; limit = evaluation["default_max_samples"] if limit is None else limit
    if not isinstance(limit, int) or limit <= 0 or limit > maximum: raise TrainingError(f"Evaluation limit must be between 1 and {maximum}.")
    if not result["best_file"].is_file(): raise TrainingError("Best checkpoint is unavailable. Train the Transformer before evaluation.")
    test_path = PROJECT_ROOT / config["preprocessing"]["output_directory"] / config["preprocessing"]["test_file"]
    if not test_path.is_file(): raise TrainingError("Prepared test split is unavailable.")
    tokenizer = load_tokenizer(result["tokenizer"]); device = torch.device("cuda" if config["resources"].get("prefer_gpu") and torch.cuda.is_available() else "cpu")
    model = build_model(config, tokenizer).to(device); checkpoint = torch.load(result["best_file"], map_location=device, weights_only=False)
    if checkpoint.get("vocab_size") != model.vocab_size or checkpoint.get("pad_id") != tokenizer.pad_id: raise TrainingError("Best checkpoint is incompatible with current tokenizer.")
    model.load_state_dict(checkpoint["model_state_dict"])
    predictions: list[str] = []; references: list[str] = []; examples: list[dict[str, str]] = []
    with test_path.open("r", encoding="utf-8") as file:
        for line in file:
            if len(predictions) >= limit: break
            try: record = json.loads(line)
            except json.JSONDecodeError: continue
            source, reference = record.get("english"), record.get("tagalog")
            if not isinstance(source, str) or not isinstance(reference, str): continue
            prediction = greedy_decode(model, tokenizer, source, device, config["model"]["max_sequence_length"])
            predictions.append(prediction); references.append(reference)
            if len(examples) < evaluation.get("example_limit", 10): examples.append({"english": source, "reference_tagalog": reference, "predicted_tagalog": prediction})
    if not predictions: raise TrainingError("No usable test examples were available for evaluation.")
    output = {"checkpoint_used": str(result["best_file"].relative_to(PROJECT_ROOT)), "test_samples_evaluated": len(predictions), "bleu": sacrebleu.corpus_bleu(predictions, [references]).score, "chrf": sacrebleu.corpus_chrf(predictions, [references]).score, "max_sequence_length": config["model"]["max_sequence_length"], "device": str(device), "evaluated_at": datetime.now(timezone.utc).isoformat(), "examples": examples}
    result_path = PROJECT_ROOT / evaluation["results_file"]; result_path.parent.mkdir(parents=True, exist_ok=True); result_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--limit", type=int); args = parser.parse_args()
    try:
        result = evaluate_model(load_config(), args.limit); print(f"BLEU: {result['bleu']:.2f}\nchrF: {result['chrf']:.2f}\nSamples: {result['test_samples_evaluated']}"); return 0
    except TrainingError as error: print(f"PHASE 6 EVALUATION: FAILED\nError: {error}"); return 1

if __name__ == "__main__": raise SystemExit(main())
