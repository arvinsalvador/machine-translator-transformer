"""Train a shared SentencePiece BPE tokenizer from Phase 3 training data only."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import sentencepiece as spm
import yaml

from src.tokenizer_utils import SentencePieceTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class TokenizerTrainingError(Exception):
    """A concise user-facing tokenizer training error."""


@dataclass
class TokenizerTrainingStats:
    """Small tokenizer training report persisted in metadata."""

    training_records: int = 0
    training_text_lines: int = 0
    requested_vocab_size: int = 0
    actual_vocab_size: int = 0
    completed: bool = False


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except FileNotFoundError as error:
        raise TokenizerTrainingError(f"Configuration file was not found: {path}") from error
    except (OSError, yaml.YAMLError) as error:
        raise TokenizerTrainingError(f"Could not read configuration: {error}") from error
    if not isinstance(config, dict):
        raise TokenizerTrainingError("Configuration must be a top-level YAML mapping.")
    return config


def validate_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate only the Phase 3/4 settings needed for local tokenizer training."""
    dataset, preprocessing, tokenizer = config.get("dataset"), config.get("preprocessing"), config.get("tokenizer")
    if not all(isinstance(section, dict) for section in (dataset, preprocessing, tokenizer)):
        raise TokenizerTrainingError("Missing dataset, preprocessing, or tokenizer configuration.")
    if not isinstance(dataset.get("acquisition"), dict) or not isinstance(preprocessing.get("output_directory"), str):
        raise TokenizerTrainingError("Phase 2/3 output configuration is incomplete.")
    for key in ("train_file", "metadata_file"):
        if not isinstance(preprocessing.get(key), str) or not preprocessing[key]:
            raise TokenizerTrainingError(f"preprocessing.{key} must be a non-empty string.")
    for key in ("model_type", "model_prefix", "metadata_file"):
        if not isinstance(tokenizer.get(key), str) or not tokenizer[key]:
            raise TokenizerTrainingError(f"tokenizer.{key} must be a non-empty string.")
    if tokenizer["model_type"] != "bpe":
        raise TokenizerTrainingError("Phase 4 requires tokenizer.model_type to be 'bpe'.")
    if isinstance(tokenizer.get("vocab_size"), bool) or not isinstance(tokenizer.get("vocab_size"), int) or tokenizer["vocab_size"] <= 0:
        raise TokenizerTrainingError("tokenizer.vocab_size must be a positive integer.")
    special_tokens = tokenizer.get("special_tokens")
    if not isinstance(special_tokens, dict) or any(not isinstance(special_tokens.get(key), str) for key in ("pad", "unknown", "bos", "eos")):
        raise TokenizerTrainingError("Tokenizer special token configuration is incomplete.")
    return dataset, preprocessing, tokenizer


def get_paths(dataset: dict[str, Any], preprocessing: dict[str, Any], tokenizer: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    """Resolve the Phase 3 train split and final tokenizer artifact paths."""
    train_path = PROJECT_ROOT / preprocessing["output_directory"] / preprocessing["train_file"]
    preparation_metadata = PROJECT_ROOT / preprocessing["output_directory"] / preprocessing["metadata_file"]
    prefix = PROJECT_ROOT / tokenizer["model_prefix"]
    metadata = prefix.parent / tokenizer["metadata_file"]
    return train_path, preparation_metadata, prefix, metadata


def verify_preconditions(train_path: Path, preparation_metadata: Path) -> None:
    """Require a non-empty completed Phase 3 training split; never use other splits."""
    if not train_path.is_file() or train_path.stat().st_size == 0:
        raise TokenizerTrainingError(f"Phase 3 training split was not found: {train_path}. Complete Phase 3 first.")
    if preparation_metadata.is_file():
        try:
            metadata = json.loads(preparation_metadata.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TokenizerTrainingError(f"Could not read Phase 3 metadata: {error}") from error
        if not isinstance(metadata, dict) or metadata.get("completed") is not True:
            raise TokenizerTrainingError("Phase 3 data preparation is incomplete. Complete Phase 3 before training the tokenizer.")


def build_training_corpus(train_path: Path, corpus_path: Path, max_records: int | None = None, progress_callback: Callable[[str], None] | None = None) -> tuple[int, int]:
    """Write English and Tagalog lines from train.jsonl only; no validation/test access."""
    records = text_lines = 0
    with train_path.open("r", encoding="utf-8") as train_file, corpus_path.open("w", encoding="utf-8") as corpus_file:
        for line in train_file:
            if max_records is not None and records >= max_records:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            english, tagalog = record.get("english"), record.get("tagalog")
            if not isinstance(english, str) or not isinstance(tagalog, str) or not english.strip() or not tagalog.strip():
                continue
            corpus_file.write(english.strip() + "\n")
            corpus_file.write(tagalog.strip() + "\n")
            records += 1
            text_lines += 2
    if progress_callback:
        progress_callback("Training corpus prepared.")
    if not text_lines:
        raise TokenizerTrainingError("The Phase 3 training split contains no usable English–Tagalog pairs.")
    return records, text_lines


def validate_special_tokens(tokenizer: SentencePieceTokenizer, configured: dict[str, str]) -> dict[str, int]:
    """Verify explicit PAD/UNK/BOS/EOS pieces and the required stable IDs."""
    expected = {"pad": 0, "unk": 1, "bos": 2, "eos": 3}
    actual = tokenizer.get_special_token_ids()
    for key, expected_id in expected.items():
        if actual[key] != expected_id or tokenizer.piece_to_id(configured["unknown" if key == "unk" else key]) != expected_id:
            raise TokenizerTrainingError(f"Special token validation failed for {key.upper()}; expected ID {expected_id}.")
    return actual


def train_tokenizer(config: dict[str, Any], force: bool = False, max_records: int | None = None, progress_callback: Callable[[str], None] | None = None) -> TokenizerTrainingStats:
    """Train a shared BPE model atomically from only the prepared train split."""
    dataset, preprocessing, settings = validate_config(config)
    train_path, preparation_metadata, prefix, metadata_path = get_paths(dataset, preprocessing, settings)
    verify_preconditions(train_path, preparation_metadata)
    model_path, vocab_path = prefix.with_suffix(".model"), prefix.with_suffix(".vocab")
    if any(path.exists() for path in (model_path, vocab_path, metadata_path)) and not force:
        raise TokenizerTrainingError("Tokenizer already exists. Use --force to replace it.")
    if max_records is not None and max_records <= 0:
        raise TokenizerTrainingError("--max-records must be a positive integer.")

    prefix.parent.mkdir(parents=True, exist_ok=True)
    corpus_path = prefix.parent / "tokenizer_training_corpus.txt.tmp"
    temporary_prefix = prefix.parent / (".tmp_" + prefix.name)
    temporary_model, temporary_vocab = temporary_prefix.with_suffix(".model"), temporary_prefix.with_suffix(".vocab")
    temporary_metadata = metadata_path.with_name(metadata_path.name + ".tmp")
    for path in (corpus_path, temporary_model, temporary_vocab, temporary_metadata):
        path.unlink(missing_ok=True)
    stats = TokenizerTrainingStats(requested_vocab_size=settings["vocab_size"])
    try:
        if progress_callback:
            progress_callback("Preparing corpus from training split...")
        stats.training_records, stats.training_text_lines = build_training_corpus(train_path, corpus_path, max_records, progress_callback)
        if progress_callback:
            progress_callback("Training SentencePiece BPE...")
        special = settings["special_tokens"]
        spm.SentencePieceTrainer.train(input=str(corpus_path), model_prefix=str(temporary_prefix), model_type="bpe", vocab_size=settings["vocab_size"], character_coverage=settings.get("character_coverage", 1.0), hard_vocab_limit=False, pad_id=0, unk_id=1, bos_id=2, eos_id=3, pad_piece=special["pad"], unk_piece=special["unknown"], bos_piece=special["bos"], eos_piece=special["eos"], num_threads=config.get("resources", {}).get("max_cpu_threads", 2), minloglevel=2)
        if progress_callback:
            progress_callback("Validating tokenizer...")
        trained = SentencePieceTokenizer(temporary_model)
        special_ids = validate_special_tokens(trained, special)
        stats.actual_vocab_size = trained.get_vocab_size()
        stats.completed = True
        write_metadata(temporary_metadata, settings, stats, special_ids, model_path, vocab_path)
        temporary_model.replace(model_path)
        temporary_vocab.replace(vocab_path)
        temporary_metadata.replace(metadata_path)
        if progress_callback:
            progress_callback("Complete.")
        return stats
    except KeyboardInterrupt:
        raise TokenizerTrainingError("Tokenizer training cancelled. Existing tokenizer files were left untouched.")
    except Exception as error:
        if isinstance(error, TokenizerTrainingError):
            raise
        raise TokenizerTrainingError(f"SentencePiece training failed: {error}") from error
    finally:
        corpus_path.unlink(missing_ok=True)
        for path in (temporary_model, temporary_vocab, temporary_metadata):
            path.unlink(missing_ok=True)


def write_metadata(path: Path, settings: dict[str, Any], stats: TokenizerTrainingStats, special_ids: dict[str, int], model_path: Path, vocab_path: Path) -> None:
    """Persist small, inspectable metadata after model validation succeeds."""
    def display_path(value: Path) -> str:
        try:
            return str(value.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(value)
    metadata = {"model_type": settings["model_type"], **asdict(stats), "character_coverage": settings.get("character_coverage", 1.0), "training_split": "train", **{f"{name}_id": value for name, value in special_ids.items()}, "model_file": display_path(model_path), "vocab_file": display_path(vocab_path), "trained_at": datetime.now(timezone.utc).isoformat()}
    with path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
        metadata_file.write("\n")


def print_summary(config: dict[str, Any], stats: TokenizerTrainingStats) -> None:
    """Print a concise model-free Phase 4 completion summary."""
    _, _, settings = validate_config(config)
    special = settings["special_tokens"]
    print("=" * 60)
    print("PHASE 4 – TOKENIZER TRAINING")
    print("=" * 60)
    print("Tokenizer             : SentencePiece BPE")
    print("Training split        : train only")
    print(f"Training pairs        : {stats.training_records:,}")
    print(f"Training text lines   : {stats.training_text_lines:,}")
    print(f"Requested vocabulary  : {stats.requested_vocab_size:,}")
    print(f"Actual vocabulary     : {stats.actual_vocab_size:,}")
    for name, token, token_id in (("PAD", special["pad"], 0), ("UNK", special["unknown"], 1), ("BOS", special["bos"], 2), ("EOS", special["eos"], 3)):
        print(f"{name:3} {token:14}: {token_id}")
    print("Status                : SUCCESS")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a shared SentencePiece BPE tokenizer from train.jsonl.")
    parser.add_argument("--force", action="store_true", help="Replace an existing tokenizer after successful training.")
    parser.add_argument("--max-records", type=int, help="Development-only cap on training-split records.")
    arguments = parser.parse_args()
    try:
        config = load_config()
        stats = train_tokenizer(config, arguments.force, arguments.max_records)
        print_summary(config, stats)
        return 0
    except TokenizerTrainingError as error:
        print(f"PHASE 4 TOKENIZER TRAINING: FAILED\nError: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
