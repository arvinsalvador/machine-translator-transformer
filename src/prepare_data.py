"""Conservatively prepare Phase 2 JSONL pairs for later tokenizer training."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class PreparationError(Exception):
    """A clear user-facing error for data-preparation preconditions."""


@dataclass
class PreparationStats:
    """Counters persisted in preparation metadata and shown in the UI."""

    raw_records: int = 0
    malformed_json: int = 0
    missing_english: int = 0
    missing_tagalog: int = 0
    empty_english: int = 0
    empty_tagalog: int = 0
    too_short: int = 0
    too_long: int = 0
    duplicates: int = 0
    accepted_records: int = 0
    train_records: int = 0
    validation_records: int = 0
    test_records: int = 0
    completed: bool = False


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load configuration without importing any later-phase ML components."""
    try:
        with path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except FileNotFoundError as error:
        raise PreparationError(f"Configuration file was not found: {path}") from error
    except (OSError, yaml.YAMLError) as error:
        raise PreparationError(f"Could not read configuration: {error}") from error
    if not isinstance(config, dict):
        raise PreparationError("Configuration must be a top-level YAML mapping.")
    return config


def validate_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate only settings needed by preprocessing and splitting."""
    dataset, preprocessing = config.get("dataset"), config.get("preprocessing")
    project = config.get("project")
    if not all(isinstance(value, dict) for value in (dataset, preprocessing, project)):
        raise PreparationError("Missing dataset, preprocessing, or project configuration.")
    acquisition = dataset.get("acquisition")
    if not isinstance(acquisition, dict) or not isinstance(acquisition.get("output_file"), str):
        raise PreparationError("Missing dataset.acquisition.output_file configuration.")
    for key in ("train_ratio", "validation_ratio", "test_ratio"):
        if not isinstance(dataset.get(key), (int, float)):
            raise PreparationError(f"dataset.{key} must be numeric.")
    if abs(dataset["train_ratio"] + dataset["validation_ratio"] + dataset["test_ratio"] - 1.0) > 1e-6:
        raise PreparationError("Dataset split ratios must total 1.0.")
    for key in ("min_words", "max_words", "max_characters"):
        if isinstance(preprocessing.get(key), bool) or not isinstance(preprocessing.get(key), int) or preprocessing[key] <= 0:
            raise PreparationError(f"preprocessing.{key} must be a positive integer.")
    if preprocessing["min_words"] > preprocessing["max_words"]:
        raise PreparationError("preprocessing.min_words cannot exceed preprocessing.max_words.")
    for key in ("output_directory", "train_file", "validation_file", "test_file", "metadata_file"):
        if not isinstance(preprocessing.get(key), str) or not preprocessing[key]:
            raise PreparationError(f"preprocessing.{key} must be a non-empty string.")
    if isinstance(project.get("random_seed"), bool) or not isinstance(project.get("random_seed"), int):
        raise PreparationError("project.random_seed must be an integer.")
    return dataset, preprocessing


def normalize_text(text: str, settings: dict[str, Any]) -> str:
    """Apply only conservative Unicode and whitespace normalization."""
    if settings.get("normalize_unicode", True):
        text = unicodedata.normalize("NFC", text)
    if settings.get("remove_control_characters", True):
        text = "".join(" " if unicodedata.category(char) == "Cc" else char for char in text)
    if settings.get("collapse_whitespace", True):
        text = re.sub(r"\s+", " ", text)
    return text.strip()


def duplicate_key(english: str, tagalog: str) -> tuple[str, str]:
    """Compare normalized pairs case-insensitively without changing stored text."""
    return english.casefold(), tagalog.casefold()


def validate_record(record: Any, settings: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Return a clean minimal record or a single reason to count and skip it."""
    if not isinstance(record, dict):
        return None, "missing_english"
    if "english" not in record or not isinstance(record["english"], str):
        return None, "missing_english"
    if "tagalog" not in record or not isinstance(record["tagalog"], str):
        return None, "missing_tagalog"
    english = normalize_text(record["english"], settings)
    tagalog = normalize_text(record["tagalog"], settings)
    if not english:
        return None, "empty_english"
    if not tagalog:
        return None, "empty_tagalog"
    word_counts = (len(english.split()), len(tagalog.split()))  # Words are not tokenizer tokens.
    if min(word_counts) < settings["min_words"]:
        return None, "too_short"
    if max(word_counts) > settings["max_words"] or max(len(english), len(tagalog)) > settings["max_characters"]:
        return None, "too_long"
    clean = {"source_id": str(record.get("source_id", "")), "english": english, "tagalog": tagalog}
    if isinstance(record.get("score"), (int, float)) and not isinstance(record.get("score"), bool):
        clean["score"] = record["score"]
    return clean, None


def get_paths(dataset: dict[str, Any], settings: dict[str, Any]) -> tuple[Path, Path, dict[str, Path]]:
    """Resolve configured input, metadata, and processed output paths."""
    raw_path = PROJECT_ROOT / dataset["acquisition"]["output_file"]
    output_directory = PROJECT_ROOT / settings["output_directory"]
    metadata = output_directory / settings["metadata_file"]
    splits = {"train": output_directory / settings["train_file"], "validation": output_directory / settings["validation_file"], "test": output_directory / settings["test_file"]}
    return raw_path, metadata, splits


def verify_preconditions(raw_path: Path, acquisition_metadata: Path) -> None:
    """Refuse missing or explicitly incomplete Phase 2 acquisitions."""
    if not raw_path.is_file() or raw_path.stat().st_size == 0:
        raise PreparationError(f"Phase 2 raw dataset was not found: {raw_path}. Complete acquisition first.")
    if acquisition_metadata.is_file():
        try:
            with acquisition_metadata.open("r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
        except (OSError, json.JSONDecodeError) as error:
            raise PreparationError(f"Could not read Phase 2 acquisition metadata: {error}") from error
        if not isinstance(metadata, dict) or metadata.get("completed") is not True:
            raise PreparationError("Phase 2 acquisition is incomplete. Complete dataset acquisition before running Phase 3.")


def calculate_split_counts(total: int, dataset: dict[str, Any]) -> tuple[int, int, int]:
    """Use floor validation/test counts and give the exact remainder to train."""
    validation = int(total * dataset["validation_ratio"])
    test = int(total * dataset["test_ratio"])
    return total - validation - test, validation, test


def generate_split_indexes(total: int, seed: int, counts: tuple[int, int, int]) -> dict[str, set[int]]:
    """Generate deterministic, disjoint exact split memberships by clean-file index."""
    indexes = list(range(total))
    random.Random(seed).shuffle(indexes)
    train_count, validation_count, _ = counts
    return {"train": set(indexes[:train_count]), "validation": set(indexes[train_count:train_count + validation_count]), "test": set(indexes[train_count + validation_count:])}


def prepare_dataset(config: dict[str, Any], force: bool = False, progress_callback: Callable[[PreparationStats], None] | None = None) -> PreparationStats:
    """Clean raw JSONL in two passes and atomically publish exact data splits."""
    dataset, settings = validate_config(config)
    raw_path, metadata_path, outputs = get_paths(dataset, settings)
    acquisition_metadata = raw_path.parent / dataset["acquisition"]["metadata_file"].split("/")[-1]
    verify_preconditions(raw_path, acquisition_metadata)
    if any(path.exists() for path in (*outputs.values(), metadata_path)) and not force:
        raise PreparationError("Processed dataset already exists. Use --force to replace it.")

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path = metadata_path.parent / ".clean_pairs.tmp"
    temporary_outputs = {name: path.with_name(path.name + ".tmp") for name, path in outputs.items()}
    for path in (clean_path, *temporary_outputs.values(), metadata_path.with_name(metadata_path.name + ".tmp")):
        path.unlink(missing_ok=True)
    stats = PreparationStats()
    seen: set[tuple[str, str]] = set()
    progress = tqdm(desc="Processing records", unit="record", disable=progress_callback is not None)

    try:
        with raw_path.open("r", encoding="utf-8") as raw_file, clean_path.open("w", encoding="utf-8") as clean_file:
            for line in raw_file:
                stats.raw_records += 1
                progress.update(1)
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    stats.malformed_json += 1
                    continue
                clean, reason = validate_record(record, settings)
                if clean is None:
                    setattr(stats, reason or "missing_english", getattr(stats, reason or "missing_english") + 1)
                    continue
                key = duplicate_key(clean["english"], clean["tagalog"])
                if settings.get("deduplicate", True) and key in seen:
                    stats.duplicates += 1
                    continue
                seen.add(key)
                json.dump(clean, clean_file, ensure_ascii=False)
                clean_file.write("\n")
                stats.accepted_records += 1
                if progress_callback:
                    progress_callback(stats)

        counts = calculate_split_counts(stats.accepted_records, dataset)
        memberships = generate_split_indexes(stats.accepted_records, config["project"]["random_seed"], counts)
        with clean_path.open("r", encoding="utf-8") as clean_file, temporary_outputs["train"].open("w", encoding="utf-8") as train_file, temporary_outputs["validation"].open("w", encoding="utf-8") as validation_file, temporary_outputs["test"].open("w", encoding="utf-8") as test_file:
            writers = {"train": train_file, "validation": validation_file, "test": test_file}
            for index, line in enumerate(clean_file):
                split = next(name for name, indexes in memberships.items() if index in indexes)
                writers[split].write(line)
        stats.train_records, stats.validation_records, stats.test_records = counts
        stats.completed = True
        write_metadata(metadata_path.with_name(metadata_path.name + ".tmp"), raw_path, dataset, settings, config["project"]["random_seed"], stats)
        for name, path in outputs.items():
            temporary_outputs[name].replace(path)
        metadata_path.with_name(metadata_path.name + ".tmp").replace(metadata_path)
        return stats
    except KeyboardInterrupt:
        raise PreparationError("Data preparation cancelled. Existing processed files were left untouched.")
    finally:
        progress.close()
        clean_path.unlink(missing_ok=True)
        for path in temporary_outputs.values():
            path.unlink(missing_ok=True)
        metadata_path.with_name(metadata_path.name + ".tmp").unlink(missing_ok=True)


def write_metadata(path: Path, raw_path: Path, dataset: dict[str, Any], settings: dict[str, Any], seed: int, stats: PreparationStats) -> None:
    """Write small UTF-8 preparation metadata to a temporary output path."""
    try:
        input_file = str(raw_path.relative_to(PROJECT_ROOT))
    except ValueError:
        input_file = str(raw_path)
    metadata = {"input_file": input_file, **asdict(stats), "train_ratio": dataset["train_ratio"], "validation_ratio": dataset["validation_ratio"], "test_ratio": dataset["test_ratio"], "random_seed": seed, "preprocessing": {key: settings[key] for key in ("normalize_unicode", "collapse_whitespace", "remove_control_characters", "deduplicate", "min_words", "max_words", "max_characters")}, "processed_at": datetime.now(timezone.utc).isoformat()}
    with path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
        metadata_file.write("\n")


def print_summary(config: dict[str, Any], stats: PreparationStats) -> None:
    """Print a compact terminal report without per-record noise."""
    _, settings = validate_config(config)
    print("=" * 60)
    print("PHASE 3 – DATA PREPARATION")
    print("=" * 60)
    for label, value in (("Raw records", stats.raw_records), ("Malformed JSON", stats.malformed_json), ("Missing English", stats.missing_english), ("Missing Tagalog", stats.missing_tagalog), ("Empty English", stats.empty_english), ("Empty Tagalog", stats.empty_tagalog), ("Too short", stats.too_short), ("Too long", stats.too_long), ("Duplicates", stats.duplicates), ("Clean records", stats.accepted_records), ("TRAIN", stats.train_records), ("VALIDATION", stats.validation_records), ("TEST", stats.test_records)):
        print(f"{label:20}: {value:,}")
    print(f"Output              : {settings['output_directory']}/")
    print("Status              : SUCCESS")
    print("=" * 60)


def main() -> int:
    """Run the CLI entry point for Phase 3."""
    parser = argparse.ArgumentParser(description="Prepare raw English–Tagalog JSONL pairs.")
    parser.add_argument("--force", action="store_true", help="Replace existing processed outputs after success.")
    arguments = parser.parse_args()
    try:
        config = load_config()
        stats = prepare_dataset(config, force=arguments.force)
        print_summary(config, stats)
        return 0
    except PreparationError as error:
        print(f"PHASE 3 DATA PREPARATION: FAILED\nError: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
