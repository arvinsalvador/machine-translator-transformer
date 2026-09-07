"""Stream a resource-limited English–Tagalog raw dataset acquisition."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class AcquisitionError(Exception):
    """A user-facing acquisition error that should return a non-zero status."""


@dataclass
class AcquisitionStats:
    """Small running counters suitable for terminal and UI progress reporting."""

    requested_pairs: int
    scan_limit: int
    records_scanned: int = 0
    pairs_saved: int = 0
    missing_english: int = 0
    missing_tagalog: int = 0
    malformed_captions: int = 0
    mode: str = "quick"
    completed: bool = False
    stop_reason: str = "running"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the project configuration safely."""
    try:
        with path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except FileNotFoundError as error:
        raise AcquisitionError(f"Configuration file was not found: {path}") from error
    except yaml.YAMLError as error:
        raise AcquisitionError(f"Configuration YAML is malformed: {error}") from error
    except OSError as error:
        raise AcquisitionError(f"Could not read configuration file: {error}") from error
    if not isinstance(config, dict):
        raise AcquisitionError("Configuration must contain a top-level YAML mapping.")
    return config


def dataset_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the Phase 2 dataset settings and return them."""
    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        raise AcquisitionError("Missing required configuration section: dataset.")
    acquisition = dataset.get("acquisition")
    if not isinstance(acquisition, dict):
        raise AcquisitionError("Missing required configuration section: dataset.acquisition.")
    required = ("name", "split", "max_samples", "quick_samples", "source_language", "target_language")
    missing = [key for key in required if key not in dataset]
    if missing:
        raise AcquisitionError("Missing dataset settings: " + ", ".join(missing))
    source, target = dataset["source_language"], dataset["target_language"]
    if not isinstance(source, dict) or not isinstance(target, dict):
        raise AcquisitionError("Source and target language settings must be mappings.")
    for key in ("field",):
        if not isinstance(source.get(key), str) or not source[key]:
            raise AcquisitionError(f"dataset.source_language.{key} must be a non-empty string.")
    if not isinstance(target.get("code"), str) or not target["code"]:
        raise AcquisitionError("dataset.target_language.code must be a non-empty string.")
    for key in ("max_samples", "quick_samples"):
        if isinstance(dataset[key], bool) or not isinstance(dataset[key], int) or dataset[key] <= 0:
            raise AcquisitionError(f"dataset.{key} must be a positive integer.")
    multiplier = acquisition.get("scan_multiplier")
    if isinstance(multiplier, bool) or not isinstance(multiplier, int) or multiplier <= 0:
        raise AcquisitionError("dataset.acquisition.scan_multiplier must be a positive integer.")
    for key in ("output_file", "metadata_file"):
        if not isinstance(acquisition.get(key), str) or not acquisition[key]:
            raise AcquisitionError(f"dataset.acquisition.{key} must be a non-empty string.")
    return dataset


def get_requested_limit(dataset: dict[str, Any], mode: str, limit: int | None) -> int:
    """Resolve a safe requested-pair limit without bypassing max_samples."""
    if mode not in ("quick", "normal"):
        raise AcquisitionError("Mode must be 'quick' or 'normal'.")
    maximum = dataset["max_samples"]
    requested = dataset["quick_samples"] if mode == "quick" else maximum
    if limit is None:
        return requested
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise AcquisitionError("--limit must be a positive integer.")
    if limit > maximum:
        raise AcquisitionError(f"--limit ({limit}) cannot exceed dataset.max_samples ({maximum}).")
    return limit


def extract_translation(captions: Any, language_code: str) -> str | None:
    """Find a non-empty translation in expected list/tuple caption entries."""
    if not isinstance(captions, (list, tuple)):
        return None
    for entry in captions:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        if entry[0] == language_code and isinstance(entry[1], str):
            translation = entry[1].strip()
            return translation or None
    return None


def captions_are_malformed(captions: Any) -> bool:
    """Identify structurally invalid caption containers without raising errors."""
    if not isinstance(captions, (list, tuple)):
        return True
    return any(not isinstance(entry, (list, tuple)) or len(entry) < 2 for entry in captions)


def extract_pair(record: Any, source_field: str, target_code: str) -> tuple[dict[str, Any] | None, str | None]:
    """Return a minimally valid pair and a rejection reason, if any."""
    if not isinstance(record, dict):
        return None, "malformed_captions"
    english = record.get(source_field)
    if not isinstance(english, str) or not english.strip():
        return None, "missing_english"
    captions = record.get("captions")
    tagalog = extract_translation(captions, target_code)
    if tagalog is None:
        return None, "malformed_captions" if captions_are_malformed(captions) else "missing_tagalog"
    pair = {"source_id": str(record.get("id", "")), "english": english.strip(), "tagalog": tagalog}
    if "score" in record and isinstance(record["score"], (int, float)) and not isinstance(record["score"], bool):
        pair["score"] = record["score"]
    return pair, None


def get_output_paths(dataset: dict[str, Any]) -> tuple[Path, Path]:
    """Resolve configured output paths within the project directory."""
    acquisition = dataset["acquisition"]
    output = PROJECT_ROOT / acquisition["output_file"]
    metadata = PROJECT_ROOT / acquisition["metadata_file"]
    return output, metadata


def existing_output_has_data(path: Path) -> bool:
    """Check whether a previous acquisition has a non-empty JSONL output."""
    return path.is_file() and path.stat().st_size > 0


def write_metadata(path: Path, dataset: dict[str, Any], stats: AcquisitionStats) -> None:
    """Atomically write small acquisition metadata after the JSONL is finalized."""
    metadata = {
        "dataset": dataset["name"],
        "split": dataset["split"],
        "target_language": dataset["target_language"]["code"],
        **asdict(stats),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
        metadata_file.write("\n")
    temporary.replace(path)


def default_dataset_loader(name: str, split: str, streaming: bool) -> Iterable[dict[str, Any]]:
    """Import Datasets only when a real remote acquisition is requested."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise AcquisitionError("The 'datasets' package is unavailable. Install requirements first.") from error
    if not streaming:
        raise AcquisitionError("Dataset acquisition must use streaming=True.")
    return load_dataset(name, split=split, streaming=True)


def acquire_dataset(
    config: dict[str, Any],
    mode: str = "quick",
    limit: int | None = None,
    force: bool = False,
    progress_callback: Callable[[AcquisitionStats], None] | None = None,
    dataset_loader: Callable[[str, str, bool], Iterable[dict[str, Any]]] = default_dataset_loader,
) -> AcquisitionStats:
    """Stream records, write valid pairs incrementally, and finalize atomically."""
    dataset = dataset_config(config)
    requested = get_requested_limit(dataset, mode, limit)
    scan_limit = requested * dataset["acquisition"]["scan_multiplier"]
    output_path, metadata_path = get_output_paths(dataset)
    if existing_output_has_data(output_path) and not force:
        raise AcquisitionError(f"Raw dataset already exists at {output_path}. Use --force to replace it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    temporary_path.unlink(missing_ok=True)
    stats = AcquisitionStats(requested_pairs=requested, scan_limit=scan_limit, mode=mode)
    source_field = dataset["source_language"]["field"]
    target_code = dataset["target_language"]["code"]
    progress = tqdm(total=requested, desc="Acquiring", unit="pair", disable=progress_callback is not None)

    try:
        try:
            records = dataset_loader(dataset["name"], dataset["split"], True)
            iterator = iter(records)
        except AcquisitionError:
            raise
        except Exception as error:
            raise AcquisitionError(f"Could not start dataset stream: {error}") from error

        with temporary_path.open("w", encoding="utf-8") as output_file:
            while stats.records_scanned < scan_limit and stats.pairs_saved < requested:
                try:
                    record = next(iterator)
                except StopIteration:
                    stats.stop_reason = "stream_exhausted"
                    break
                except Exception as error:
                    raise AcquisitionError(f"Dataset stream interrupted: {error}") from error
                stats.records_scanned += 1
                pair, reason = extract_pair(record, source_field, target_code)
                if pair is None:
                    setattr(stats, reason or "malformed_captions", getattr(stats, reason or "malformed_captions") + 1)
                else:
                    json.dump(pair, output_file, ensure_ascii=False)
                    output_file.write("\n")
                    stats.pairs_saved += 1
                    progress.update(1)
                if progress_callback:
                    progress_callback(stats)
            else:
                stats.stop_reason = "target_reached" if stats.pairs_saved == requested else "scan_limit_reached"

        if stats.pairs_saved == requested:
            stats.completed, stats.stop_reason = True, "target_reached"
        elif stats.stop_reason == "running":
            stats.stop_reason = "scan_limit_reached"
        temporary_path.replace(output_path)
        write_metadata(metadata_path, dataset, stats)
        return stats
    except KeyboardInterrupt:
        temporary_path.unlink(missing_ok=True)
        raise AcquisitionError("Acquisition cancelled. No partial dataset was saved.")
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        progress.close()


def print_summary(config: dict[str, Any], stats: AcquisitionStats) -> None:
    """Print a concise final acquisition summary."""
    dataset = dataset_config(config)
    output_path, _ = get_output_paths(dataset)
    print("=" * 60)
    print("PHASE 2 DATASET ACQUISITION")
    print("=" * 60)
    print(f"Dataset             : {dataset['name']}")
    print(f"Split               : {dataset['split']}")
    print(f"Mode                : {stats.mode}")
    print(f"Requested pairs     : {stats.requested_pairs:,}")
    print(f"Records scanned     : {stats.records_scanned:,}")
    print(f"Pairs saved         : {stats.pairs_saved:,}")
    print(f"Missing English     : {stats.missing_english:,}")
    print(f"Missing Tagalog     : {stats.missing_tagalog:,}")
    print(f"Malformed captions  : {stats.malformed_captions:,}")
    print(f"Stop reason         : {stats.stop_reason}")
    print(f"Output              : {output_path.relative_to(PROJECT_ROOT)}")
    print("Status              : " + ("SUCCESS" if stats.completed else "INCOMPLETE"))
    print("=" * 60)


def parse_arguments() -> argparse.Namespace:
    """Parse the small, deliberately constrained CLI surface."""
    parser = argparse.ArgumentParser(description="Stream a limited English–Tagalog raw dataset.")
    parser.add_argument("--mode", choices=("quick", "normal"), default="quick")
    parser.add_argument("--limit", type=int, help="Safe pair limit; cannot exceed dataset.max_samples.")
    parser.add_argument("--force", action="store_true", help="Replace an existing raw acquisition after success.")
    return parser.parse_args()


def main() -> int:
    """Run dataset acquisition from the command line."""
    arguments = parse_arguments()
    try:
        config = load_config()
        dataset = dataset_config(config)
        requested = get_requested_limit(dataset, arguments.mode, arguments.limit)
        scan_limit = requested * dataset["acquisition"]["scan_multiplier"]
        print("Dataset Acquisition")
        print(f"Mode: {arguments.mode.upper()}")
        print(f"Target: {requested:,} pairs")
        print(f"Maximum Scan: {scan_limit:,} records")
        stats = acquire_dataset(config, arguments.mode, arguments.limit, arguments.force)
        print_summary(config, stats)
        return 0 if stats.completed else 2
    except AcquisitionError as error:
        print(f"PHASE 2 DATASET ACQUISITION: FAILED\nError: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
