"""Safe final project and classroom-demo readiness validation.

This module reads small metadata files and a few JSONL records. Demo mode also
loads the existing tokenizer/checkpoint, runs one synthetic forward pass, and
performs one real inference. It never invokes acquisition or any training step.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import sacrebleu
import sentencepiece
import streamlit
import torch
import yaml

from src.check_environment import validate_config as validate_environment_config
from src.inference import InferenceError, load_inference_components, translate_text
from src.inspect_model import synthetic_validation
from src.model import architecture_summary, build_model
from src.tokenizer_utils import load_tokenizer
from src.train import PROJECT_ROOT, paths as training_paths
from src.train_tokenizer import load_config


PASS = "PASS"
WARNING = "WARNING"
FAIL = "FAIL"
SKIPPED = "SKIPPED"


@dataclass
class CheckResult:
    """One concise validation outcome."""

    name: str
    status: str
    message: str
    metrics: dict[str, Any] | None = None


def result(name: str, status: str, message: str, **metrics: Any) -> CheckResult:
    if status not in {PASS, WARNING, FAIL, SKIPPED}:
        raise ValueError(f"Unknown validation status: {status}")
    return CheckResult(name, status, message, metrics or None)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def sample_jsonl(path: Path, count: int = 3) -> list[dict[str, Any]]:
    """Read at most a few non-empty JSONL records."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if len(records) >= count:
                break
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Non-object JSONL record in {path}.")
            records.append(value)
    return records


def check_environment(config: dict[str, Any]) -> CheckResult:
    try:
        validate_environment_config(config)
        return result("Environment", PASS, "Runtime dependencies and configuration loaded.", python=platform.python_version(), pytorch=torch.__version__)
    except Exception as error:
        return result("Environment", FAIL, str(error))


def check_directories(root: Path) -> CheckResult:
    required = [root / value for value in ("data/raw", "data/processed", "data/tokenizer", "models/checkpoints", "logs")]
    missing = [str(path.relative_to(root)) for path in required if not path.is_dir()]
    return result("Directories", FAIL, "Missing: " + ", ".join(missing)) if missing else result("Directories", PASS, "Required project directories exist.")


def check_dataset(config: dict[str, Any], root: Path, status_only: bool) -> CheckResult:
    acquisition = config["dataset"]["acquisition"]
    raw, metadata_path = root / acquisition["output_file"], root / acquisition["metadata_file"]
    if not raw.is_file() or not metadata_path.is_file():
        return result("Dataset Acquisition", FAIL, "Raw dataset or acquisition metadata is missing.")
    try:
        metadata = read_json(metadata_path)
        if metadata.get("completed") is not True or not isinstance(metadata.get("pairs_saved"), int) or metadata["pairs_saved"] <= 0:
            return result("Dataset Acquisition", FAIL, "Acquisition metadata is incomplete or invalid.")
        if not status_only:
            samples = sample_jsonl(raw)
            if not samples or any(not isinstance(row.get(key), str) or not row[key].strip() for row in samples for key in ("english", "tagalog")):
                return result("Dataset Acquisition", FAIL, "Raw JSONL sample is structurally invalid.")
        return result("Dataset Acquisition", PASS, "Completed raw acquisition is available.", pairs=metadata["pairs_saved"], file_bytes=raw.stat().st_size)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return result("Dataset Acquisition", FAIL, str(error))


def check_preparation(config: dict[str, Any], root: Path, status_only: bool) -> CheckResult:
    settings = config["preprocessing"]
    directory = root / settings["output_directory"]
    paths = {name: directory / settings[f"{name}_file"] for name in ("train", "validation", "test")}
    metadata_path = directory / settings["metadata_file"]
    if any(not path.is_file() for path in (*paths.values(), metadata_path)):
        return result("Data Preparation", FAIL, "One or more prepared split artifacts are missing.")
    try:
        metadata = read_json(metadata_path)
        counts = [metadata.get(f"{name}_records") for name in ("train", "validation", "test")]
        if metadata.get("completed") is not True or not all(isinstance(value, int) and value >= 0 for value in counts) or sum(counts) != metadata.get("accepted_records"):
            return result("Data Preparation", FAIL, "Preparation metadata counts are invalid.")
        if not status_only:
            for path in paths.values():
                if not sample_jsonl(path, 1):
                    return result("Data Preparation", FAIL, f"Prepared split is empty: {path.name}")
        return result("Data Preparation", PASS, "Prepared splits and exact counts are valid.", train=counts[0], validation=counts[1], test=counts[2], clean=metadata["accepted_records"])
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return result("Data Preparation", FAIL, str(error))


def tokenizer_locations(config: dict[str, Any], root: Path) -> tuple[Path, Path, Path]:
    prefix = root / config["tokenizer"]["model_prefix"]
    return prefix.with_suffix(".model"), prefix.with_suffix(".vocab"), prefix.parent / config["tokenizer"]["metadata_file"]


def check_tokenizer(config: dict[str, Any], root: Path, status_only: bool) -> CheckResult:
    model_path, vocab_path, metadata_path = tokenizer_locations(config, root)
    if any(not path.is_file() for path in (model_path, vocab_path, metadata_path)):
        return result("Tokenizer", FAIL, "Tokenizer model, vocabulary, or metadata is missing.")
    if status_only:
        return result("Tokenizer", PASS, "Tokenizer artifacts are available.", file_bytes=model_path.stat().st_size)
    try:
        tokenizer = load_tokenizer(model_path)
        special = tokenizer.get_special_token_ids()
        if any(value < 0 for value in special.values()):
            return result("Tokenizer", FAIL, "One or more special token IDs are unavailable.")
        for text in ("A student is reading a book.", "Ang estudyante ay nagbabasa ng libro."):
            if not tokenizer.decode(tokenizer.encode(text)):
                return result("Tokenizer", FAIL, "Tokenizer round-trip produced empty text.")
        return result("Tokenizer", PASS, "English/Tagalog encode-decode checks passed.", vocabulary=tokenizer.get_vocab_size(), **special)
    except Exception as error:
        return result("Tokenizer", FAIL, str(error))


def check_model(config: dict[str, Any], root: Path, status_only: bool) -> CheckResult:
    if status_only:
        return result("Transformer Model", SKIPPED, "Synthetic forward pass omitted in status-only mode.")
    try:
        tokenizer = load_tokenizer(tokenizer_locations(config, root)[0])
        model = build_model(config, tokenizer)
        shape = synthetic_validation(model, tokenizer)
        summary = architecture_summary(model, config)
        return result("Transformer Model", PASS, f"Synthetic forward pass succeeded: {shape}.", parameters=summary["total"], vocabulary=summary["vocab_size"])
    except Exception as error:
        return result("Transformer Model", FAIL, str(error))


def check_training(config: dict[str, Any], root: Path, status_only: bool) -> CheckResult:
    locations = training_paths(config)
    required = (locations["best_file"], locations["last_file"], locations["metadata_file"], locations["history_file"])
    if any(not path.is_file() for path in required):
        return result("Training Checkpoint", FAIL, "Required Phase 6 training artifacts are missing.")
    try:
        metadata = read_json(locations["metadata_file"])
        if metadata.get("completed") is not True:
            return result("Training Checkpoint", FAIL, "Training metadata does not report completion.")
        if not status_only:
            load_inference_components(config)  # Strict architecture, tokenizer, and state-dict validation.
        return result("Training Checkpoint", PASS, "Best checkpoint and training metadata are compatible.", best_epoch=metadata.get("best_epoch"), validation_loss=metadata.get("best_validation_loss"), profile=metadata.get("profile"))
    except (OSError, ValueError, json.JSONDecodeError, InferenceError) as error:
        return result("Training Checkpoint", FAIL, str(error))


def check_evaluation(config: dict[str, Any], root: Path) -> CheckResult:
    path = root / config["evaluation"]["results_file"]
    if not path.is_file():
        return result("Evaluation", WARNING, "Evaluation results are unavailable; run Phase 6 evaluation before presenting.")
    try:
        metadata = read_json(path)
        if not isinstance(metadata.get("bleu"), (int, float)) or not isinstance(metadata.get("chrf"), (int, float)) or not isinstance(metadata.get("test_samples_evaluated"), int) or metadata["test_samples_evaluated"] <= 0:
            return result("Evaluation", FAIL, "Evaluation metadata contains invalid metrics.")
        return result("Evaluation", PASS, "Saved evaluation metrics are valid.", bleu=metadata["bleu"], chrf=metadata["chrf"], samples=metadata["test_samples_evaluated"])
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return result("Evaluation", FAIL, str(error))


def check_inference(config: dict[str, Any], status_only: bool) -> CheckResult:
    if status_only:
        return result("Inference", SKIPPED, "Real inference omitted in status-only mode.")
    try:
        output = translate_text("A student is reading a book.", config)
        if not isinstance(output.get("translation"), str) or output["generated_token_count"] > config["model"]["max_sequence_length"]:
            return result("Inference", FAIL, "Inference returned an invalid or unbounded result.")
        return result("Inference", PASS, "One real trained-model inference completed.", generated_tokens=output["generated_token_count"], duration_ms=output["duration_ms"], device=output["device"])
    except Exception as error:
        return result("Inference", FAIL, str(error))


def check_ui() -> CheckResult:
    try:
        module = importlib.import_module("app.app")
        expected = ("dashboard", "dataset_page", "preparation_page", "tokenizer_page", "model_page", "training_page", "evaluation_page", "translator_page", "about_page")
        missing = [name for name in expected if not callable(getattr(module, name, None))]
        return result("Streamlit Imports", FAIL, "Missing pages: " + ", ".join(missing)) if missing else result("Streamlit Imports", PASS, "All final navigation page functions import successfully.")
    except Exception as error:
        return result("Streamlit Imports", FAIL, str(error))


def check_resources(config: dict[str, Any], root: Path) -> CheckResult:
    try:
        compose = yaml.safe_load((root / "docker-compose.yml").read_text(encoding="utf-8"))["services"]["translator"]
        environment = compose.get("environment", {})
        okay = float(compose.get("cpus", 0)) <= 2 and str(compose.get("mem_limit", "")).lower() == "4g" and environment.get("PYTHONPATH") == "/app" and config["training"]["batch_size"] <= 16 and config["resources"]["dataloader_workers"] <= 2
        return result("Resource Configuration", PASS, "2 CPU / 4 GB, bounded batch/workers, and PYTHONPATH are intact.") if okay else result("Resource Configuration", FAIL, "One or more resource safeguards have regressed.")
    except Exception as error:
        return result("Resource Configuration", FAIL, str(error))


def overall_status(checks: list[CheckResult]) -> str:
    critical = {"Environment", "Directories", "Dataset Acquisition", "Data Preparation", "Tokenizer", "Transformer Model", "Training Checkpoint", "Inference", "Streamlit Imports", "Resource Configuration"}
    if any(item.status == FAIL and item.name in critical for item in checks):
        return "NOT READY"
    if any(item.status in {WARNING, SKIPPED} for item in checks):
        return "READY WITH WARNINGS"
    return "READY FOR DEMONSTRATION"


def validate_project(config: dict[str, Any], root: Path = PROJECT_ROOT, status_only: bool = False) -> tuple[list[CheckResult], str]:
    """Run safe checks only; deliberately contains no mutating pipeline calls."""
    checks = [check_environment(config), check_directories(root), check_dataset(config, root, status_only), check_preparation(config, root, status_only), check_tokenizer(config, root, status_only), check_model(config, root, status_only), check_training(config, root, status_only), check_evaluation(config, root), check_inference(config, status_only), check_ui(), check_resources(config, root)]
    return checks, overall_status(checks)


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely validate project and demo readiness.")
    parser.add_argument("--demo", action="store_true", help="Run synthetic model and one real inference smoke test.")
    parser.add_argument("--status-only", action="store_true", help="Report artifact availability without model/inference execution.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable results.")
    args = parser.parse_args()
    try:
        checks, overall = validate_project(load_config(), status_only=args.status_only)
    except Exception as error:
        print(f"Project validation could not start: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"checks": [asdict(item) for item in checks], "overall": overall}, indent=2))
    else:
        print("=" * 68); print("TRANSFORMER TRANSLATOR – DEMO READINESS"); print("=" * 68)
        for item in checks: print(f"{item.name:25} {item.status:8} {item.message}")
        print("-" * 68); print(f"Overall Status: {overall}"); print("=" * 68)
    return 1 if overall == "NOT READY" else 0


if __name__ == "__main__":
    raise SystemExit(main())
