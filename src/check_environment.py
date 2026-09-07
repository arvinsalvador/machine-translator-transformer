"""Validate the Phase 1 environment without downloading data or training."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
REQUIRED_SECTIONS = ("project", "dataset", "tokenizer", "model", "training", "resources")
REQUIRED_DIRECTORIES = ("data/raw", "data/processed", "data/tokenizer", "models/checkpoints", "logs")


def load_config(path: Path) -> dict[str, Any]:
    """Safely load the YAML configuration as a mapping."""
    if not path.is_file():
        raise ValueError(f"Configuration file was not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise ValueError(f"Configuration YAML is malformed: {error}") from error
    except OSError as error:
        raise ValueError(f"Could not read configuration file: {error}") from error
    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a top-level YAML mapping.")
    return config


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Get a required mapping section with a useful validation error."""
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid required configuration section: '{name}'.")
    return value


def required(values: dict[str, Any], key: str, label: str) -> Any:
    """Get a required setting with a useful validation error."""
    if key not in values:
        raise ValueError(f"Missing required setting: '{label}.{key}'.")
    return values[key]


def positive_integer(value: Any, label: str) -> None:
    """Ensure a setting is a positive integer (but not a Boolean)."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer.")


def validate_config(config: dict[str, Any]) -> None:
    """Check important configuration sections and safe relationships."""
    for name in REQUIRED_SECTIONS:
        section(config, name)
    dataset, tokenizer = section(config, "dataset"), section(config, "tokenizer")
    model, training = section(config, "model"), section(config, "training")
    resources = section(config, "resources")

    d_model = required(model, "d_model", "model")
    heads = required(model, "attention_heads", "model")
    positive_integer(d_model, "model.d_model")
    positive_integer(heads, "model.attention_heads")
    if d_model % heads:
        raise ValueError("model.d_model must be divisible by model.attention_heads.")

    ratios = [required(dataset, key, "dataset") for key in ("train_ratio", "validation_ratio", "test_ratio")]
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in ratios):
        raise ValueError("Dataset split ratios must be numeric.")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Dataset train, validation, and test ratios must total 1.0.")

    max_samples = required(dataset, "max_samples", "dataset")
    quick_samples = required(dataset, "quick_samples", "dataset")
    positive_integer(max_samples, "dataset.max_samples")
    if isinstance(quick_samples, bool) or not isinstance(quick_samples, int) or quick_samples < 0:
        raise ValueError("dataset.quick_samples must be a non-negative integer.")
    if quick_samples > max_samples:
        raise ValueError("dataset.quick_samples must not exceed dataset.max_samples.")

    for values, key, label in ((tokenizer, "vocab_size", "tokenizer.vocab_size"),
                               (training, "batch_size", "training.batch_size"),
                               (training, "epochs", "training.epochs"),
                               (resources, "dataloader_workers", "resources.dataloader_workers"),
                               (resources, "max_cpu_threads", "resources.max_cpu_threads")):
        positive_integer(required(values, key, label.rsplit(".", 1)[0]), label)


def validate_directories() -> None:
    """Verify the empty storage locations required by later phases."""
    missing = [directory for directory in REQUIRED_DIRECTORIES if not (PROJECT_ROOT / directory).is_dir()]
    if missing:
        raise ValueError("Required directories are missing: " + ", ".join(missing))


def print_report(config: dict[str, Any], device: str) -> None:
    """Print the Phase 1 readiness report."""
    project, dataset = config["project"], config["dataset"]
    tokenizer, model = config["tokenizer"], config["model"]
    training, resources = config["training"], config["resources"]
    print("=" * 60)
    print("ENGLISH–TAGALOG TRANSLATOR: ENVIRONMENT REPORT")
    print("=" * 60)
    print(f"Project: {project['name']} ({project['version']})")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Selected device: {device}")
    print("Configured CPU limit: 2 CPUs")
    print()
    print(f"Dataset: {dataset['name']}")
    print(f"Maximum samples: {dataset['max_samples']}")
    print(f"Quick/debug samples: {dataset['quick_samples']}")
    print(f"Source language: {dataset['source_language']['name']} ({dataset['source_language']['code']})")
    print(f"Target language: {dataset['target_language']['name']} ({dataset['target_language']['code']})")
    print(f"Vocabulary size: {tokenizer['vocab_size']}")
    print(f"Maximum sequence length: {model['max_sequence_length']}")
    print(f"d_model: {model['d_model']}; attention heads: {model['attention_heads']}")
    print(f"Encoder layers: {model['encoder_layers']}; decoder layers: {model['decoder_layers']}")
    print(f"Feedforward dimension: {model['feedforward_dimension']}")
    print(f"Batch size: {training['batch_size']}; maximum epochs: {training['epochs']}")
    print(f"Learning rate: {training['learning_rate']}")
    print(f"Configured dataloader workers: {resources['dataloader_workers']}")
    print(f"Configured maximum CPU threads: {resources['max_cpu_threads']}")
    print("=" * 60)
    print("PHASE 1 ENVIRONMENT CHECK: PASSED")
    print("=" * 60)


def main() -> int:
    """Run safe configuration and runtime checks."""
    try:
        config = load_config(CONFIG_PATH)
        validate_config(config)
        validate_directories()
        resources = config["resources"]
        device = "cuda" if resources["prefer_gpu"] and torch.cuda.is_available() else "cpu"
        if device == "cpu":
            torch.set_num_threads(resources["max_cpu_threads"])
        print_report(config, device)
        return 0
    except (KeyError, TypeError, ValueError) as error:
        print(f"PHASE 1 ENVIRONMENT CHECK: FAILED\nError: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
