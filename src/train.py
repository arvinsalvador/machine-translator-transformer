"""Resource-limited Phase 6 training with checkpoints, validation, and resume."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.model import build_model
from src.tokenizer_utils import load_tokenizer
from src.training_data import TranslationDataset, collate_translation_batch
from src.train_tokenizer import get_paths as tokenizer_paths, load_config, validate_config as validate_tokenizer_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TrainingError(Exception): pass


def paths(config: dict[str, Any]) -> dict[str, Path]:
    dataset, preprocessing, tokenizer_settings = validate_tokenizer_config(config)
    train, prep_meta, prefix, tokenizer_meta = tokenizer_paths(dataset, preprocessing, tokenizer_settings)
    validation = PROJECT_ROOT / preprocessing["output_directory"] / preprocessing["validation_file"]
    checkpoint = config["training"]["checkpoint"]
    return {"train": train, "validation": validation, "preparation": prep_meta, "tokenizer": prefix.with_suffix(".model"), "tokenizer_meta": tokenizer_meta, **{key: PROJECT_ROOT / checkpoint[key] for key in ("best_file", "last_file", "history_file", "metadata_file")}}


def check_preconditions(config: dict[str, Any]) -> dict[str, Path]:
    result = paths(config)
    for key in ("train", "validation", "tokenizer"):
        if not result[key].is_file() or result[key].stat().st_size == 0:
            raise TrainingError(f"Required {key} artifact is unavailable. Complete the preceding phase first.")
    for key in ("preparation", "tokenizer_meta"):
        try:
            if json.loads(result[key].read_text(encoding="utf-8")).get("completed") is not True:
                raise TrainingError(f"Required metadata is incomplete: {result[key]}")
        except (OSError, json.JSONDecodeError) as error:
            raise TrainingError(f"Could not validate metadata: {result[key]}") from error
    return result


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def save_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary); os.replace(temporary, path)


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, optimizer: AdamW | None = None, clip_norm: float = 1.0, callback: Callable[[dict], None] | None = None) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = token_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, (source, decoder_input, expected) in enumerate(loader, 1):
            source, decoder_input, expected = source.to(device), decoder_input.to(device), expected.to(device)
            if training: optimizer.zero_grad(set_to_none=True)
            logits = model(source, decoder_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), expected.reshape(-1))
            if training:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm); optimizer.step()
            valid = expected.ne(model.pad_id).sum().item()
            total_loss += loss.item() * valid; token_count += valid
            if callback: callback({"event": "batch_progress", "batch": batch_index, "total_batches": len(loader), "loss": loss.item()})
    return total_loss / max(token_count, 1)


def train_model(config: dict[str, Any], profile: str = "quick", resume: bool = False, force: bool = False, progress_callback: Callable[[dict], None] | None = None) -> dict[str, Any]:
    if profile not in ("quick", "normal"): raise TrainingError("Profile must be quick or normal.")
    result = check_preconditions(config); settings = config["training"]; profile_settings = settings["profiles"][profile]
    existing = result["last_file"].exists() or result["best_file"].exists()
    if existing and not (resume or force): raise TrainingError("Checkpoints already exist. Use --resume or --force.")
    set_seed(config["project"]["random_seed"])
    tokenizer = load_tokenizer(result["tokenizer"]); pad_id = tokenizer.pad_id
    max_length = config["model"]["max_sequence_length"]
    train_data = TranslationDataset(result["train"], tokenizer, max_length, profile_settings["max_train_records"])
    validation_data = TranslationDataset(result["validation"], tokenizer, max_length, profile_settings["max_validation_records"])
    if not train_data or not validation_data: raise TrainingError("No usable train or validation records remain after token-length safeguards.")
    collate = lambda batch: collate_translation_batch(batch, pad_id)
    workers = min(int(config["resources"]["dataloader_workers"]), 2)
    train_loader = DataLoader(train_data, batch_size=settings["batch_size"], shuffle=bool(settings.get("shuffle", True)), num_workers=workers, collate_fn=collate)
    validation_loader = DataLoader(validation_data, batch_size=settings["batch_size"], shuffle=False, num_workers=workers, collate_fn=collate)
    device = torch.device("cuda" if config["resources"].get("prefer_gpu") and torch.cuda.is_available() else "cpu")
    model = build_model(config, tokenizer).to(device); criterion = nn.CrossEntropyLoss(ignore_index=pad_id); optimizer = AdamW(model.parameters(), lr=settings["learning_rate"])
    history: list[dict[str, Any]] = []; start_epoch = 1; best = float("inf"); best_epoch = 0; stale = 0
    if resume:
        if not result["last_file"].is_file(): raise TrainingError("No last checkpoint exists to resume.")
        checkpoint = torch.load(result["last_file"], map_location=device, weights_only=False)
        if checkpoint.get("vocab_size") != model.vocab_size or checkpoint.get("pad_id") != pad_id or checkpoint.get("model_config") != config["model"]: raise TrainingError("Checkpoint architecture/tokenizer is incompatible with current configuration.")
        model.load_state_dict(checkpoint["model_state_dict"]); optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch, best, best_epoch, stale, history = checkpoint["epoch"] + 1, checkpoint["best_validation_loss"], checkpoint["best_epoch"], checkpoint["early_stopping_counter"], checkpoint["history"]
    requested_epochs = profile_settings["epochs"]
    for epoch in range(start_epoch, requested_epochs + 1):
        started = time.time()
        if progress_callback: progress_callback({"event": "epoch_started", "epoch": epoch, "epochs": requested_epochs})
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer, settings["gradient_clip_norm"], progress_callback)
        validation_loss = run_epoch(model, validation_loader, criterion, device)
        entry = {"epoch": epoch, "training_loss": train_loss, "validation_loss": validation_loss, "learning_rate": settings["learning_rate"], "duration_seconds": time.time() - started}; history.append(entry)
        improved = validation_loss < best
        if improved:
            best, best_epoch, stale = validation_loss, epoch, 0
            save_atomic({"model_state_dict": model.state_dict(), "model_config": config["model"], "vocab_size": model.vocab_size, "pad_id": pad_id, "epoch": epoch, "validation_loss": validation_loss}, result["best_file"])
        else: stale += 1
        checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "best_validation_loss": best, "best_epoch": best_epoch, "early_stopping_counter": stale, "history": history, "model_config": config["model"], "vocab_size": model.vocab_size, "pad_id": pad_id}
        save_atomic(checkpoint, result["last_file"])
        if progress_callback: progress_callback({"event": "epoch_completed", **entry, "best_validation_loss": best})
        if settings["early_stopping"].get("enabled") and stale >= settings["early_stopping"]["patience"]: break
    completed = True
    result["history_file"].write_text(json.dumps({"profile": profile, "epochs": history}, indent=2), encoding="utf-8")
    metadata = {"profile": profile, "training_records": len(train_data), "validation_records": len(validation_data), "batch_size": settings["batch_size"], "requested_epochs": requested_epochs, "completed_epochs": len(history), "learning_rate": settings["learning_rate"], "optimizer": "AdamW", "gradient_clip_norm": settings["gradient_clip_norm"], "best_validation_loss": best, "best_epoch": best_epoch, "device": str(device), "random_seed": config["project"]["random_seed"], "completed": completed, "stopped_early": len(history) < requested_epochs, "completed_at": datetime.now(timezone.utc).isoformat()}
    result["metadata_file"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {**metadata, "history": history}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--profile", choices=("quick", "normal"), default="quick"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--force", action="store_true"); args = parser.parse_args()
    try:
        output = train_model(load_config(), args.profile, args.resume, args.force)
        print(f"Training complete: best validation loss {output['best_validation_loss']:.4f} at epoch {output['best_epoch']}")
        return 0
    except TrainingError as error:
        print(f"PHASE 6 TRAINING: FAILED\nError: {error}", file=sys.stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
