"""Bounded JSONL translation datasets and dynamic padding for Phase 6."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.tokenizer_utils import SentencePieceTokenizer


class TranslationDataset(Dataset):
    """Indexes usable JSONL pairs; tokenizes on access, not into a tensor cache."""

    def __init__(self, path: Path, tokenizer: SentencePieceTokenizer, max_length: int, limit: int | None = None) -> None:
        self.path, self.tokenizer, self.max_length = path, tokenizer, max_length
        self.records: list[tuple[str, str]] = []
        self.skipped_source_too_long = self.skipped_target_too_long = 0
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if limit is not None and len(self.records) >= limit:
                    break
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                source, target = record.get("english"), record.get("tagalog")
                if not isinstance(source, str) or not isinstance(target, str):
                    continue
                source_ids = tokenizer.encode(source, add_bos=True, add_eos=True)
                target_ids = tokenizer.encode(target, add_bos=True, add_eos=True)
                if len(source_ids) > max_length:
                    self.skipped_source_too_long += 1
                    continue
                if len(target_ids) > max_length:
                    self.skipped_target_too_long += 1
                    continue
                self.records.append((source, target))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[list[int], list[int], list[int]]:
        source, target = self.records[index]
        source_ids = self.tokenizer.encode(source, add_bos=True, add_eos=True)
        full_target = self.tokenizer.encode(target, add_bos=True, add_eos=True)
        return source_ids, full_target[:-1], full_target[1:]  # Teacher forcing: previous token predicts next token.


def collate_translation_batch(batch: list[tuple[list[int], list[int], list[int]]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dynamically pad source, decoder-input, and expected-output sequences."""
    def pad(sequences: list[list[int]]) -> torch.Tensor:
        length = max(len(sequence) for sequence in sequences)
        return torch.tensor([sequence + [pad_id] * (length - len(sequence)) for sequence in sequences], dtype=torch.long)
    sources, decoder_inputs, expected_outputs = zip(*batch)
    return pad(list(sources)), pad(list(decoder_inputs)), pad(list(expected_outputs))
