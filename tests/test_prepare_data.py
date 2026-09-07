"""Offline synthetic tests for Phase 3 preprocessing and exact splitting."""

import json
import tempfile
import unittest
from pathlib import Path

from src.prepare_data import PreparationError, calculate_split_counts, normalize_text, prepare_dataset, validate_record


def config_for(directory: Path) -> dict:
    raw = directory / "raw.jsonl"
    return {"project": {"random_seed": 42}, "dataset": {"train_ratio": 0.8, "validation_ratio": 0.1, "test_ratio": 0.1, "acquisition": {"output_file": str(raw), "metadata_file": str(directory / "acquisition_metadata.json")}}, "preprocessing": {"normalize_unicode": True, "collapse_whitespace": True, "remove_control_characters": True, "deduplicate": True, "min_words": 1, "max_words": 3, "max_characters": 30, "output_directory": str(directory / "processed"), "train_file": "train.jsonl", "validation_file": "validation.jsonl", "test_file": "test.jsonl", "metadata_file": "preparation_metadata.json", "preview_rows": 5}}


def write_raw(path: Path, records: list[object]) -> None:
    with path.open("w", encoding="utf-8") as raw_file:
        for record in records:
            raw_file.write(record if isinstance(record, str) else json.dumps(record, ensure_ascii=False))
            raw_file.write("\n")


class PrepareDataTests(unittest.TestCase):
    def test_conservative_whitespace_and_unicode_normalization(self) -> None:
        settings = config_for(Path("."))["preprocessing"]
        self.assertEqual(normalize_text("  A   dog\n is\t running. ", settings), "A dog is running.")
        self.assertEqual(normalize_text("Magandang umaga.", settings), "Magandang umaga.")

    def test_missing_empty_and_length_rejections(self) -> None:
        settings = config_for(Path("."))["preprocessing"]
        self.assertEqual(validate_record({"tagalog": "Hi"}, settings)[1], "missing_english")
        self.assertEqual(validate_record({"english": "Hi", "tagalog": ""}, settings)[1], "empty_tagalog")
        self.assertEqual(validate_record({"english": "one two three four", "tagalog": "apat"}, settings)[1], "too_long")
        self.assertEqual(validate_record({"english": "a" * 31, "tagalog": "b"}, settings)[1], "too_long")

    def test_preparation_handles_bad_json_deduplicates_and_writes_unicode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory, config = Path(temporary_directory), None
            config = config_for(directory)
            write_raw(directory / "raw.jsonl", [{"source_id": "1", "english": "A Dog", "tagalog": "Isang Aso"}, "{bad json", {"source_id": "2", "english": "a dog", "tagalog": "isang aso"}, {"source_id": "3", "english": "Good morning", "tagalog": "Magandang umaga."}])
            stats = prepare_dataset(config)
            self.assertEqual((stats.raw_records, stats.malformed_json, stats.duplicates, stats.accepted_records), (4, 1, 1, 2))
            train = (directory / "processed" / "train.jsonl").read_text(encoding="utf-8")
            self.assertIn("Magandang umaga.", train)
            metadata = json.loads((directory / "processed" / "preparation_metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["completed"])

    def test_splits_are_deterministic_exact_and_do_not_leak_duplicates(self) -> None:
        records = [{"source_id": str(index), "english": f"English {index}", "tagalog": f"Tagalog {index}"} for index in range(11)]
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for directory in (Path(first), Path(second)):
                write_raw(directory / "raw.jsonl", records)
                prepare_dataset(config_for(directory))
            first_splits = [(Path(first) / "processed" / name).read_text(encoding="utf-8") for name in ("train.jsonl", "validation.jsonl", "test.jsonl")]
            second_splits = [(Path(second) / "processed" / name).read_text(encoding="utf-8") for name in ("train.jsonl", "validation.jsonl", "test.jsonl")]
            self.assertEqual(first_splits, second_splits)
            self.assertEqual(sum(len(text.splitlines()) for text in first_splits), 11)
            pairs = [json.loads(line)["english"] for text in first_splits for line in text.splitlines()]
            self.assertEqual(len(pairs), len(set(pairs)))

    def test_overwrite_protection_and_exact_counts(self) -> None:
        self.assertEqual(sum(calculate_split_counts(19, {"validation_ratio": .1, "test_ratio": .1})), 19)
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory, config = Path(temporary_directory), None
            config = config_for(directory)
            write_raw(directory / "raw.jsonl", [{"english": "One", "tagalog": "Isa"}])
            prepare_dataset(config)
            with self.assertRaises(PreparationError):
                prepare_dataset(config)


if __name__ == "__main__":
    unittest.main()
