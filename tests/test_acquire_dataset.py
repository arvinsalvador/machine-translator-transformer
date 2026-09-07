"""Offline unit tests for the Phase 2 streaming acquisition helpers."""

import json
import tempfile
import unittest
from pathlib import Path

from src.acquire_dataset import AcquisitionError, acquire_dataset, extract_pair, extract_translation, get_requested_limit


def sample_config(directory: Path) -> dict:
    return {"dataset": {"name": "synthetic/dataset", "split": "train", "source_language": {"name": "English", "field": "eng_caption"}, "target_language": {"name": "Tagalog", "code": "tgl_Latn"}, "max_samples": 20, "quick_samples": 2, "acquisition": {"scan_multiplier": 3, "output_file": str(directory / "pairs.jsonl"), "metadata_file": str(directory / "metadata.json"), "preview_rows": 5}}}


class AcquireDatasetTests(unittest.TestCase):
    def test_extract_translation_finds_tagalog(self) -> None:
        self.assertEqual(extract_translation([["fra_Latn", "Bonjour"], ["tgl_Latn", "Kamusta"]], "tgl_Latn"), "Kamusta")

    def test_missing_or_malformed_translation_is_safe(self) -> None:
        self.assertIsNone(extract_translation([["fra_Latn", "Bonjour"]], "tgl_Latn"))
        self.assertIsNone(extract_translation(["invalid", ("tgl_Latn",)], "tgl_Latn"))

    def test_empty_english_and_tagalog_are_rejected(self) -> None:
        self.assertEqual(extract_pair({"eng_caption": " ", "captions": []}, "eng_caption", "tgl_Latn")[1], "missing_english")
        self.assertEqual(extract_pair({"eng_caption": "Hello", "captions": [["tgl_Latn", "  "]]}, "eng_caption", "tgl_Latn")[1], "missing_tagalog")

    def test_limit_cannot_exceed_configured_maximum(self) -> None:
        with self.assertRaises(AcquisitionError):
            get_requested_limit(sample_config(Path("."))["dataset"], "quick", 21)

    def test_scan_limit_and_unicode_jsonl_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            records = iter([{"id": 1, "eng_caption": "Good morning.", "captions": [["tgl_Latn", "Magandang umaga."]]}, {"id": 2, "eng_caption": "Hello.", "captions": [["tgl_Latn", "Kamusta."]]}])
            stats = acquire_dataset(sample_config(directory), dataset_loader=lambda *_: records)
            self.assertTrue(stats.completed)
            self.assertEqual(stats.scan_limit, 6)
            self.assertIn("Magandang umaga.", (directory / "pairs.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(json.loads((directory / "metadata.json").read_text(encoding="utf-8"))["pairs_saved"], 2)

    def test_existing_file_requires_force(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "pairs.jsonl").write_text('{"english":"old"}\n', encoding="utf-8")
            with self.assertRaises(AcquisitionError):
                acquire_dataset(sample_config(directory), dataset_loader=lambda *_: [])


if __name__ == "__main__":
    unittest.main()
