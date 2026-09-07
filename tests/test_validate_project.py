"""Fast tests for final readiness states and lightweight artifact checks."""

import json
import tempfile
import unittest
from pathlib import Path

from src.validate_project import FAIL, PASS, WARNING, CheckResult, check_dataset, check_evaluation, overall_status, result


class ValidationTests(unittest.TestCase):
    def test_status_values_and_readiness(self):
        self.assertEqual(result("x", PASS, "ok").status, PASS)
        self.assertEqual(result("x", WARNING, "review").status, WARNING)
        self.assertEqual(result("x", FAIL, "bad").status, FAIL)
        self.assertEqual(overall_status([CheckResult("Environment", PASS, "ok")]), "READY FOR DEMONSTRATION")
        self.assertEqual(overall_status([CheckResult("Evaluation", WARNING, "missing")]), "READY WITH WARNINGS")
        self.assertEqual(overall_status([CheckResult("Tokenizer", FAIL, "missing")]), "NOT READY")

    def test_missing_and_valid_raw_dataset(self):
        config = {"dataset": {"acquisition": {"output_file": "data/raw/pairs.jsonl", "metadata_file": "data/raw/meta.json"}}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(check_dataset(config, root, True).status, FAIL)
            directory = root / "data/raw"; directory.mkdir(parents=True)
            (directory / "pairs.jsonl").write_text('{"english":"Hello","tagalog":"Kamusta"}\n', encoding="utf-8")
            (directory / "meta.json").write_text(json.dumps({"completed": True, "pairs_saved": 1}), encoding="utf-8")
            self.assertEqual(check_dataset(config, root, False).status, PASS)

    def test_missing_and_valid_evaluation(self):
        config = {"evaluation": {"results_file": "results.json"}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(check_evaluation(config, root).status, WARNING)
            (root / "results.json").write_text(json.dumps({"bleu": 1.0, "chrf": 2.0, "test_samples_evaluated": 3}), encoding="utf-8")
            self.assertEqual(check_evaluation(config, root).status, PASS)


if __name__ == "__main__": unittest.main()
