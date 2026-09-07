"""Local SentencePiece tests; no network or real project artifacts are used."""

import json
import tempfile
import unittest
from pathlib import Path

from src.tokenizer_utils import SentencePieceTokenizer
from src.train_tokenizer import TokenizerTrainingError, build_training_corpus, train_tokenizer


def config_for(directory: Path) -> dict:
    return {"dataset": {"acquisition": {"output_file": str(directory / "raw.jsonl")}}, "preprocessing": {"output_directory": str(directory / "processed"), "train_file": "train.jsonl", "metadata_file": "preparation_metadata.json"}, "tokenizer": {"model_type": "bpe", "vocab_size": 64, "character_coverage": 1.0, "model_prefix": str(directory / "tokenizer" / "en_tl"), "metadata_file": "tokenizer_metadata.json", "special_tokens": {"pad": "<pad>", "unknown": "<unk>", "bos": "<s>", "eos": "</s>"}}, "resources": {"max_cpu_threads": 2}}


def write_train(directory: Path) -> None:
    processed = directory / "processed"
    processed.mkdir()
    records = [{"english": "TRAIN_ONLY_PHRASE A dog is running.", "tagalog": "Ang aso ay tumatakbo."}, {"english": "A student reads a book.", "tagalog": "Ang bata ay nagbabasa ng libro."}]
    with (processed / "train.jsonl").open("w", encoding="utf-8") as train_file:
        for record in records:
            json.dump(record, train_file, ensure_ascii=False)
            train_file.write("\n")
    (processed / "validation.jsonl").write_text('{"english":"VALIDATION_SECRET_PHRASE"}\n', encoding="utf-8")
    (processed / "test.jsonl").write_text('{"english":"TEST_SECRET_PHRASE"}\n', encoding="utf-8")


class TokenizerTrainingTests(unittest.TestCase):
    def test_corpus_uses_training_split_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            write_train(directory)
            corpus = directory / "corpus.txt"
            records, lines = build_training_corpus(directory / "processed" / "train.jsonl", corpus)
            text = corpus.read_text(encoding="utf-8")
            self.assertEqual((records, lines), (2, 4))
            self.assertIn("TRAIN_ONLY_PHRASE", text)
            self.assertNotIn("VALIDATION_SECRET_PHRASE", text)
            self.assertNotIn("TEST_SECRET_PHRASE", text)

    def test_training_special_ids_encoding_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            write_train(directory)
            stats = train_tokenizer(config_for(directory))
            model, vocab = directory / "tokenizer" / "en_tl.model", directory / "tokenizer" / "en_tl.vocab"
            self.assertTrue(model.is_file())
            self.assertTrue(vocab.is_file())
            self.assertTrue(stats.completed)
            tokenizer = SentencePieceTokenizer(model)
            self.assertEqual(tokenizer.get_special_token_ids(), {"pad": 0, "unk": 1, "bos": 2, "eos": 3})
            english = tokenizer.encode("A student reads.", add_bos=True, add_eos=True)
            tagalog = tokenizer.encode("Ang bata ay nagbabasa ng libro.")
            self.assertTrue(all(isinstance(token_id, int) for token_id in english))
            self.assertEqual((english[0], english[-1]), (2, 3))
            self.assertTrue(tokenizer.decode(english))
            self.assertTrue(tokenizer.decode(tagalog))
            with self.assertRaises(ValueError):
                tokenizer.encode("many words repeated many words repeated", max_length=2)
            metadata = json.loads((directory / "tokenizer" / "tokenizer_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["training_split"], "train")
            self.assertEqual(metadata["pad_id"], 0)
            self.assertFalse((directory / "tokenizer" / "tokenizer_training_corpus.txt.tmp").exists())

    def test_overwrite_protection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            write_train(directory)
            config = config_for(directory)
            train_tokenizer(config)
            with self.assertRaises(TokenizerTrainingError):
                train_tokenizer(config)


if __name__ == "__main__":
    unittest.main()
