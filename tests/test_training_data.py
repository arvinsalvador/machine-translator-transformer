"""Fast local tests for teacher forcing and dynamic padding."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from src.training_data import TranslationDataset, collate_translation_batch


class FakeTokenizer:
    pad_id, bos_id, eos_id = 0, 2, 3
    def encode(self, text, add_bos=False, add_eos=False):
        ids = [10] * len(text.split())
        return ([2] if add_bos else []) + ids + ([3] if add_eos else [])


class TrainingDataTests(unittest.TestCase):
    def test_teacher_forcing_padding_and_overlength_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in [{"english": "one two", "tagalog": "isa dalawa"}, {"english": "one two three four", "tagalog": "mahaba"}]), encoding="utf-8")
            dataset = TranslationDataset(path, FakeTokenizer(), 4)
            self.assertEqual(len(dataset), 1); source, decoder, expected = dataset[0]
            self.assertEqual(source, [2, 10, 10, 3]); self.assertEqual(decoder, [2, 10, 10]); self.assertEqual(expected, [10, 10, 3])
            batch = collate_translation_batch([(source, decoder, expected), ([2, 3], [2, 3], [3, 3])], 0)
            self.assertEqual(tuple(batch[0].shape), (2, 4)); self.assertEqual(batch[0][1, -1].item(), 0)

    def test_cross_entropy_ignores_padding(self) -> None:
        logits = torch.tensor([[[100., 0.], [0., 5.]]]); targets = torch.tensor([[0, 1]])
        loss = nn.CrossEntropyLoss(ignore_index=0)(logits.reshape(-1, 2), targets.reshape(-1))
        self.assertLess(loss.item(), 0.01)


if __name__ == "__main__": unittest.main()
