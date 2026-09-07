"""Fast local inference safety tests without external services."""

import unittest

import torch

from src.inference import InferenceError, greedy_decode, prepare_source


class FakeTokenizer:
    bos_id, eos_id, pad_id = 2, 3, 0
    def encode(self, text, add_bos=False, add_eos=False, max_length=None):
        ids = [5] * len(text.split()); ids = ([2] if add_bos else []) + ids + ([3] if add_eos else [])
        if max_length and len(ids) > max_length: raise ValueError("too long")
        return ids
    def decode(self, ids): return "decoded"


class EosModel(torch.nn.Module):
    def forward(self, source, target):
        logits = torch.zeros(1, target.size(1), 8); logits[0, -1, 3] = 1; return logits


class InferenceTests(unittest.TestCase):
    def test_input_validation_and_boundaries(self):
        tokenizer = FakeTokenizer()
        with self.assertRaises(InferenceError): prepare_source("   ", tokenizer, 8)
        with self.assertRaises(InferenceError): prepare_source("one two three four five six seven", tokenizer, 8)
        self.assertEqual(prepare_source("hello world", tokenizer, 8), [2, 5, 5, 3])

    def test_greedy_starts_bos_and_stops_eos(self):
        ids, stopped = greedy_decode(EosModel(), FakeTokenizer(), [2, 5, 3], torch.device("cpu"), 8)
        self.assertEqual(ids, [2, 3]); self.assertEqual(stopped, "eos")


if __name__ == "__main__": unittest.main()
