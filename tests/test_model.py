"""CPU-only synthetic tests for the untrained Phase 5 Transformer."""

import unittest

import torch

from src.model import ModelConfigurationError, PositionalEncoding, TransformerTranslator, count_parameters, validate_model_config


def small_model(max_length: int = 12) -> TransformerTranslator:
    return TransformerTranslator(vocab_size=40, pad_id=0, d_model=32, attention_heads=4, encoder_layers=1, decoder_layers=1, feedforward_dimension=64, dropout=0.0, max_sequence_length=max_length)


class TransformerModelTests(unittest.TestCase):
    def test_positional_encoding_shape_and_limit(self) -> None:
        encoding = PositionalEncoding(32, 10, 0.0)
        self.assertEqual(tuple(encoding(torch.zeros(2, 10, 32)).shape), (2, 10, 32))
        with self.assertRaises(ValueError):
            encoding(torch.zeros(2, 11, 32))

    def test_invalid_divisibility_fails(self) -> None:
        with self.assertRaises(ModelConfigurationError):
            validate_model_config({"model": {"d_model": 127, "attention_heads": 4, "encoder_layers": 1, "decoder_layers": 1, "feedforward_dimension": 64, "dropout": 0.1, "max_sequence_length": 8}})

    def test_masks_block_padding_and_future_positions(self) -> None:
        model = small_model()
        tokens = torch.tensor([[2, 17, 3, 0, 0]], dtype=torch.long)
        self.assertEqual(model.key_padding_mask(tokens).tolist(), [[False, False, False, True, True]])
        mask = model.causal_mask(5, torch.device("cpu"))
        self.assertEqual(tuple(mask.shape), (5, 5))
        self.assertTrue(mask[0, 4].item())
        self.assertFalse(mask[4, 0].item())

    def test_forward_shapes_raw_logits_and_shared_embedding(self) -> None:
        model = small_model()
        source = torch.tensor([[2, 5, 3, 0, 0, 0], [2, 6, 7, 3, 0, 0]], dtype=torch.long)
        target = torch.tensor([[2, 8, 3, 0], [2, 9, 10, 3]], dtype=torch.long)
        model.eval()
        with torch.no_grad():
            output = model(source, target)
        self.assertEqual(tuple(output.shape), (2, 4, 40))
        self.assertFalse(torch.allclose(output.softmax(-1).sum(-1), output.sum(-1)))
        self.assertIs(model.token_embedding, model.token_embedding)

    def test_batch_one_max_length_and_input_validation(self) -> None:
        model = small_model(5)
        source = torch.tensor([[2, 4, 4, 3, 0]], dtype=torch.long)
        target = torch.tensor([[2, 4, 3]], dtype=torch.long)
        self.assertEqual(tuple(model(source, target).shape), (1, 3, 40))
        with self.assertRaises(ValueError):
            model(torch.ones(1, 6, dtype=torch.long), target)
        with self.assertRaises(TypeError):
            model(source.float(), target)
        with self.assertRaises(ValueError):
            model(torch.empty(1, 0, dtype=torch.long), target)
        with self.assertRaises(ValueError):
            model(torch.tensor([[40]], dtype=torch.long), target)

    def test_parameter_counts_are_positive_and_trainable(self) -> None:
        model = small_model()
        counts = count_parameters(model)
        self.assertGreater(counts["total"], 0)
        self.assertEqual(counts["total"], counts["trainable"])


if __name__ == "__main__":
    unittest.main()
