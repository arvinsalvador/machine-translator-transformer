"""Reusable SentencePiece tokenizer helpers for later project phases."""

from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm


class SentencePieceTokenizer:
    """Small wrapper exposing shared-vocabulary encoding and special token IDs."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Tokenizer model was not found: {self.model_path}")
        self.processor = spm.SentencePieceProcessor(model_file=str(self.model_path))

    @property
    def pad_id(self) -> int:
        return self.processor.pad_id()

    @property
    def unk_id(self) -> int:
        return self.processor.unk_id()

    @property
    def bos_id(self) -> int:
        return self.processor.bos_id()

    @property
    def eos_id(self) -> int:
        return self.processor.eos_id()

    def get_vocab_size(self) -> int:
        """Return the actual vocabulary size produced by SentencePiece."""
        return self.processor.get_piece_size()

    def get_special_token_ids(self) -> dict[str, int]:
        """Return stable IDs used later for padding and sequence boundaries."""
        return {"pad": self.pad_id, "unk": self.unk_id, "bos": self.bos_id, "eos": self.eos_id}

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False, max_length: int | None = None, truncate: bool = False) -> list[int]:
        """Encode text, optionally adding boundaries and enforcing an explicit limit."""
        ids = list(self.processor.encode(text, out_type=int))
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        if max_length is not None and len(ids) > max_length:
            if not truncate:
                raise ValueError(f"Encoded sequence has {len(ids)} tokens, exceeding max_length={max_length}.")
            ids = ids[:max_length]
            if add_eos and ids:
                ids[-1] = self.eos_id
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode IDs while omitting boundary and padding symbols."""
        excluded = {self.pad_id, self.bos_id, self.eos_id}
        return self.processor.decode([token_id for token_id in ids if token_id not in excluded])

    def piece_to_id(self, piece: str) -> int:
        return self.processor.piece_to_id(piece)

    def id_to_piece(self, token_id: int) -> str:
        return self.processor.id_to_piece(token_id)

    def pieces(self, ids: list[int]) -> list[str]:
        return [self.id_to_piece(token_id) for token_id in ids]


def load_tokenizer(model_path: str | Path) -> SentencePieceTokenizer:
    """Load an already trained tokenizer without performing any training."""
    return SentencePieceTokenizer(model_path)


def main() -> int:
    """Provide a compact CLI tokenizer inspection command, not translation."""
    parser = argparse.ArgumentParser(description="Inspect an existing SentencePiece tokenizer.")
    parser.add_argument("--model", default="data/tokenizer/en_tl.model")
    parser.add_argument("--text", required=True)
    parser.add_argument("--no-bos", action="store_true")
    parser.add_argument("--no-eos", action="store_true")
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.model)
    ids = tokenizer.encode(args.text, add_bos=not args.no_bos, add_eos=not args.no_eos)
    print(f"Input: {args.text}")
    print(f"Pieces: {tokenizer.pieces(ids)}")
    print(f"IDs: {ids}")
    print(f"Decoded: {tokenizer.decode(ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
