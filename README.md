# English–Tagalog Transformer Translator

This university homework project will build a small English → Tagalog machine translator using a custom PyTorch Transformer encoder-decoder.

Dataset: `visheratin/laion-coco-nllb`

## Current phase

**Phase 5 – Transformer Encoder–Decoder Model**

Phases 1–4 are completed. Phase 5 implements the compact custom PyTorch Transformer encoder–decoder. It is **untrained**: no optimizer, loss calculation, backpropagation, checkpoint, BLEU, or translation is performed until Phase 6.

## Project phases

1. Phase 1 – Project Setup and Resource Configuration (Completed)
2. Phase 2 – Dataset Acquisition (Completed / available)
3. Phase 3 – Data Preparation and Dataset Splitting (Completed)
4. Phase 4 – Tokenizer Training (Completed)
5. Phase 5 – Transformer Model Implementation (Current)
6. Phase 6 – Model Training and Evaluation
7. Phase 7 – Translator Interface

## Resource limits

- Initial dataset maximum: 20,000 pairs; quick/debug dataset: 1,000 pairs
- Vocabulary: 8,000; sequence length: 64; d_model: 128
- Encoder layers: 2; decoder layers: 2; batch size: 16; maximum epochs: 5
- Docker CPU limit: approximately 2 CPUs; Docker RAM limit: approximately 4 GB

## Setup and validation

```bash
docker compose build
docker compose run --rm translator
```

The environment checker validates configuration, required directories, and resource settings.

## Dataset acquisition

Acquisition is streaming-only and stops at both a configured pair limit and a bounded scan limit. Output is written incrementally to `data/raw/en_tl_pairs.jsonl`, with metadata in `data/raw/acquisition_metadata.json`.

```bash
# Quick acquisition (1,000 pairs by default)
docker compose run --rm translator python -m src.acquire_dataset --mode quick

# Normal acquisition (up to 20,000 pairs)
docker compose run --rm translator python -m src.acquire_dataset --mode normal

# Small safe manual test
docker compose run --rm translator python -m src.acquire_dataset --mode quick --limit 100
```

An existing non-empty raw JSONL file is protected from accidental replacement. Use `--force` only when deliberately replacing it.

Start the Dataset UI with:

```bash
docker compose run --rm --service-ports translator python -m streamlit run app/app.py --server.address=0.0.0.0
```

Then open `http://localhost:8501`. The Dataset page defaults to Quick mode and previews only the first few saved records.

## Data preparation

Phase 3 reads the raw JSONL line by line, preserves linguistic casing and punctuation, applies NFC/whitespace/control-character normalization, rejects structurally invalid or oversized pairs, and removes duplicate English–Tagalog pairs before splitting. Word counts use `text.split()` as a simple safeguard; this is not tokenizer tokenization.

The exact deterministic split uses the configured random seed:

- Training: 80% — used to update Transformer weights in a later phase.
- Validation: 10% — used to monitor model development and training.
- Testing: 10% — held out for later final evaluation.

```bash
docker compose run --rm translator python -m src.prepare_data
docker compose run --rm translator python -m src.prepare_data --force
```

Successful preparation creates:

- `data/processed/train.jsonl`
- `data/processed/validation.jsonl`
- `data/processed/test.jsonl`
- `data/processed/preparation_metadata.json`

The command requires a completed Phase 2 acquisition and refuses to overwrite existing processed output unless `--force` is supplied. Use the Streamlit command above and select **Data Preparation** to run the same backend through the UI.

## Tokenizer training

Phase 4 uses one shared SentencePiece BPE vocabulary for English and Tagalog. Both languages use Latin script and share names, numbers, punctuation, and borrowed words; a shared vocabulary keeps this small academic encoder-decoder implementation simple.

The requested vocabulary target is 8,000 pieces. SentencePiece uses `hard_vocab_limit=False`, so a small debug corpus can safely produce a smaller actual vocabulary. Special-token IDs are explicitly validated: PAD `<pad>` = 0, UNK `<unk>` = 1, BOS `<s>` = 2, and EOS `</s>` = 3.

```bash
docker compose run --rm translator python -m src.train_tokenizer
docker compose run --rm translator python -m src.train_tokenizer --force

# Inspect an existing tokenizer; this performs tokenization only.
docker compose run --rm translator python -m src.tokenizer_utils --text "A student is reading a book."
```

Expected tokenizer artifacts:

- `data/tokenizer/en_tl.model`
- `data/tokenizer/en_tl.vocab`
- `data/tokenizer/tokenizer_metadata.json`

In Streamlit, select **Tokenizer** to train with deliberate overwrite protection, inspect the first 20 vocabulary pieces, and try tokenizing English or Tagalog text. No Transformer model or translation is implemented in this phase.

## Transformer model

Phase 5 builds one small batch-first Transformer encoder–decoder from the actual trained tokenizer vocabulary. It uses a shared token embedding, sinusoidal positional encoding, 2 encoder layers, 2 decoder layers, 4 attention heads, `d_model=128`, feed-forward dimension 512, dropout 0.1, source/target padding masks, and a decoder causal mask. The final linear projection returns raw logits; it deliberately does not apply softmax.

- Self-attention lets tokens within a sequence attend to one another.
- Masked self-attention prevents the decoder from seeing future target tokens.
- Cross-attention lets the decoder attend to the encoder's English memory.

Inspect the untrained model and run its tiny synthetic forward-pass check:

```bash
docker compose run --rm translator python -m src.inspect_model
docker compose run --rm translator python -m unittest discover -s tests -v
```

Use the Streamlit command above and select **Model** for an architecture diagram, parameter counts, and the same no-grad synthetic validation. Training becomes available in Phase 6.
