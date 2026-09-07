# English–Tagalog Transformer Translator

This university homework project will build a small English → Tagalog machine translator using a custom PyTorch Transformer encoder-decoder.

Dataset: `visheratin/laion-coco-nllb`

## Current phase

**Phase 1 – Project Setup and Resource Configuration**

This phase validates the container environment and configuration only. It does not download datasets or models, train a tokenizer or model, evaluate, or translate text.

## Project phases

1. Phase 1 – Project Setup and Resource Configuration
2. Phase 2 – Dataset Acquisition
3. Phase 3 – Data Preparation
4. Phase 4 – Tokenizer Training
5. Phase 5 – Transformer Model Implementation
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

The environment checker validates configuration, required directories, and resource settings. The Streamlit placeholder intentionally does not load or translate anything.
