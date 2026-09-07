# Presentation Guide

This guide supports an 8–15 minute presentation of the English–Tagalog Transformer Machine Translator. Shorten it by using only the first paragraph or first two bullets in each section.

## 1. Introduction

> The task was to train a machine translator using Transformers. We implemented our own small English-to-Tagalog Transformer encoder–decoder in PyTorch, using parallel captions from LAION-COCO-NLLB. The goal is to demonstrate the complete learning pipeline, not to match a commercial translator.

## 2. Homework requirement

- **Homework:** Train a machine translator using Transformers.
- **Dataset:** `visheratin/laion-coco-nllb`
- **Direction:** English → Tagalog
- **Implementation:** A custom PyTorch Transformer is trained locally. Final inference does not call an online translator or a pretrained translation model.

## 3. Project objective

The project turns parallel English–Tagalog text into a locally usable translator. It covers acquisition, cleaning, splitting, subword tokenization, encoder–decoder training, held-out evaluation, and autoregressive inference.

## 4. Dataset

The English source is `eng_caption`; the Tagalog target uses the `tgl_Latn` language entry. LAION-COCO-NLLB is much larger than this assignment needs, so acquisition streams records and stops at a configured maximum of 20,000 usable pairs. This controls network transfer, disk, RAM, CPU use, and training time.

The Tagalog captions are machine-generated multilingual translations. They must not be described as manually verified human references.

Current artifacts show 20,000 raw pairs. The current prepared run read 1,000 of them, removed 5 duplicate pairs, and retained 995 clean pairs:

| Split | Purpose | Actual records | Proportion |
|---|---|---:|---:|
| Training | Update model weights | 797 | 80% |
| Validation | Monitor generalization and early stopping | 99 | 10% |
| Test | Final BLEU/chrF evaluation only | 99 | 10% |

Duplicates are removed before splitting to reduce leakage. The split is deterministic with random seed 42.

## 5. Data pipeline

```text
LAION-COCO-NLLB
       ↓
Streaming Acquisition
       ↓
English–Tagalog Pairs
       ↓
Cleaning and Deduplication
       ↓
Train / Validation / Test
       ↓
SentencePiece BPE
       ↓
Transformer Encoder–Decoder
       ↓
Training
       ↓
Evaluation
       ↓
Translator
```

Cleaning normalizes Unicode and whitespace, removes control characters, rejects unusable or oversized records, and preserves meaningful casing and punctuation.

## 6. Tokenizer

> The Transformer does not process words directly. It processes integer token IDs.

The shared SentencePiece BPE tokenizer learns reusable pieces from the **training split only**. For example, `A student is reading a book.` may be represented as pieces such as `▁A`, `▁student`, `▁is`, smaller pieces for an unfamiliar word, punctuation, and then integer IDs. The exact pieces must be shown live because they depend on the trained tokenizer.

A fixed word-level vocabulary struggles with unseen names, inflections, and spelling variations. BPE can split unfamiliar words into reusable subwords. Sharing one 8,000-piece vocabulary is reasonable here because English and Tagalog mainly use Latin script and share punctuation, numbers, names, and borrowed terms.

Special tokens:

- `<pad>` / PAD (ID 0): fills shorter sequences in a batch and is ignored by loss.
- `<unk>` / UNK (ID 1): represents text the learned pieces cannot cover.
- `<s>` / BOS (ID 2): marks the beginning of a sequence.
- `</s>` / EOS (ID 3): marks its end.

## 7. Transformer architecture

The actual model contains 2,981,952 trainable parameters and uses:

- `d_model=128`, 4 attention heads
- 2 encoder layers and 2 decoder layers
- feed-forward dimension 512
- dropout 0.1 and maximum sequence length 64
- one shared token embedding plus sinusoidal positional encoding
- a linear output projection to 8,000 raw vocabulary logits

Attention alone does not know whether a token is first, second, or third. Positional encoding adds order information to the token embeddings.

## 8. Encoder

> The encoder reads the complete English sentence and produces contextual representations. Through self-attention, every source token can consider the relevance of other source tokens.

For `The boy is playing outside.`, the representation of `playing` can attend to `boy` and `outside`. Self-attention therefore represents a token in context rather than in isolation.

## 9. Decoder

> The decoder generates the Tagalog sentence token by token. It uses masked self-attention over previously available target tokens and cross-attention over the encoder output.

- **Masked self-attention:** During training, future target tokens are hidden. When predicting `bata` in `<s> Isang bata ang nagbabasa </s>`, the decoder cannot already inspect `ang nagbabasa`.
- **Cross-attention:** The decoder attends to the English encoder memory. This connects source meaning to Tagalog generation.
- **Autoregression:** During inference, each generated token becomes input for the next step.

## 10. Training

Teacher forcing shifts the target by one position:

```text
Full target:     <BOS> Isang aso ang tumatakbo <EOS>
Decoder input:   <BOS> Isang aso ang tumatakbo
Expected output:       Isang aso ang tumatakbo <EOS>
```

The correct previous Tagalog tokens are supplied during training so the decoder learns to predict the next token. Cross-entropy loss compares the model's raw vocabulary logits with the correct next-token ID. Lower loss generally means better prediction on that dataset; it is **not** an accuracy percentage. Padding is ignored because it is only a batching aid.

```text
Prediction → Loss → Backpropagation → Gradients
           → Gradient clipping → AdamW → Updated weights
```

The process repeats for many batches. Dynamic padding pads each batch only to its longest sequence. If validation loss fails to improve for the configured patience of 2 epochs, early stopping can reduce wasted computation and overfitting.

The project retains `best_model.pt` and `last_model.pt`, rather than many checkpoints. The best model is selected by validation loss. The current normal run completed 5 epochs on CPU; its best epoch was 5 with validation loss 5.2966. Training loss fell from 7.7978 to 3.8811. Validation loss also fell during this run, so early stopping was not triggered.

## 11. Evaluation

The test set is used only after training. The saved evaluation used all 99 held-out examples:

- **BLEU:** 2.3527
- **chrF:** 19.3452

BLEU measures word/token n-gram overlap between generated and reference translations. A valid alternative wording can still receive a low BLEU score. chrF compares character n-grams and can capture partial word and morphological similarity missed by word-level overlap. Neither metric is percentage accuracy.

These low values agree with the visible limitation of repetitive predictions. They are reported honestly: the implementation pipeline works, but this limited model has not learned robust translation quality.

## 12. Translator demo

```text
English sentence
      ↓
SentencePiece IDs
      ↓
Encoder memory
      ↓
Decoder starts with BOS
      ↓
Highest-logit next token → append → repeat
      ↓
EOS or maximum length
      ↓
SentencePiece decode
      ↓
Tagalog output
```

Greedy decoding selects the highest-logit token at each step. It is simple, fast, deterministic, and appropriate for a resource-limited proof-of-concept. Beam search could potentially improve generation but is outside this assignment's scope.

## 13. Resource controls

| Control | Configured value |
|---|---:|
| Maximum streamed acquisition | 20,000 pairs |
| Quick acquisition | 1,000 pairs |
| Quick training | Up to 1,000 train records, 2 epochs |
| Batch size | 16 |
| Maximum sequence length | 64 tokens |
| Model dimension | 128 |
| Encoder / decoder layers | 2 / 2 |
| Normal training maximum | 5 epochs |
| Data-loader / CPU threads | 2 / 2 |
| Docker limits | About 2 CPUs / 4 GB RAM |
| Evaluation maximum | 2,000 samples; default 200 |

> The project was intentionally designed as a proof-of-concept rather than a large production translation model.

## 14. Limitations

1. The current prepared training subset is small.
2. Data is primarily short, caption-oriented text.
3. Tagalog references are machine-generated rather than human-verified.
4. The architecture has deliberately limited capacity.
5. Training is limited to at most five normal-profile epochs.
6. Greedy decoding considers only the best immediate token.
7. Development and the saved run are CPU-focused.
8. Results are not equivalent to commercial translation systems.

These are deliberate scope constraints. If an output repeats words or is incorrect, show it honestly and connect it to the data, capacity, training, and decoding limits.

## 15. Conclusion

> The most important result is the complete pipeline: parallel data, preprocessing, subword tokenization, an encoder–decoder Transformer, training, evaluation, and local autoregressive inference. The project demonstrates how a Transformer translator is built and learned, while keeping computation bounded and its limitations visible.

Continue with [DEMO_RUNBOOK.md](DEMO_RUNBOOK.md) for the exact live sequence and [EXPECTED_QA.md](EXPECTED_QA.md) for trainer questions.
