# Architecture Explanation

```text
English text
    │
    ▼
Shared SentencePiece BPE ──────────────┐
    │ source token IDs                 │ same vocabulary
    ▼                                  │ for target IDs
Token Embedding + Positional Encoding  │
    │                                  │
    ▼                                  │
Encoder × 2                            │
    ├── Multi-Head Self-Attention      │
    ├── Feed-Forward Network           │
    └── Residual paths + normalization │
    │                                  │
    ▼                                  │
English contextual memory              │
    │                                  │
    └──────────────────────┐           │
                           ▼           ▼
                       Decoder × 2 ◄── Previous target IDs
                           ├── Masked Self-Attention
                           ├── Cross-Attention to memory
                           ├── Feed-Forward Network
                           └── Residual paths + normalization
                           │
                           ▼
                    Linear Projection
                           │
                           ▼
                 8,000 Vocabulary Logits
                           │
                           ▼
                    Highest-logit Token
                           │
                           └── repeat until EOS or length limit
```

## Blocks

- **SentencePiece BPE:** Converts either language into reusable subword pieces and integer IDs. The shared vocabulary contains 8,000 pieces, including PAD, UNK, BOS, and EOS.
- **Token embedding:** Maps every discrete ID to a learned 128-dimensional vector. English and Tagalog use the same embedding table.
- **Positional encoding:** Adds sinusoidal position signals because attention alone does not know sequence order.
- **Encoder:** Two layers process the complete English sequence. Self-attention lets each source position use relevant context from other source positions.
- **English contextual memory:** The encoder output stores source-side representations used by every decoder layer.
- **Masked self-attention:** Lets each target position use only target tokens available so far, preventing future-token leakage.
- **Cross-attention:** Lets the decoder select relevant information from English memory while generating Tagalog.
- **Feed-forward network:** Applies a learned nonlinear transformation at every position. Its internal dimension is 512.
- **Residual paths and normalization:** Help information and gradients move through the network and stabilize optimization. These are provided by PyTorch's Transformer layers.
- **Linear projection:** Converts each 128-dimensional decoder state to 8,000 raw vocabulary logits. Cross-entropy consumes raw logits during training; greedy inference selects the highest one.
- **Stopping:** Decoding begins with BOS and stops when EOS is predicted or the 64-token model limit is reached.

## Training view

```text
Target:          <BOS> Isang aso ang tumatakbo <EOS>
Decoder input:   <BOS> Isang aso ang tumatakbo
Correct output:        Isang aso ang tumatakbo <EOS>
                              │
                              ▼
                    Cross-Entropy Loss
                              │
                              ▼
        Backpropagation → Gradient Clipping → AdamW
                              │
                              ▼
                       Updated Weights
```

Padding positions are excluded from loss. Training uses dynamic per-batch padding, while validation and inference use `model.eval()` and avoid gradient allocation.
