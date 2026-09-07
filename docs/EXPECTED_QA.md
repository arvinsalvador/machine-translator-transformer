# Expected Trainer Q&A

The first 30 questions cover the likely presentation discussion. The final section adds implementation-level questions.

## Core questions

1. **Did you train this model yourself?**  
   Yes. The project implements a small PyTorch Transformer encoder–decoder and trains its weights on prepared English–Tagalog pairs. Inference does not use a pretrained translation model.

2. **Why did you not use all roughly 879,000 training records?**  
   This is an academic proof-of-concept with bounded CPU, RAM, storage, network use, and training time. Acquisition is capped at 20,000 usable pairs; the current prepared run used a smaller 1,000-record input.

3. **Why English to Tagalog?**  
   It provides a clear translation task using a language present in the assigned multilingual dataset and can be demonstrated locally.

4. **Where is the encoder?**  
   It is the source-side Transformer in `src/model.py`. It converts English token embeddings into contextual memory for the decoder.

5. **Where is the decoder?**  
   It is the target-side Transformer component. It receives previous target tokens and attends to encoder memory to predict the next Tagalog token.

6. **Why does the decoder need masking?**  
   During training it must not see future target tokens; otherwise the correct answer would leak into the prediction.

7. **What is cross-attention?**  
   It lets the target decoder attend to the English representations produced by the encoder.

8. **Why SentencePiece?**  
   It converts text to subword units and IDs and can represent unfamiliar words as combinations of learned pieces.

9. **Why use a shared tokenizer?**  
   Both languages mainly use Latin script and share punctuation, numbers, names, and borrowed terms. One vocabulary also keeps the small model simpler.

10. **What is teacher forcing?**  
    During training, correct previous target tokens are supplied while the model learns to predict the next target token.

11. **What is the difference between train, validation, and test?**  
    Training updates weights. Validation monitors generalization and drives checkpoint selection and early stopping. Test data is reserved for final evaluation.

12. **What loss function is used?**  
    PyTorch `CrossEntropyLoss` with the padding ID ignored.

13. **Why ignore padding?**  
    Padding only aligns variable-length examples in a batch. It is not a real translation target and should not affect loss.

14. **What optimizer is used?**  
    AdamW with a configured learning rate of 0.0003.

15. **Why use gradient clipping?**  
    It limits excessively large gradient norms and helps keep training stable. The configured norm is 1.0.

16. **What is BLEU?**  
    BLEU measures n-gram overlap between generated and reference translations.

17. **What is chrF?**  
    chrF compares character n-grams and can capture partial-word and morphological similarity.

18. **Are BLEU and chrF accuracy percentages?**  
    No. They are translation-similarity metrics, not direct percentage accuracy.

19. **Why greedy decoding?**  
    It is simple, fast, deterministic, and resource-efficient for this proof-of-concept.

20. **Could beam search improve translation?**  
    Potentially, because it retains multiple candidate sequences, but it costs more computation and is outside this limited implementation.

21. **Why is the model small?**  
    The goal is to demonstrate every Transformer stage on limited hardware. The small design preserves encoder–decoder attention while keeping training feasible.

22. **Does this use NLLB during translation?**  
    No. NLLB relates to how the source dataset's multilingual captions were produced. Final inference uses this project's custom Transformer and checkpoint.

23. **Were the Tagalog references translated by humans?**  
    They are machine-generated multilingual captions and should not be described as manually verified human translations.

24. **Why can the output be wrong or repetitive?**  
    The model is small, the current prepared data is limited and caption-like, training is short, references are noisy, and decoding is greedy. It is educational rather than production-grade.

25. **How was resource usage controlled?**  
    The project caps acquisition, sequence length, batch size, layers, epochs, worker threads, CPU, RAM, checkpoints, and evaluation samples.

26. **Does translation require internet?**  
    No. Once the tokenizer and checkpoint are available, inference runs locally.

27. **What happens if the decoder never predicts EOS?**  
    Greedy generation also stops at the model's maximum sequence length, so it always terminates.

28. **Can it translate documents?**  
    No. It is designed for one short, caption-like sentence. Document translation is outside scope.

29. **What would you improve with more resources?**  
    Possible future work includes more high-quality parallel data, more capacity and training, learning-rate scheduling, beam search, and human evaluation.

30. **What is the most important thing demonstrated?**  
    The complete pipeline: parallel data → preprocessing → subword tokenization → encoder–decoder Transformer → training → evaluation → autoregressive inference.

## Technical questions

31. **Why must `d_model` be divisible by the number of attention heads?**  
    Multi-head attention splits each token representation evenly across heads. Here 128 dimensions divide into four 32-dimensional heads.

32. **Why are logits not softmaxed before `CrossEntropyLoss`?**  
    PyTorch cross-entropy expects raw logits and internally applies a numerically stable log-softmax. Applying softmax first would be redundant and less stable.

33. **How are padding masks and causal masks different?**  
    Padding masks hide non-data PAD positions for each example. The causal mask hides future target positions for every decoder sequence.

34. **Why is test data not used for early stopping?**  
    Using it during model selection would leak test information and make the final evaluation optimistic. Validation data is used instead.

35. **Why can validation loss rise while training loss falls?**  
    The model may be fitting training-specific patterns that do not generalize. That divergence is a sign of possible overfitting.

36. **What is overfitting?**  
    It is learning the training data too specifically, including noise, while performance on unseen data stops improving or worsens.

37. **Why is positional encoding needed?**  
    Attention by itself does not encode token order. Positional encoding gives the model information about sequence positions.

38. **Why are decoder input and expected output shifted?**  
    The input ends one token earlier so each position predicts the following token, including EOS at the final position.

39. **Why call `model.eval()`?**  
    It switches modules such as dropout from training behavior to deterministic evaluation behavior.

40. **Why use `torch.no_grad()` during inference?**  
    Gradients are unnecessary when weights are not being updated. Disabling them reduces memory use and overhead.

41. **Why does dynamic padding save resources?**  
    Each batch is padded only to its longest example instead of a global fixed length, reducing unnecessary tensor positions and attention work.

42. **Why must vocabulary size match the checkpoint?**  
    Embedding and output-projection tensor shapes depend on vocabulary size. A different size makes saved weights structurally incompatible.

43. **What happens if the tokenizer and checkpoint do not match?**  
    Token IDs would mean different pieces or tensor dimensions might differ, causing invalid translations or a load failure. The validator checks compatibility.

44. **What are query, key, and value intuitively?**  
    A query represents what a token is looking for, keys describe what other tokens offer, and values carry the information combined according to attention scores.

45. **Why use a feed-forward network after attention?**  
    Attention mixes information across positions; the feed-forward block then transforms each position's features nonlinearly.

46. **What does dropout do?**  
    During training it randomly suppresses some activations, discouraging reliance on individual paths and helping regularization. It is disabled by `model.eval()`.

47. **Why keep both best and last checkpoints?**  
    The best checkpoint supports inference using the strongest validation result, while the last checkpoint preserves resumable optimizer and epoch state.

48. **What does the output projection produce?**  
    It maps each decoder hidden vector to one raw score, or logit, for every item in the 8,000-piece vocabulary.

49. **Why can exposure differ between training and inference?**  
    Teacher forcing supplies correct previous tokens in training, while inference must consume its own predictions. An early inference mistake can therefore influence later tokens.

50. **What evidence shows that the pipeline works despite low quality?**  
    Artifacts pass compatibility checks, the model forward pass has the expected shape, five training epochs reduced losses, held-out evaluation completes, and local autoregressive inference terminates safely. Low metrics describe learned quality, not a broken execution path.
