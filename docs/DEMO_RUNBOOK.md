# Demo Runbook

Use this checklist before and during the presentation. Do not retrain or reacquire data as part of the normal live demo.

## Before presentation day

- [ ] Docker Desktop is running.
- [ ] WSL is running.
- [ ] The project directory is available.
- [ ] The Docker image is built with `docker compose build`.
- [ ] `data/raw/en_tl_pairs.jsonl` and acquisition metadata exist.
- [ ] Processed train, validation, and test artifacts exist.
- [ ] `data/tokenizer/en_tl.model` exists.
- [ ] `models/checkpoints/best_model.pt` exists.
- [ ] `models/checkpoints/evaluation_results.json` exists.
- [ ] Final validation passes.
- [ ] Streamlit starts successfully.
- [ ] The browser can access `http://localhost:8501`.
- [ ] One translation has been tested.
- [ ] The CLI fallback command is copied somewhere convenient.

Run the safe validator before presenting. It loads existing artifacts and performs one bounded inference; it does not acquire data, train, or evaluate the dataset:

```bash
docker compose run --rm translator \
  python -m src.validate_project --demo
```

Expected overall result with the current artifacts: `READY FOR DEMONSTRATION`.

## Start the interface

```bash
docker compose run --rm --service-ports translator \
  python -m streamlit run app/app.py --server.address=0.0.0.0
```

Open `http://localhost:8501`. Keep that terminal running until the demo ends.

## Live demo order

### 1. Dashboard

Say: “This dashboard summarizes the seven implementation phases and shows whether their artifacts are available.” Point out dataset status, 8,000-piece vocabulary, model configuration, checkpoint profile and best epoch, and saved BLEU/chrF metrics.

### 2. Dataset

Show the dataset name, `eng_caption`, `tgl_Latn`, metadata, and preview. Say: “Acquisition is streaming and capped. We do not download the entire dataset.” The current raw artifact contains 20,000 pairs.

Do **not** click **Acquire Dataset** during the presentation.

### 3. Data Preparation

Show raw and accepted counts, duplicates, and the 797/99/99 split. Say: “Unicode and whitespace are normalized, unusable records are rejected, and duplicate pairs are removed before splitting to reduce leakage.”

Do **not** click **Prepare Dataset**.

### 4. Tokenizer

Enter `A student is reading a book.` and click **Tokenize**. Show the learned pieces, integer IDs, BOS/EOS, and decoded text. If useful, also try `Ang bata ay nagbabasa ng libro.`

Say: “The Transformer receives IDs, not raw words. A shared BPE vocabulary breaks unfamiliar words into reusable subwords for both languages.” Do **not** click **Train Tokenizer**.

### 5. Model

Point out encoder, decoder, self-attention, masked attention, cross-attention, 2,981,952 trainable parameters, and the vocabulary-logit output. Say: “This architecture exists independently from the training loop.” The synthetic validation button is safe because it performs no training.

### 6. Training

Explain the saved normal run: 797 training records, 99 validation records, batch size 16, five completed epochs, best epoch 5, and best validation loss 5.2966. Show the training and validation concepts.

Do **not** start Normal training live. If the trainer explicitly requires a live training demonstration, select **quick** only; it is capped at approximately 1,000 training records and two epochs. Confirm first that replacing checkpoints is not selected.

### 7. Evaluation

Show the saved Dashboard metrics or existing evaluation artifact: BLEU 2.3527 and chrF 19.3452 over 99 held-out samples. Explain that these are similarity metrics, not percentage accuracy. Avoid clicking **Evaluate Model** unless specifically requested because the results already exist.

### 8. Translator

Start with:

> A student is reading a book.

Then, if time permits, try one or two of these short caption-like inputs:

- `A dog is running outside.`
- `A woman is holding an umbrella.`
- `A child is playing with a ball.`
- `A group of people are walking on the beach.`

Never store or promise an expected Tagalog answer. Let the trained checkpoint produce the result live.

## Input strategy

Start with short, concrete, caption-like sentences that resemble the training domain. Avoid long paragraphs, legal text, technical essays, and rare idioms. The input is bounded to 500 characters and 64 tokenizer positions including BOS/EOS.

## Fallbacks

### Streamlit fails

Use CLI inference:

```bash
docker compose run --rm translator \
  python -m src.inference \
  --text "A student is reading a book."
```

If the Compose `translator` service is already running with a long-lived command, the equivalent is:

```bash
docker compose exec translator \
  python -m src.inference \
  --text "A student is reading a book."
```

The normal launch in this repository uses `docker compose run`, so the first fallback is the reliable default.

### Model output is poor or repetitive

Say:

> This is a small Transformer trained on a deliberately limited dataset and small number of epochs. The objective is to demonstrate the complete Transformer training pipeline rather than compete with production translation systems.

Do not hide, replace, or invent the result. The saved evaluation already documents low BLEU and repetitive outputs.

### No EOS is generated

Say: “Generation also has a hard maximum sequence length, so it terminates safely even when EOS is not produced.”

### The trainer asks for live training

Use the **quick** profile only, and only when explicitly required:

```bash
docker compose run --rm translator python -m src.train --profile quick
```

Quick mode uses up to 1,000 training records, up to 200 validation records, and two epochs. Existing checkpoints are protected unless replacement is deliberately requested. Do not change Docker resource limits live.

### Internet is unavailable

Inference does not require internet once the tokenizer and checkpoint exist. The translator runs locally. Do not attempt acquisition; use the prepared artifacts.

## Final presentation checklist

Before presenting:

- [ ] Run the final validator.
- [ ] Confirm `best_model.pt` and the tokenizer exist.
- [ ] Start Streamlit and test one translation.
- [ ] Open Dashboard, Model, and Evaluation once.
- [ ] Keep the CLI fallback command ready.
- [ ] Keep Docker Desktop running.
- [ ] Avoid running Normal training live.

During the presentation:

- [ ] Explain the problem and assigned dataset.
- [ ] Explain preprocessing and the data split.
- [ ] Explain the tokenizer and token IDs.
- [ ] Explain encoder, decoder, and attention.
- [ ] Explain teacher forcing, loss, and weight updates.
- [ ] Show training state and BLEU/chrF.
- [ ] Translate one short sentence.
- [ ] State the limitations honestly.

## End the demo

Return to the terminal running Streamlit and press `Ctrl+C`.
