"""Phase 2 Streamlit interface for safe raw dataset acquisition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

from src.acquire_dataset import AcquisitionError, AcquisitionStats, acquire_dataset, get_output_paths
from src.prepare_data import PreparationError, PreparationStats, get_paths, prepare_dataset
from src.tokenizer_utils import SentencePieceTokenizer
from src.train_tokenizer import TokenizerTrainingError, TokenizerTrainingStats, get_paths as tokenizer_paths, train_tokenizer
from src.inspect_model import synthetic_validation
from src.model import architecture_summary, build_model
from src.tokenizer_utils import load_tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config() -> dict[str, Any]:
    """Load display configuration; acquisition validates it before use."""
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            return yaml.safe_load(config_file) or {}
    except (OSError, yaml.YAMLError):
        return {}


def load_metadata(path: Path) -> dict[str, Any] | None:
    """Read prior acquisition metadata safely, if present."""
    try:
        with path.open("r", encoding="utf-8") as metadata_file:
            data = json.load(metadata_file)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def read_preview(path: Path, rows: int) -> list[dict[str, str]]:
    """Read only a few first JSONL records, never the entire dataset."""
    preview: list[dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8") as data_file:
            for _ in range(rows):
                line = data_file.readline()
                if not line:
                    break
                record = json.loads(line)
                preview.append({"English": str(record.get("english", ""))[:200], "Tagalog": str(record.get("tagalog", ""))[:200]})
    except (OSError, json.JSONDecodeError):
        return []
    return preview


def dashboard(config: dict[str, Any]) -> None:
    """Show project phase and resource settings."""
    dataset = config.get("dataset", {})
    st.title("English → Tagalog Transformer Translator")
    raw_metadata = load_metadata(PROJECT_ROOT / config.get("dataset", {}).get("acquisition", {}).get("metadata_file", ""))
    phase_two = "Completed" if raw_metadata and raw_metadata.get("completed") else "Available (not completed)"
    preprocessing = config.get("preprocessing", {})
    tokenizer_settings = config.get("tokenizer", {})
    preparation_metadata = load_metadata(PROJECT_ROOT / preprocessing.get("output_directory", "") / preprocessing.get("metadata_file", ""))
    tokenizer_metadata = load_metadata((PROJECT_ROOT / tokenizer_settings.get("model_prefix", "data/tokenizer/en_tl")).parent / tokenizer_settings.get("metadata_file", "tokenizer_metadata.json"))
    st.subheader("Phase 5 – Transformer Model")
    st.success("Phase 1 – Project Setup: Completed")
    st.write(f"Phase 2 – Dataset Acquisition: {phase_two}")
    st.write("Phase 3 – Data Preparation: " + ("Completed" if preparation_metadata and preparation_metadata.get("completed") else "Available (not completed)"))
    st.write("Phase 4 – Tokenizer Training: " + ("Completed" if tokenizer_metadata and tokenizer_metadata.get("completed") else "Available (not completed)"))
    st.info("Phase 5 – Transformer Model: Current")
    st.write("Phase 6 – Training & Evaluation through Phase 7 – Translator Interface: Not Started")
    if tokenizer_metadata and tokenizer_metadata.get("completed"):
        st.success("A tokenizer has already been trained.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset maximum", dataset.get("max_samples", "Unknown"))
    col2.metric("Quick mode samples", dataset.get("quick_samples", "Unknown"))
    col3.metric("Docker limits", "2 CPUs / 4 GB")


def preparation_page(config: dict[str, Any]) -> None:
    """Run the shared Phase 3 backend and show only bounded file previews."""
    dataset, settings = config.get("dataset", {}), config.get("preprocessing", {})
    try:
        raw_path, metadata_path, outputs = get_paths(dataset, settings)
    except (KeyError, TypeError):
        st.error("Preprocessing configuration is incomplete.")
        return
    st.title("Data Preparation")
    st.write(f"**Input raw dataset:** {raw_path.relative_to(PROJECT_ROOT)}")
    raw_metadata = load_metadata(raw_path.parent / dataset.get("acquisition", {}).get("metadata_file", "").split("/")[-1])
    if raw_metadata:
        st.metric("Raw pair count", raw_metadata.get("pairs_saved", "Unknown"))
    st.json({key: settings.get(key) for key in ("normalize_unicode", "collapse_whitespace", "remove_control_characters", "deduplicate", "min_words", "max_words", "max_characters")})
    st.write(f"Split: {dataset.get('train_ratio')} / {dataset.get('validation_ratio')} / {dataset.get('test_ratio')} · Seed: {config.get('project', {}).get('random_seed')}")
    raw_ready = raw_path.is_file() and raw_path.stat().st_size > 0 and (not raw_metadata or raw_metadata.get("completed") is True)
    if not raw_ready:
        st.warning("Complete Phase 2 dataset acquisition first.")
    existing = any(path.exists() for path in (*outputs.values(), metadata_path))
    if existing:
        st.warning("Processed dataset already exists.")
    force = st.checkbox("Replace existing processed dataset", disabled=not existing)
    if st.button("Prepare Dataset", type="primary", disabled=not raw_ready):
        progress_bar, status = st.progress(0), st.empty()

        def update_progress(stats: PreparationStats) -> None:
            progress_bar.progress(min(stats.raw_records / max(stats.raw_records + 1, 1), 0.99))
            status.write(f"Processed: {stats.raw_records} · Accepted: {stats.accepted_records} · Duplicates: {stats.duplicates}")

        try:
            stats = prepare_dataset(config, force=force, progress_callback=update_progress)
            progress_bar.progress(1.0)
            st.success("Dataset preparation complete. The project is ready for Phase 4 – Tokenizer Training.")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Raw", stats.raw_records)
            col2.metric("Clean", stats.accepted_records)
            col3.metric("Duplicates", stats.duplicates)
            col4.metric("Rejected", stats.raw_records - stats.accepted_records - stats.duplicates)
        except PreparationError as error:
            st.error(str(error))

    metadata = load_metadata(metadata_path)
    if metadata:
        st.subheader("Preparation statistics")
        st.json({key: metadata.get(key) for key in ("raw_records", "accepted_records", "malformed_json", "missing_english", "missing_tagalog", "empty_english", "empty_tagalog", "too_long", "duplicates", "train_records", "validation_records", "test_records", "processed_at")})
        st.bar_chart({"Train": [metadata.get("train_records", 0)], "Validation": [metadata.get("validation_records", 0)], "Test": [metadata.get("test_records", 0)]})
    if all(path.is_file() for path in outputs.values()):
        st.subheader("Processed split previews")
        tabs = st.tabs(("Training", "Validation", "Testing"))
        for tab, split in zip(tabs, ("train", "validation", "test")):
            with tab:
                st.dataframe(read_preview(outputs[split], int(settings.get("preview_rows", 5))), use_container_width=True, hide_index=True)


def dataset_page(config: dict[str, Any]) -> None:
    """Show acquisition controls, metadata, and a bounded preview."""
    dataset = config.get("dataset", {})
    acquisition = dataset.get("acquisition", {})
    output_path, metadata_path = get_output_paths(dataset)
    st.title("Dataset Acquisition")
    st.write(f"**Dataset Name:** {dataset.get('name', 'Unknown')}")
    st.write(f"**Split:** {dataset.get('split', 'train')}")
    st.write("**Source:** English / eng_caption")
    st.write("**Target:** Tagalog / tgl_Latn")
    col1, col2 = st.columns(2)
    col1.metric("Normal limit", dataset.get("max_samples", "Unknown"))
    col2.metric("Quick limit", dataset.get("quick_samples", "Unknown"))
    st.caption(f"Output: {output_path.relative_to(PROJECT_ROOT)}")

    metadata = load_metadata(metadata_path)
    existing = output_path.is_file() and output_path.stat().st_size > 0
    if existing:
        st.warning("An acquired dataset already exists.")
        if metadata:
            st.json({key: metadata.get(key) for key in ("mode", "requested_pairs", "pairs_saved", "records_scanned", "timestamp_utc", "completed", "stop_reason")})

    mode_label = st.selectbox("Mode", ("Quick", "Normal"), index=0)
    force = st.checkbox("Replace existing dataset", disabled=not existing)
    if st.button("Acquire Dataset", type="primary"):
        progress_bar = st.progress(0)
        status = st.empty()

        def update_progress(stats: AcquisitionStats) -> None:
            progress_bar.progress(min(stats.pairs_saved / stats.requested_pairs, 1.0))
            status.write(f"Pairs acquired: {stats.pairs_saved} / {stats.requested_pairs} · Records scanned: {stats.records_scanned}")

        try:
            stats = acquire_dataset(config, mode_label.lower(), force=force, progress_callback=update_progress)
            progress_bar.progress(1.0 if stats.completed else stats.pairs_saved / stats.requested_pairs)
            if stats.completed:
                st.success(f"Acquisition complete: {stats.pairs_saved} pairs saved.")
            else:
                st.warning(f"Acquisition stopped: {stats.stop_reason}. Saved {stats.pairs_saved} pairs.")
            st.json({"pairs_saved": stats.pairs_saved, "records_scanned": stats.records_scanned, "missing_english": stats.missing_english, "missing_tagalog": stats.missing_tagalog, "malformed_captions": stats.malformed_captions, "stop_reason": stats.stop_reason})
        except AcquisitionError as error:
            st.error(str(error))

    if existing:
        preview = read_preview(output_path, int(acquisition.get("preview_rows", 5)))
        if preview:
            st.subheader("Raw pair preview")
            st.dataframe(preview, use_container_width=True, hide_index=True)
    st.info("Data preparation and cleaning become available in Phase 3.")


def tokenizer_page(config: dict[str, Any]) -> None:
    """Train and inspect a tokenizer via the Phase 4 backend only."""
    dataset, preprocessing, settings = config.get("dataset", {}), config.get("preprocessing", {}), config.get("tokenizer", {})
    try:
        train_path, preparation_metadata_path, prefix, metadata_path = tokenizer_paths(dataset, preprocessing, settings)
    except (KeyError, TypeError):
        st.error("Tokenizer configuration is incomplete.")
        return
    model_path = prefix.with_suffix(".model")
    preparation_metadata = load_metadata(preparation_metadata_path)
    prepared = train_path.is_file() and train_path.stat().st_size > 0 and (not preparation_metadata or preparation_metadata.get("completed") is True)
    metadata = load_metadata(metadata_path)
    st.title("Tokenizer")
    st.write("**Type:** SentencePiece BPE · shared English–Tagalog vocabulary")
    st.write("**Training data:** Training split only; validation and test data are excluded.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Requested vocabulary", settings.get("vocab_size", "Unknown"))
    col2.metric("Character coverage", settings.get("character_coverage", "Unknown"))
    col3.metric("Training split", "train.jsonl")
    st.write(f"**Model path:** {model_path.relative_to(PROJECT_ROOT)}")
    st.json(settings.get("special_tokens", {}))
    if not prepared:
        st.warning("Complete Phase 3 data preparation first.")
    existing = model_path.is_file()
    if existing:
        st.warning("Tokenizer already trained.")
        if metadata:
            st.json({key: metadata.get(key) for key in ("requested_vocab_size", "actual_vocab_size", "training_records", "training_text_lines", "pad_id", "unk_id", "bos_id", "eos_id", "trained_at")})
    force = st.checkbox("Replace existing tokenizer", disabled=not existing)
    if st.button("Train Tokenizer", type="primary", disabled=not prepared):
        stage = st.status("Preparing tokenizer...", expanded=True)
        try:
            def update_stage(message: str) -> None:
                stage.write(message)
            stats = train_tokenizer(config, force=force, progress_callback=update_stage)
            stage.update(label="Tokenizer training complete", state="complete")
            st.success(f"Tokenizer trained with {stats.actual_vocab_size} pieces.")
        except TokenizerTrainingError as error:
            stage.update(label="Tokenizer training failed", state="error")
            st.error(str(error))

    if model_path.is_file():
        tokenizer = SentencePieceTokenizer(model_path)
        st.subheader("Try the Tokenizer")
        text = st.text_area("Enter English or Tagalog text", "A group of students are using computers.")
        add_bos = st.checkbox("Add BOS", value=True)
        add_eos = st.checkbox("Add EOS", value=True)
        if st.button("Tokenize"):
            ids = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
            st.write("Pieces:", tokenizer.pieces(ids))
            st.write("IDs:", ids)
            st.metric("Token count", len(ids))
            st.write("Decoded:", tokenizer.decode(ids))
            maximum = config.get("model", {}).get("max_sequence_length", 64)
            if len(ids) > maximum:
                st.warning(f"This sequence exceeds the configured model maximum of {maximum} tokens.")
        st.subheader("Vocabulary preview")
        st.dataframe([{"ID": index, "Piece": tokenizer.id_to_piece(index)} for index in range(min(20, tokenizer.get_vocab_size()))], use_container_width=True, hide_index=True)


def model_page(config: dict[str, Any]) -> None:
    """Inspect and synthetically validate the untrained Phase 5 architecture."""
    settings = config.get("tokenizer", {})
    model_prefix = PROJECT_ROOT / settings.get("model_prefix", "data/tokenizer/en_tl")
    model_path = model_prefix.with_suffix(".model")
    st.title("Transformer Model")
    st.caption("Untrained architecture – Phase 5. No optimizer, loss, backpropagation, or translation is performed.")
    if not model_path.is_file():
        st.warning("Complete Phase 4 tokenizer training first.")
        return
    tokenizer = load_tokenizer(model_path)
    model = build_model(config, tokenizer)
    summary = architecture_summary(model, config)
    cols = st.columns(4)
    cols[0].metric("Vocabulary", summary["vocab_size"])
    cols[1].metric("Model dimension", summary["d_model"])
    cols[2].metric("Attention heads", summary["attention_heads"])
    cols[3].metric("Max sequence", summary["max_sequence_length"])
    st.write(f"Encoder layers: {summary['encoder_layers']} · Decoder layers: {summary['decoder_layers']} · Feedforward: {summary['feedforward_dimension']} · Dropout: {summary['dropout']}")
    st.write("**Embedding:** one shared English–Tagalog SentencePiece embedding. **Output:** raw vocabulary logits (no softmax).")
    st.code("English tokens → Shared Embedding → Positional Encoding → Encoder × 2\n"
            "                                             ↓ memory\n"
            "Tagalog tokens → Shared Embedding → Positional Encoding → Decoder × 2 → Linear Projection → Vocabulary Logits")
    st.write("**Encoder:** processes the complete English source sequence into contextual memory.\n\n**Decoder:** uses masked self-attention and cross-attention to that memory.\n\n**Masked attention:** prevents seeing future target tokens.\n\n**Cross-attention:** uses English encoder information when predicting future Tagalog tokens.")
    metrics = st.columns(3)
    metrics[0].metric("Total parameters", f"{summary['total']:,}")
    metrics[1].metric("Trainable parameters", f"{summary['trainable']:,}")
    metrics[2].metric("Parameter memory only", f"{summary['parameter_memory_bytes'] / 1_000_000:.1f} MB")
    if st.button("Validate Model Architecture", type="primary"):
        try:
            shape = synthetic_validation(model, tokenizer)
            st.success(f"Forward pass successful: output shape {shape}. No training performed.")
        except (RuntimeError, ValueError, TypeError) as error:
            st.error(str(error))


def about_page() -> None:
    """Keep the project objective and Phase 2 scope explicit."""
    st.title("About")
    st.write("This project will implement a small custom PyTorch Transformer encoder-decoder for English → Tagalog translation.")
    st.write("Phases 2 and 3 acquire and prepare only raw text pairs. They do not download images, train a tokenizer or model, or translate text.")


def main() -> None:
    """Render the simple extensible Phase 2 navigation."""
    st.set_page_config(page_title="English–Tagalog Translator", layout="wide")
    config = load_config()
    page = st.sidebar.radio("Navigation", ("Dashboard", "Dataset", "Data Preparation", "Tokenizer", "Model", "About"))
    if page == "Dashboard":
        dashboard(config)
    elif page == "Dataset":
        dataset_page(config)
    elif page == "Data Preparation":
        preparation_page(config)
    elif page == "Tokenizer":
        tokenizer_page(config)
    elif page == "Model":
        model_page(config)
    else:
        about_page()


if __name__ == "__main__":
    main()
