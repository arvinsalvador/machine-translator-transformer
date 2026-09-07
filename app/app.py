"""Phase 2 Streamlit interface for safe raw dataset acquisition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

from src.acquire_dataset import AcquisitionError, AcquisitionStats, acquire_dataset, get_output_paths


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
    st.subheader("Phase 2 – Dataset Acquisition")
    st.success("Phase 1 – Project Setup: Completed")
    st.info("Phase 2 – Dataset Acquisition: Current")
    st.write("Phase 3 – Data Preparation through Phase 7 – Translator Interface: Not Started")
    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset maximum", dataset.get("max_samples", "Unknown"))
    col2.metric("Quick mode samples", dataset.get("quick_samples", "Unknown"))
    col3.metric("Docker limits", "2 CPUs / 4 GB")


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


def about_page() -> None:
    """Keep the project objective and Phase 2 scope explicit."""
    st.title("About")
    st.write("This project will implement a small custom PyTorch Transformer encoder-decoder for English → Tagalog translation.")
    st.write("Phase 2 streams only the configured number of raw text pairs. It does not download images, train a model, or translate text.")


def main() -> None:
    """Render the simple extensible Phase 2 navigation."""
    st.set_page_config(page_title="English–Tagalog Translator", layout="wide")
    config = load_config()
    page = st.sidebar.radio("Navigation", ("Dashboard", "Dataset", "About"))
    if page == "Dashboard":
        dashboard(config)
    elif page == "Dataset":
        dataset_page(config)
    else:
        about_page()


if __name__ == "__main__":
    main()
