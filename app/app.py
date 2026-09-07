"""Phase 1 Streamlit placeholder; no model or dataset is loaded here."""

from pathlib import Path

import streamlit as st
import yaml


def load_project_info() -> dict:
    """Read optional metadata without preventing the placeholder from loading."""
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            return (yaml.safe_load(config_file) or {}).get("project", {})
    except (OSError, yaml.YAMLError):
        return {}


def main() -> None:
    """Render the deliberately minimal Phase 1 page."""
    st.set_page_config(page_title="English–Tagalog Translator")
    st.title("English → Tagalog Transformer Translator")
    st.info("Phase 1 setup is complete. The translation model has not been trained yet.")
    project = load_project_info()
    if project:
        st.caption(f"{project.get('name', 'Translator project')} · version {project.get('version', 'unknown')}")


if __name__ == "__main__":
    main()
