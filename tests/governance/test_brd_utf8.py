from __future__ import annotations

from pathlib import Path


def test_brd_mainline_documents_are_utf8_and_free_of_replacement_mojibake() -> None:
    paths = [
        Path("docs/brd_mad_mainline.md"),
        Path("src/research_experiments/families/adaptive_sparse_mad/README.md"),
        *Path("src/research_experiments/families/blind_reconstructive_mad").rglob("*.py"),
        *Path("src/research_experiments/families/blind_reconstructive_mad").rglob("*.md"),
        *Path("configs/families/blind_reconstructive_mad").rglob("*.toml"),
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "\ufffd" not in text
        assert "锟" not in text
        assert text == path.read_bytes().decode("utf-8")
