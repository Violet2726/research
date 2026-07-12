from pathlib import Path


def test_mad_innovation_sources_are_utf8_without_mojibake() -> None:
    paths = [
        Path("docs/mad_innovation_mainline.md"),
        *Path("src/research_experiments/families/risk_controlled_trace_mad").rglob("*.py"),
        *Path("src/research_experiments/families/risk_controlled_trace_mad").rglob("*.md"),
        *Path("configs/families/risk_controlled_trace_mad").rglob("*.toml"),
        *Path("configs/families/risk_controlled_trace_mad").rglob("*.json"),
    ]
    suspicious = ("ï¿½", "锟斤拷", "Ã", "â€", "\ufffd")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in suspicious), path


def test_retired_family_directories_are_removed() -> None:
    for root in (
        Path("src/research_experiments/families/blind_reconstructive_mad"),
        Path("src/research_experiments/families/selective_gsa_mad"),
        Path("configs/families/blind_reconstructive_mad"),
        Path("configs/families/selective_gsa_mad"),
    ):
        assert not any(path for pattern in ("*.py", "*.toml", "*.md") for path in root.rglob(pattern))
