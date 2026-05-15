"""报告接口、文档命令与目录治理约束测试。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from research_experiments.cli import build_parser as build_root_parser
from research_experiments.families.budget_comm.spec import build_parser as build_budget_parser
from research_experiments.families.comm_necessary.spec import build_parser as build_comm_necessary_parser
from research_experiments.families.cue.spec import build_parser as build_cue_parser
from research_experiments.families.dog_graph.spec import build_parser as build_dog_graph_parser
from research_experiments.families.free_mad_lite.spec import build_parser as build_free_mad_parser
from research_experiments.families.imad.spec import build_parser as build_imad_parser
from research_experiments.families.multi_agent.spec import build_parser as build_multi_agent_parser
from research_experiments.families.selective_comm.spec import build_parser as build_selective_parser
from research_experiments.families.sid_lite.spec import build_parser as build_sid_parser
from research_experiments.families.single_agent.spec import build_parser as build_single_agent_parser
from research_experiments.families.sparc.spec import build_parser as build_sparc_parser
from research_experiments.families.table_critic.spec import build_parser as build_table_critic_parser


ROOT = Path(__file__).resolve().parents[2]
LEGACY_REPORT_COMMANDS = (
    "report-run",
    "report-trigger",
    "report-cue",
    "report-debate-vs-vote",
)
MARKDOWN_DOCS = [
    ROOT / "README.md",
    ROOT / "docs" / "huggingface_archive_workflow.md",
    ROOT / "docs" / "run_report_pipeline.md",
    ROOT / "src" / "research_experiments" / "families" / "budget_comm" / "README.md",
    ROOT / "src" / "research_experiments" / "families" / "comm_necessary" / "README.md",
    ROOT / "src" / "research_experiments" / "families" / "cue" / "README.md",
    ROOT / "src" / "research_experiments" / "families" / "dog_graph" / "README.md",
    ROOT / "src" / "research_experiments" / "families" / "free_mad_lite" / "README.md",
    ROOT / "src" / "research_experiments" / "families" / "imad" / "README.md",
    ROOT / "src" / "research_experiments" / "families" / "multi_agent" / "README.md",
    ROOT / "src" / "research_experiments" / "families" / "selective_comm" / "README.md",
    ROOT / "src" / "research_experiments" / "families" / "sid_lite" / "README.md",
    ROOT / "src" / "research_experiments" / "families" / "single_agent" / "README.md",
    ROOT / "src" / "research_experiments" / "families" / "sparc" / "README.md",
    ROOT / "src" / "research_experiments" / "families" / "table_critic" / "README.md",
]


def test_all_family_clis_expose_render_report() -> None:
    root_parser = build_root_parser()
    assert _subcommands(root_parser) == {"family", "matrix", "tools"}

    parsers = [
        build_single_agent_parser(),
        build_multi_agent_parser(),
        build_selective_parser(),
        build_sparc_parser(),
        build_budget_parser(),
        build_sid_parser(),
        build_free_mad_parser(),
        build_dog_graph_parser(),
        build_imad_parser(),
        build_comm_necessary_parser(),
        build_cue_parser(),
        build_table_critic_parser(),
    ]
    for parser in parsers:
        assert "render-report" in _subcommands(parser)


def test_markdown_docs_do_not_reference_legacy_report_commands() -> None:
    for path in MARKDOWN_DOCS:
        text = path.read_text(encoding="utf-8")
        for legacy_command in LEGACY_REPORT_COMMANDS:
            assert legacy_command not in text, f"{path} still references legacy command {legacy_command}"


def test_markdown_docs_reference_existing_config_paths() -> None:
    pattern = re.compile(r"configs/[A-Za-z0-9_./-]+\.toml")
    for path in MARKDOWN_DOCS:
        text = path.read_text(encoding="utf-8")
        for match in pattern.findall(text):
            assert (ROOT / match).exists(), f"{path} references missing config {match}"


def test_family_readmes_use_render_report_example() -> None:
    family_readmes = [path for path in MARKDOWN_DOCS if path.parent.name not in {"docs", "research"} and path.name == "README.md"]
    for path in family_readmes:
        text = path.read_text(encoding="utf-8")
        assert "render-report --run-dir" in text, f"{path} is missing unified render-report example"


def _subcommands(parser: argparse.ArgumentParser) -> set[str]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    return set()
