"""SGSA 与冻结 BRD 执行内核之间的共享兼容门面。"""

from research_experiments.families.blind_reconstructive_mad.config import (
    inspect_benchmarks,
    inspect_methods,
    load_control_catalog,
    load_experiment_config,
    load_protocol_config,
)
from research_experiments.families.blind_reconstructive_mad.prompts import SGSA_PROMPT_VERSION
from research_experiments.families.blind_reconstructive_mad.run.execute import run_experiment
from research_experiments.families.blind_reconstructive_mad.run.report import render_report, summarize_run
from research_experiments.families.blind_reconstructive_mad.run.validate import validate_run
from research_experiments.families.selective_gsa_mad.count100_gate import evaluate_count100_gate

__all__ = [
    "SGSA_PROMPT_VERSION",
    "evaluate_count100_gate",
    "inspect_benchmarks",
    "inspect_methods",
    "load_control_catalog",
    "load_experiment_config",
    "load_protocol_config",
    "render_report",
    "run_experiment",
    "summarize_run",
    "validate_run",
]

