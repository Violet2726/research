"""多智能体实验命令行入口。

CLI 负责暴露实验检查、执行、摘要、校验与 `Debate vs Vote` 报告生成，
方便把多智能体基线作为统一可复现实验流程运行。
"""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from experiment_core.foundation.workspace import (
    default_cache_root,
    default_reports_root,
    default_runs_root,
    workspace_defaults,
)
from multi_agent.config import (
    load_benchmarks,
    load_control_catalog,
    load_experiment_config,
    load_protocol_config,
    load_roster_config,
    phase_metadata,
    resolve_model,
)
from multi_agent.reporting import render_report, summarize_run
from multi_agent.runner import run_experiment
from multi_agent.validation import validate_run


def build_parser() -> argparse.ArgumentParser:
    """构建多智能体实验命令行解析器。"""
    load_dotenv(".env.local", override=False)
    parser = argparse.ArgumentParser(description="Vanilla MAD multi-agent baseline runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect-experiment", help="Show the resolved multi-agent experiment configuration.")
    inspect.add_argument("--experiment", required=True)
    inspect.add_argument("--model", default=None)

    run = subparsers.add_parser("run", help="Execute one configured multi-agent experiment phase.")
    run.add_argument("--experiment", required=True)
    run.add_argument("--phase", required=True)
    run.add_argument("--model", default=None)
    run.add_argument("--runs-root", default=default_runs_root("multi_agent"))
    run.add_argument("--cache-root", default=default_cache_root())

    summarize = subparsers.add_parser("summarize-run", help="Print a concise run summary from metrics.json.")
    summarize.add_argument("--run-dir", required=True)

    validate = subparsers.add_parser("validate-run", help="Run validation checks for one multi-agent run.")
    validate.add_argument("--run-dir", required=True)

    debate_vs_vote = subparsers.add_parser(
        "render-report",
        help="Generate paired Debate vs Vote analysis and a Chinese markdown report.",
    )
    debate_vs_vote.add_argument("--run-dir", required=True)
    debate_vs_vote.add_argument("--publish-dir", default=default_reports_root("multi_agent"))

    return parser


def main() -> None:
    """解析参数并分发到对应子命令。"""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "inspect-experiment":
        experiment = load_experiment_config(args.experiment)
        controls = load_control_catalog(experiment.control_catalog)
        benchmarks = load_benchmarks(experiment)
        payload = {
            "name": experiment.name,
            "description": experiment.description,
            "benchmark_configs": [str(path) for path in experiment.benchmark_configs],
            "benchmarks": [benchmark.slug for benchmark in benchmarks],
            "control_catalog": str(experiment.control_catalog),
            "control_methods": {name: _serialize_control(method) for name, method in sorted(controls.items())},
            "workspace_defaults": workspace_defaults("multi_agent"),
            "primary_model_ref": experiment.primary_model_ref,
            "phases": experiment.raw["phases"],
            "setups": [
                {
                    "name": setup.name,
                    "protocol": _serialize_protocol(load_protocol_config(setup.protocol)),
                    "roster": _serialize_roster(load_roster_config(setup.roster)),
                    "matched_controls": setup.matched_controls,
                    "calls_per_question": load_roster_config(setup.roster).agent_count
                    * (1 + load_protocol_config(setup.protocol).debate_rounds),
                }
                for setup in experiment.setups
            ],
        }
        payload["resolved_model"] = _serialize_model(resolve_model(args.model or experiment.primary_model_ref))
        for phase_name in experiment.raw["phases"]:
            phase = phase_metadata(experiment, phase_name)
            payload.setdefault("resolved_by_phase", {})[phase_name] = {
                "setups": phase["setups"],
                "split_suffix": phase.get("split_suffix"),
                "split_overrides": phase.get("split_overrides"),
            }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "run":
        experiment = load_experiment_config(args.experiment)
        backbone = resolve_model(args.model or experiment.primary_model_ref)
        run_dir = run_experiment(
            experiment=experiment,
            phase_name=args.phase,
            backbone=backbone,
            run_root=args.runs_root,
            cache_root=args.cache_root,
        )
        print(run_dir.as_posix())
        return

    if args.command == "summarize-run":
        print(json.dumps(summarize_run(args.run_dir), ensure_ascii=False, indent=2))
        return

    if args.command == "validate-run":
        print(json.dumps(validate_run(args.run_dir), ensure_ascii=False, indent=2))
        return

    if args.command == "render-report":
        print(json.dumps(render_report(args.run_dir, publish_dir=args.publish_dir), ensure_ascii=False, indent=2))
        return

    parser.error(f"Unsupported command: {args.command}")


def _serialize_protocol(protocol) -> dict[str, object]:
    """把协议信息转换为可 JSON 输出结构。"""
    return {
        "debate_rounds": protocol.debate_rounds,
        "initial_temperature": protocol.initial_temperature,
        "debate_temperature": protocol.debate_temperature,
        "top_p": protocol.top_p,
        "max_output_tokens": protocol.max_output_tokens,
    }


def _serialize_roster(roster) -> dict[str, object]:
    """把 roster 信息转换为可 JSON 输出结构。"""
    return {"agent_count": roster.agent_count}


def _serialize_control(method) -> dict[str, object]:
    """把控制方法转换为可 JSON 输出结构。"""
    return {
        "family": method.family,
        "budget_calls": method.budget_calls,
        "temperature": method.temperature,
        "top_p": method.top_p,
        "max_output_tokens": method.max_output_tokens,
    }


def _serialize_model(backbone) -> dict[str, object]:
    """把解析后的 backbone 模型配置转换为可 JSON 输出结构。"""
    return {
        "name": backbone.name,
        "provider": backbone.provider,
        "model_id": backbone.model_id,
        "base_url": backbone.base_url,
        "api_key_env": backbone.api_key_env,
        "chat_path": backbone.chat_path,
        "default_temperature": backbone.default_temperature,
        "default_top_p": backbone.default_top_p,
        "default_max_output_tokens": backbone.default_max_output_tokens,
        "reasoning_effort": backbone.reasoning_effort,
        "supports_response_format": backbone.supports_response_format,
        "response_format": backbone.response_format,
        "timeout_seconds": backbone.timeout_seconds,
        "max_retries": backbone.max_retries,
        "tags": backbone.tags,
    }

