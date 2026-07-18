"""CATCH 运行摘要与报告渲染入口。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    summary_path = root / "views" / "run_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    metrics_path = root / "views" / "metrics.json"
    gate_path = root / "diagnostics" / "gate.json"
    return {
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {},
        "gate": json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else {},
    }


def render_report(run_dir: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(run_dir)
    summary = summarize_run(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    method_title = str(manifest.get("paper_method_name") or "CATCH")
    target = Path(output_path) if output_path is not None else root / "report.md"
    gate = summary.get("gate") or {}
    metrics = summary.get("metrics") or {}
    lines = [
        f"# {method_title} run",
        "",
        f"Run mode: `{gate.get('run_mode') or 'full'}`",
        "",
        f"Gate passed: `{gate.get('passed')}`",
        "",
        "| Method | Micro | Task harmonic | Mean actual tokens |",
        "|---|---:|---:|---:|",
    ]
    for row in metrics.get("summary", []):
        if row.get("method_name") not in {
            "sc_5",
            "adaptive_sc_8",
            "catch",
            "direct_judge_3",
            "pair_judge_3",
        }:
            continue
        lines.append(
            f"| {row['method_name']} | {row['micro_accuracy']:.4f} | "
            f"{row['task_harmonic_accuracy']:.4f} | {row['mean_total_tokens']:.1f} |"
        )
    evidence = gate.get("evidence") or {}
    dependence = evidence.get("panel_false_pass_dependence") or {}
    observations = evidence.get("witness_position_and_agreement") or {}
    if dependence or observations:
        lines.extend(
            [
                "",
                "## Homogeneous witness diagnostics",
                "",
                f"- False-challenger panel pairs: `{dependence.get('false_challenger_panel_pair_count', 0)}`",
                f"- Panel false-pass rates: `{dependence.get('panel_1_false_pass_rate', 0):.4f}` / "
                f"`{dependence.get('panel_2_false_pass_rate', 0):.4f}`",
                f"- Joint false-pass rate: `{dependence.get('joint_false_pass_rate', 0):.4f}`",
                f"- Bernoulli correlation: `{dependence.get('bernoulli_correlation')}`",
                f"- Inverse-mapped panel agreement: "
                f"`{observations.get('inverse_mapped_panel_agreement_rate', 0):.4f}`",
                f"- LEFT_ONLY share among decisive raw verdicts: "
                f"`{observations.get('left_only_share_among_decisive', 0):.4f}`",
            ]
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report_path": target.as_posix(), "gate_passed": bool(gate.get("passed"))}
