"""面向研究者的 best-effort CATCH 结果报告。"""

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
    return {
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.exists()
        else {},
        "execution": {},
    }


def render_report(run_dir: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(run_dir)
    target = Path(output_path) if output_path is not None else root / "report.md"
    summary = summarize_run(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    metrics = summary.get("metrics") or {}
    execution = summary.get("execution") or metrics.get("execution") or {}
    lines = [
        f"# {manifest.get('paper_method_name') or 'CATCH'} results",
        "",
        f"Status: `{manifest.get('run_status') or 'running'}`",
        "",
        f"Phase: `{manifest.get('phase_name') or 'unknown'}`",
        "",
        "This run uses best-effort execution. Missing requests and malformed outputs are reported instead of terminating the experiment.",
        "",
        "## Result coverage by dataset",
        "",
    ]
    screening = metrics.get("screening") or {}
    if screening:
        lines.extend(
            [
                "### Screening pools",
                "",
                "| Dataset | Status | Completed | SC5 | Candidate oracle | Target oracle | Disagreements | Invalid Stage-A |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        statuses = execution.get("dataset_statuses") or {}
        for dataset in sorted(set(screening) | set(statuses)):
            row = screening.get(dataset) or {}
            status = statuses.get(dataset) or {}
            completed = int(row.get("sample_count") or status.get("screening_sample_count") or 0)
            lines.append(
                f"| {dataset} | {status.get('status', 'completed')} | {completed} | "
                f"{float(row.get('sc5_micro_accuracy') or 0):.2%} | "
                f"{float(row.get('candidate_oracle_micro') or 0):.2%} | "
                f"{float(row.get('target_oracle_micro') or 0):.2%} | "
                f"{row.get('disagreement_count', 0)} | "
                f"{row.get('invalid_stage_answer_count', 0)} |"
            )
        lines.append("")
    datasets = metrics.get("datasets") or {}
    if not datasets:
        lines.append("No evaluable dataset result has been written yet.")
    for dataset, payload in datasets.items():
        lines.extend(
            [
                f"### {dataset}",
                "",
                f"Planned: **{payload.get('planned', 0)}**; attempted: **{payload.get('attempted', 0)}**; sample errors: **{payload.get('sample_errors', 0)}**.",
                "",
                "| Method | Evaluable | Missing | Complete-case | Missing=wrong | Corrected | Harmed |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method, row in (payload.get("methods") or {}).items():
            if method not in {"sc_5", "adaptive_sc_8", "catch", "direct_judge_3", "pair_judge_3"}:
                continue
            lines.append(
                f"| {method} | {row.get('evaluable', 0)} | {row.get('missing', 0)} | "
                f"{float(row.get('complete_case_accuracy') or 0):.2%} | "
                f"{float(row.get('conservative_accuracy_missing_as_wrong') or 0):.2%} | "
                f"{row.get('corrected', 0)} | {row.get('harmed', 0)} |"
            )
        paired = payload.get("paired_complete_cases") or {}
        if paired:
            lines.extend(["", "Paired CATCH comparisons:", ""])
            for competitor, row in paired.items():
                lines.append(
                    f"- vs `{competitor}`: n={row.get('paired_sample_count', 0)}, "
                    f"CATCH={float(row.get('catch_accuracy') or 0):.2%}, "
                    f"baseline={float(row.get('competitor_accuracy') or 0):.2%}."
                )

    failures = metrics.get("failures") or {}
    interval = failures.get("request_failure_rate_wilson_95") or [0, 0]
    lines.extend(
        [
            "",
            "## Execution failures and chance variation",
            "",
            f"- Logical calls: **{failures.get('logical_call_count', 0)}**",
            f"- Exhausted request failures: **{failures.get('request_failure_count', 0)}** "
            f"({float(failures.get('request_failure_rate') or 0):.2%}; Wilson 95% CI "
            f"{float(interval[0]):.2%} to {float(interval[1]):.2%})",
            f"- Structured/answer parse failures: **{failures.get('parse_failure_count', 0)}**",
            "",
            "| Dataset | Role | Error type | Example | Count |",
            "|---|---|---|---|---:|",
        ]
    )
    for row in failures.get("by_dataset_role_error") or []:
        example = str(row.get("example") or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {row.get('dataset')} | {row.get('role')} | {row.get('error_type')} | "
            f"{example[:140]} | {row.get('count', 0)} |"
        )

    mechanism = metrics.get("mechanism") or {}
    dependence = mechanism.get("panel_false_pass_dependence") or {}
    observations = mechanism.get("witness_position_and_agreement") or {}
    lines.extend(
        [
            "",
            "## Mechanism diagnostics",
            "",
            f"- Stage-A disagreements: `{mechanism.get('triggered_sample_count', 0)}`",
            f"- Eligible indexed packets: `{mechanism.get('eligible_sample_count', 0)}` "
            f"({float(mechanism.get('eligible_rate') or 0):.2%})",
            f"- False-challenger panel pairs: `{dependence.get('false_challenger_panel_pair_count', 0)}`",
            f"- Panel false-pass rates: `{float(dependence.get('panel_1_false_pass_rate') or 0):.4f}` / "
            f"`{float(dependence.get('panel_2_false_pass_rate') or 0):.4f}`",
            f"- Joint false-pass rate: `{float(dependence.get('joint_false_pass_rate') or 0):.4f}`",
            f"- Bernoulli correlation: `{dependence.get('bernoulli_correlation')}`",
            f"- Inverse-mapped panel agreement: "
            f"`{float(observations.get('inverse_mapped_panel_agreement_rate') or 0):.4f}`",
            f"- LEFT_ONLY share among decisive raw verdicts: "
            f"`{float(observations.get('left_only_share_among_decisive') or 0):.4f}`",
        ]
    )

    costs = metrics.get("costs") or {}
    budget = execution.get("network_attempt_budget") or {}
    warnings = list(execution.get("warnings") or manifest.get("execution_warnings") or [])
    lines.extend(
        [
            "",
            "## Cost and warnings",
            "",
            f"- Cache hits: `{costs.get('cache_hits', 0)}`",
            f"- Physical network attempts: `{costs.get('physical_network_attempts', 0)}`",
            f"- Retries: `{costs.get('retry_attempts', 0)}`",
            f"- Actual total tokens: `{costs.get('actual_total_tokens', 0)}`",
            f"- Configured attempt warning threshold: `{budget.get('configured_limit')}`; "
            f"overage: `{budget.get('overage', 0)}`",
            "",
        ]
    )
    if warnings:
        lines.extend(f"- `{warning}`" for warning in warnings)
    else:
        lines.append("- No configuration or execution warning was recorded.")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report_path": target.as_posix(), "run_status": manifest.get("run_status")}
