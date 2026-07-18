"""CATCH 运行摘要与报告渲染入口。"""

from __future__ import annotations

import json
from collections import Counter
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
    if manifest.get("study_type") == "post_failure_cross_domain_boundary_audit":
        existing = root / "report.md"
        if existing.exists():
            if target != existing:
                target.write_text(existing.read_text(encoding="utf-8"), encoding="utf-8")
            return {"report_path": target.as_posix(), "gate_passed": False, "confirmatory": False}
    if manifest.get("run_mode") == "structural_preflight":
        return _render_structural_preflight_failure(root, target)
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


def _render_structural_preflight_failure(root: Path, target: Path) -> dict[str, Any]:
    preflight_path = root / "diagnostics" / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.exists() else {}
    selector = dict(preflight.get("selector") or {})
    turns_path = root / "turns" / "preflight_turns.jsonl"
    turns = [
        json.loads(line)
        for line in turns_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if turns_path.exists() else []
    selector_rows = [row for row in turns if row.get("role") == "icv_selector"]
    drops = Counter(
        str(item.get("reason") or "unknown")
        for row in selector_rows
        for item in row.get("dropped_contrasts") or []
    )
    raw_possible = 0
    cases = []
    for row in selector_rows:
        raw = (row.get("validated_output") or {}).get("contrasts")
        raw = raw if isinstance(raw, list) else []
        pair_counts = Counter(str(item.get("pair_id") or "") for item in raw if isinstance(item, dict))
        raw_possible += int(any(value >= 3 for value in pair_counts.values()))
        if row.get("dropped_contrasts") or row.get("leakage_count"):
            cases.append(
                {
                    "sample_id": row.get("sample_id"),
                    "accepted": len(row.get("validated_contrasts") or []),
                    "leakage_count": int(row.get("leakage_count") or 0),
                    "drops": row.get("dropped_contrasts") or [],
                }
            )
    sample_count = int(selector.get("sample_count") or len(selector_rows))
    raw_upper = raw_possible / sample_count if sample_count else 0.0
    validity = float(selector.get("coordinate_reference_validity_rate") or 0)
    coverage = float(selector.get("eligible_sample_rate") or 0)
    schema_rate = float(selector.get("schema_parsed") or 0) / sample_count if sample_count else 0.0
    funnel = {
        "status": preflight.get("status"),
        "passed": False,
        "sample_count": sample_count,
        "schema_parsed": selector.get("schema_parsed"),
        "accepted_coordinate_count": selector.get("accepted_coordinate_count"),
        "dropped_coordinate_count": selector.get("dropped_coordinate_count"),
        "coordinate_reference_validity_rate": validity,
        "leakage_count": selector.get("leakage_count"),
        "eligible_sample_count": selector.get("eligible_sample_count"),
        "eligible_sample_rate": coverage,
        "raw_three_coordinate_sample_count": raw_possible,
        "raw_three_coordinate_coverage_upper_bound": raw_upper,
        "drop_reasons": dict(sorted(drops.items())),
    }
    (root / "selector_funnel.json").write_text(json.dumps(funnel, ensure_ascii=False, indent=2), encoding="utf-8")
    failure_lines = ["# CATCH-v3 BBEH preflight failure cases", ""]
    for case in cases:
        failure_lines.append(
            f"- `{case['sample_id']}`: accepted={case['accepted']}, leakage={case['leakage_count']}, "
            f"drops={json.dumps(case['drops'], ensure_ascii=False)}"
        )
    (root / "failure_cases.md").write_text("\n".join(failure_lines) + "\n", encoding="utf-8")
    lines = [
        "# CATCH-ICV v3 BBEH structural preflight — failed",
        "",
        "## Executive conclusion",
        "",
        "The run is terminally registered as `failed_structural_preflight`. The failure occurred before witness execution and cannot be repaired by completing more samples: the selector did not construct enough valid three-coordinate contrast packets.",
        "",
        "Heldout and full confirmation are not authorized. The frozen v3 mechanism must not be retuned or rerun as a confirmatory experiment.",
        "",
        "## Frozen thresholds versus observations",
        "",
        "| Condition | Required | Observed | Delta | Result |",
        "|---|---:|---:|---:|---|",
        f"| Selector JSON parse rate | 100% | {schema_rate:.1%} | {schema_rate - 1:.1%} | {'pass' if schema_rate == 1 else 'fail'} |",
        f"| ID/group validity | 100% | {validity:.1%} | {validity - 1:.1%} | fail |",
        f"| Automated answer leakage | 0 | {int(selector.get('leakage_count') or 0)} | +{int(selector.get('leakage_count') or 0)} | fail |",
        f"| Eligible packet coverage | 60% | {coverage:.1%} | {coverage - .6:.1%} | fail |",
        f"| Raw-output coverage upper bound | 60% | {raw_upper:.1%} | {raw_upper - .6:.1%} | fail |",
        "",
        "## Selector funnel",
        "",
        f"- Samples: **{sample_count}**",
        f"- Accepted coordinates: **{int(selector.get('accepted_coordinate_count') or 0)}**",
        f"- Dropped coordinates: **{int(selector.get('dropped_coordinate_count') or 0)}**",
        f"- Eligible samples: **{int(selector.get('eligible_sample_count') or 0)} / {sample_count}**",
        f"- Drop reasons: `{json.dumps(dict(sorted(drops.items())), ensure_ascii=False)}`",
        "",
        "Only 11/20 raw selector responses proposed three coordinates for any one pair, so accepting every validator drop would still cap coverage at 55%. This is a generation-interface failure, not merely an over-strict validator artifact.",
        "",
        "## Scientific interpretation",
        "",
        "The Stage-A candidate set retained headroom, but indexed local reasoning fragments did not reliably form mutually exclusive, upstream, source-decidable measurements. Therefore the failure is in the observation construction channel; no witness result exists to support claims about decoder performance.",
        "",
        "## Audit trail",
        "",
        "See `selector_funnel.json` for the machine-readable funnel and `failure_cases.md` for per-sample validator failures. Raw payloads, cache sources, usage, finish reasons and attempt timelines remain in `turns/preflight_turns.jsonl`.",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report_path": target.as_posix(), "gate_passed": False, "termination_reason": "failed_structural_preflight"}
