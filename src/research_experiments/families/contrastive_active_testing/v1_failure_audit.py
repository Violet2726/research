"""面向冻结 CATCH-Cert v1 运行的运行后金标机制审计生成器。"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ERROR_LABELS = (
    "answer_link_missing",
    "target_candidate_missed",
    "commitment_direction_error",
    "condition_not_necessary",
    "condition_not_sufficient",
    "global_obligation_missing",
    "source_grounding_invalid",
    "verifier_semantic_error",
    "verifier_format_erasure",
    "panel_disagreement",
    "adapter_conflict",
    "decoder_only_abstention",
    "false_pass",
)


def write_v1_mechanism_audit(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    root = Path(run_dir)
    routers = _read_jsonl(root / "turns" / "router_decisions.jsonl")
    predictions = _read_jsonl(root / "views" / "predictions.jsonl")
    payload = build_v1_mechanism_audit(routers=routers, predictions=predictions, seed=seed)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "catch_cert_v1_mechanism_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (target / "catch_cert_v1_mechanism_audit.md").write_text(
        render_v1_mechanism_audit_markdown(payload),
        encoding="utf-8",
    )
    return payload


def build_v1_mechanism_audit(
    *,
    routers: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    seed: int = 42,
) -> dict[str, Any]:
    v1 = [row for row in routers if row.get("protocol_version") == "catch_cert_v1" and row.get("triggered")]
    prediction_by_sample = {
        str(row.get("sample_id")): row for row in predictions if row.get("method_name") == "catch_cert"
    }
    recoverable = [
        row for row in v1 if row.get("target_oracle_correct") and row.get("gold_candidate_key") != row.get("anchor_key")
    ]
    risk_pool = [row for row in v1 if row.get("gold_candidate_key") == row.get("anchor_key")]
    unrecoverable = [row for row in v1 if not row.get("candidate_oracle_correct")]
    risk = _matched_risk_selection(recoverable, risk_pool, count=52, seed=seed)
    selected_unrecoverable = _stable_select(unrecoverable, 16, seed=seed, role="unrecoverable")
    selected: list[tuple[str, dict[str, Any]]] = [
        *[("recoverable_wrong", row) for row in recoverable],
        *[("sc_correct_risk", row) for row in risk],
        *[("candidate_oracle_failure", row) for row in selected_unrecoverable],
    ]
    cases = [
        _audit_case(case_class, router, prediction_by_sample.get(str(router.get("sample_id"))))
        for case_class, router in selected
    ]
    automatic_counts = Counter(
        label for case in cases for label, value in dict(case["automatic_labels"]).items() if value is True
    )
    return {
        "schema_version": "catch_cert_v1_mechanism_audit_v1",
        "source_protocol": "catch_cert_v1",
        "selection_is_gold_after_run_development_only": True,
        "seed": seed,
        "requested_counts": {
            "recoverable_wrong": 52,
            "sc_correct_risk": 52,
            "candidate_oracle_failure": 16,
        },
        "actual_counts": dict(Counter(case["case_class"] for case in cases)),
        "automatic_error_counts": dict(sorted(automatic_counts.items())),
        "error_label_vocabulary": list(ERROR_LABELS),
        "annotation_protocol": {
            "reviewers_required": 2,
            "reviewer_fields": ["labels", "certificate_necessary", "certificate_sufficient", "notes"],
            "adjudication_required_on_disagreement": True,
            "automatic_labels_are_hints_not_human_ground_truth": True,
        },
        "seqbench_executor_golden_tests_passed": False,
        "cases": cases,
    }


def render_v1_mechanism_audit_markdown(payload: dict[str, Any]) -> str:
    counts = payload.get("actual_counts") or {}
    lines = [
        "# CATCH-Cert v1 机制审计队列",
        "",
        "> 本审计使用 development gold 进行事后分层，只用于机制诊断，不属于确认性结果。自动标签是待复核提示，不能替代双人标注。",
        "",
        "## 样本构成",
        "",
        f"- 可恢复错误：{counts.get('recoverable_wrong', 0)}",
        f"- SC5 正确风险样本：{counts.get('sc_correct_risk', 0)}",
        f"- candidate oracle 也失败：{counts.get('candidate_oracle_failure', 0)}",
        "",
        "## 自动预标记计数",
        "",
    ]
    for label, count in (payload.get("automatic_error_counts") or {}).items():
        lines.append(f"- `{label}`：{count}")
    lines.extend(
        [
            "",
            "## 逐题队列",
            "",
            "| 类别 | 数据集 | 样本 | 任务 | anchor | gold | resolver | 自动标签 |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for case in payload.get("cases") or []:
        labels = ", ".join(label for label, value in case["automatic_labels"].items() if value is True)
        lines.append(
            f"| {case['case_class']} | {case['dataset']} | {case['sample_id']} | {case.get('task') or '—'} | "
            f"{_table(case.get('anchor_key'))} | {_table(case.get('gold_candidate_key'))} | "
            f"{_table(case.get('resolver'))} | {_table(labels)} |"
        )
    lines.extend(
        [
            "",
            "## 标注要求",
            "",
            "每题由两名标注者独立判断必要性、充分性、commitment 方向、全局义务、source grounding、verifier 错误与最终 false-pass；分歧必须仲裁。",
            "",
        ]
    )
    return "\n".join(lines)


def _audit_case(case_class: str, router: dict[str, Any], prediction: dict[str, Any] | None) -> dict[str, Any]:
    graphs = dict(router.get("claim_graphs") or {})
    public_to_key = dict(router.get("candidate_public_to_answer_class_key") or {})
    answer_visible: dict[str, bool] = {}
    for public, key in public_to_key.items():
        text = " ".join(
            str(node.get("normalized_value") or "") for node in (graphs.get(public) or {}).get("nodes") or []
        )
        answer_visible[public] = _normalized(str(key)) in _normalized(text)
    panels = router.get("verifier_panels") or []
    panel_disagreement = False
    if len(panels) == 2:
        first = dict(panels[0].get("results") or {})
        second = dict(panels[1].get("results") or {})
        panel_disagreement = (
            any(
                first[test_id].get("observed_outcome") != second[test_id].get("observed_outcome")
                or first[test_id].get("support_status") != second[test_id].get("support_status")
                for test_id in set(first) & set(second)
            )
            or not first
            or not second
        )
    source = str(router.get("audit_source_question") or "")
    source_grounding_invalid = any(
        ref and _normalized(str(ref)) not in _normalized(source)
        for panel in panels
        for result in dict(panel.get("results") or {}).values()
        for ref in result.get("source_refs") or []
    )
    eligible = set(router.get("eligible_challengers") or [])
    gold = router.get("gold_candidate_key")
    target_keys = {
        public_to_key.get(str(public))
        for pair in router.get("public_pairs") or []
        for public in (pair.get("left_candidate"), pair.get("right_candidate"))
    }
    correct_target_missed = bool(gold and gold != router.get("anchor_key") and gold not in target_keys)
    decision = dict(router.get("decision") or {})
    override = bool(decision.get("override_accepted"))
    score = float((prediction or {}).get("score") or 0)
    global_operator = _global_operator(router)
    tests_text = " ".join(
        str(test.get("question_or_operation") or "") for test in router.get("certificate_tests") or []
    ).casefold()
    global_missing = bool(global_operator and not _operator_covered(global_operator, tests_text))
    format_erasure = any(
        not bool(panel.get("top_level_valid"))
        or int(panel.get("valid_test_count") or 0) < int(panel.get("expected_test_count") or 0)
        for panel in panels
    )
    labels = {
        "answer_link_missing": not all(answer_visible.values()) if answer_visible else True,
        "target_candidate_missed": correct_target_missed,
        "commitment_direction_error": None,
        "condition_not_necessary": None,
        "condition_not_sufficient": None,
        "global_obligation_missing": global_missing,
        "source_grounding_invalid": source_grounding_invalid,
        "verifier_semantic_error": None,
        "verifier_format_erasure": format_erasure,
        "panel_disagreement": panel_disagreement,
        "adapter_conflict": bool(router.get("adapter_conflicts")),
        "decoder_only_abstention": decision.get("resolver") == "abstention" and not panel_disagreement,
        "false_pass": override and score < 1.0,
    }
    return {
        "case_class": case_class,
        "dataset": router.get("dataset"),
        "sample_id": router.get("sample_id"),
        "task": router.get("task"),
        "anchor_key": router.get("anchor_key"),
        "gold_candidate_key": gold,
        "resolver": decision.get("resolver"),
        "candidate_answer_visible_by_public_id": answer_visible,
        "eligible_challengers": sorted(eligible),
        "gold_challenger_certificate_eligible": bool(gold and gold in eligible),
        "test_count": len(router.get("certificate_tests") or []),
        "certificate_count": len(router.get("certificates") or []),
        "automatic_labels": labels,
        "reviewer_1": _blank_annotation(),
        "reviewer_2": _blank_annotation(),
        "adjudicated": _blank_annotation(),
    }


def _matched_risk_selection(
    recoverable: list[dict[str, Any]],
    risk_pool: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_stratum: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in risk_pool:
        by_stratum[(str(row.get("dataset")), str(row.get("task")))].append(row)
    for key, values in by_stratum.items():
        by_stratum[key] = _stable_select(values, len(values), seed=seed, role=f"risk:{key}")
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for reference in recoverable:
        key = (str(reference.get("dataset")), str(reference.get("task")))
        candidates = [row for row in by_stratum.get(key, []) if str(row.get("sample_id")) not in used]
        if not candidates:
            candidates = [
                row
                for row in risk_pool
                if row.get("dataset") == reference.get("dataset") and str(row.get("sample_id")) not in used
            ]
        if not candidates:
            continue
        winner = _stable_select(candidates, 1, seed=seed, role=f"match:{reference.get('sample_id')}")[0]
        selected.append(winner)
        used.add(str(winner.get("sample_id")))
        if len(selected) == count:
            return selected
    remaining = [row for row in risk_pool if str(row.get("sample_id")) not in used]
    selected.extend(_stable_select(remaining, count - len(selected), seed=seed, role="risk-fill"))
    return selected[:count]


def _global_operator(router: dict[str, Any]) -> str | None:
    task = str(router.get("task") or "").casefold()
    question = str(router.get("audit_source_question") or "").casefold()
    if task == "dyck_languages" or "first step" in question or "first thing" in question:
        return "earliest"
    if task in {"spatial_reasoning", "object_placements"} or "final" in question:
        return "final"
    if task in {"murder_mysteries", "movie_recommendation"} or "most likely" in question:
        return "argmax"
    if task in {"word_sorting", "sarc_triples"}:
        return "complete"
    return None


def _operator_covered(operator: str, tests_text: str) -> bool:
    terms = {
        "earliest": ("earlier", "before", "prefix", "first"),
        "final": ("final", "current", "after all", "end"),
        "argmax": ("more", "most", "compare", "stronger", "primary"),
        "complete": ("all", "complete", "exact", "every"),
    }
    return any(term in tests_text for term in terms[operator])


def _blank_annotation() -> dict[str, Any]:
    return {
        "labels": [],
        "certificate_necessary": None,
        "certificate_sufficient": None,
        "notes": "",
    }


def _stable_select(rows: list[dict[str, Any]], count: int, *, seed: int, role: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}\0{role}\0{row.get('dataset')}\0{row.get('sample_id')}".encode()
        ).hexdigest(),
    )[: max(0, count)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def _table(value: Any) -> str:
    return str(value or "—").replace("|", "\\|").replace("\n", " ")
