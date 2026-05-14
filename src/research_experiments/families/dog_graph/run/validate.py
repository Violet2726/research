"""DoG 运行产物验证。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research_experiments.families.shared.validate_common import load_jsonl, validate_shared_contracts
from research_experiments.reporting.report_views import load_json_payload


STATIC_REQUIRED_PREDICTION_FIELDS = {
    "subgraph_node_count",
    "subgraph_edge_count",
    "evidence_triples",
    "answer_path",
    "communication_grounded",
    "graph_view_kind",
}

STATIC_REQUIRED_GRAPH_TRACE_FIELDS = {
    "graph_view_kind",
    "subgraph_node_count",
    "subgraph_edge_count",
    "evidence_triples",
    "answer_path",
}

PAPER_REQUIRED_PREDICTION_FIELDS = {
    "topic_entity_id",
    "hop_index",
    "selected_relations",
    "reasoning_triples",
    "enough_answer_decision",
    "simplified_question",
    "used_direct_fallback",
    "retrieval_backend",
}

PAPER_REQUIRED_RETRIEVAL_FIELDS = {
    "hop_index",
    "head_entities",
    "selected_relations",
    "tail_entities",
    "reasoning_triples",
    "retrieval_backend",
}

PAPER_REQUIRED_RELATION_FIELDS = {
    "candidate_relations",
    "selected_relations",
    "selector_raw_text",
}

PAPER_REQUIRED_SIMPLIFICATION_FIELDS = {
    "original_question",
    "simplified_question",
    "reasoning_triples",
    "role_outputs",
}

PAPER_REQUIRED_ANSWER_ATTEMPT_FIELDS = {
    "attempt_kind",
    "decision",
    "raw_text",
    "reasoning_triples",
}


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    """检查 DoG run 是否满足最小分析契约。"""

    root = Path(run_dir)
    manifest = load_json_payload(root / "manifest.json") if (root / "manifest.json").exists() else {}
    experiment_kind = str(manifest.get("experiment_kind") or "static")
    if experiment_kind == "paper":
        return _validate_paper_run(root)
    return _validate_static_run(root)


def _validate_static_run(root: Path) -> dict[str, Any]:
    required = [
        "manifest.json",
        "agent_turns.jsonl",
        "debate_messages.jsonl",
        "graph_trace.jsonl",
        "final_predictions.jsonl",
        "metrics.json",
        "graph_diagnostics.json",
        "report.md",
        "figure_manifest.json",
        "archive_manifest.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    agent_rows = load_jsonl(root / "agent_turns.jsonl") if (root / "agent_turns.jsonl").exists() else []
    debate_rows = load_jsonl(root / "debate_messages.jsonl") if (root / "debate_messages.jsonl").exists() else []
    graph_rows = load_jsonl(root / "graph_trace.jsonl") if (root / "graph_trace.jsonl").exists() else []
    prediction_rows = load_jsonl(root / "final_predictions.jsonl") if (root / "final_predictions.jsonl").exists() else []
    request_failures = sum(1 for row in agent_rows if row.get("output_status") == "request_fail")
    schema_failures = sum(1 for row in agent_rows if row.get("output_status") == "schema_fail")
    missing_prediction_fields = sorted(
        field
        for field in STATIC_REQUIRED_PREDICTION_FIELDS
        if prediction_rows and any(field not in row for row in prediction_rows)
    )
    missing_graph_trace_fields = sorted(
        field
        for field in STATIC_REQUIRED_GRAPH_TRACE_FIELDS
        if graph_rows and any(field not in row for row in graph_rows)
    )
    ungrounded_debate_messages = [
        f"{row.get('sample_id')}#r{row.get('round_index')}"
        for row in debate_rows
        if row.get("method_mode") == "debate" and not row.get("sender_evidence_triples")
    ]
    graph_view_mismatches = [
        row.get("sample_id")
        for row in prediction_rows
        if not str(row.get("graph_view_kind") or "").strip()
    ]
    shared_contracts = validate_shared_contracts(root)
    figure_contract = shared_contracts["figure_contract"]
    archive_contract = shared_contracts["archive_contract"]
    return {
        "run_dir": str(root),
        "experiment_kind": "static",
        "passed": not missing
        and request_failures == 0
        and schema_failures == 0
        and bool(prediction_rows)
        and bool(graph_rows)
        and not missing_prediction_fields
        and not missing_graph_trace_fields
        and not ungrounded_debate_messages
        and not graph_view_mismatches
        and figure_contract["passed"]
        and archive_contract["passed"],
        "missing_files": missing,
        "request_failures": request_failures,
        "schema_failures": schema_failures,
        "prediction_rows": len(prediction_rows),
        "graph_trace_rows": len(graph_rows),
        "missing_prediction_fields": missing_prediction_fields,
        "missing_graph_trace_fields": missing_graph_trace_fields,
        "ungrounded_debate_messages": ungrounded_debate_messages[:20],
        "graph_view_mismatches": graph_view_mismatches[:20],
        "figure_contract": figure_contract,
        "archive_contract": archive_contract,
    }


def _validate_paper_run(root: Path) -> dict[str, Any]:
    required = [
        "manifest.json",
        "agent_turns.jsonl",
        "debate_messages.jsonl",
        "graph_trace.jsonl",
        "retrieval_trace.jsonl",
        "relation_selection_trace.jsonl",
        "simplification_trace.jsonl",
        "answer_attempt_trace.jsonl",
        "final_predictions.jsonl",
        "metrics.json",
        "graph_diagnostics.json",
        "report.md",
        "figure_manifest.json",
        "archive_manifest.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    agent_rows = load_jsonl(root / "agent_turns.jsonl") if (root / "agent_turns.jsonl").exists() else []
    graph_rows = load_jsonl(root / "graph_trace.jsonl") if (root / "graph_trace.jsonl").exists() else []
    retrieval_rows = load_jsonl(root / "retrieval_trace.jsonl") if (root / "retrieval_trace.jsonl").exists() else []
    relation_rows = load_jsonl(root / "relation_selection_trace.jsonl") if (root / "relation_selection_trace.jsonl").exists() else []
    simplification_rows = load_jsonl(root / "simplification_trace.jsonl") if (root / "simplification_trace.jsonl").exists() else []
    answer_attempt_rows = load_jsonl(root / "answer_attempt_trace.jsonl") if (root / "answer_attempt_trace.jsonl").exists() else []
    prediction_rows = load_jsonl(root / "final_predictions.jsonl") if (root / "final_predictions.jsonl").exists() else []
    request_failures = sum(1 for row in agent_rows if row.get("output_status") == "request_fail")
    schema_failures = sum(1 for row in agent_rows if row.get("output_status") == "schema_fail")
    missing_prediction_fields = _find_missing_fields(prediction_rows, PAPER_REQUIRED_PREDICTION_FIELDS)
    missing_retrieval_fields = _find_missing_fields(retrieval_rows, PAPER_REQUIRED_RETRIEVAL_FIELDS)
    missing_relation_fields = _find_missing_fields(relation_rows, PAPER_REQUIRED_RELATION_FIELDS)
    missing_simplification_fields = _find_missing_fields(simplification_rows, PAPER_REQUIRED_SIMPLIFICATION_FIELDS)
    missing_answer_attempt_fields = _find_missing_fields(answer_attempt_rows, PAPER_REQUIRED_ANSWER_ATTEMPT_FIELDS)
    dynamic_graph_rows = [
        row.get("sample_id")
        for row in graph_rows
        if row.get("graph_view_kind") != "dynamic_retrieval"
    ]
    shared_contracts = validate_shared_contracts(root)
    figure_contract = shared_contracts["figure_contract"]
    archive_contract = shared_contracts["archive_contract"]
    return {
        "run_dir": str(root),
        "experiment_kind": "paper",
        "passed": not missing
        and request_failures == 0
        and schema_failures == 0
        and bool(prediction_rows)
        and bool(retrieval_rows)
        and not missing_prediction_fields
        and not missing_retrieval_fields
        and not missing_relation_fields
        and not missing_simplification_fields
        and not missing_answer_attempt_fields
        and not dynamic_graph_rows
        and figure_contract["passed"]
        and archive_contract["passed"],
        "missing_files": missing,
        "request_failures": request_failures,
        "schema_failures": schema_failures,
        "prediction_rows": len(prediction_rows),
        "graph_trace_rows": len(graph_rows),
        "retrieval_trace_rows": len(retrieval_rows),
        "relation_selection_trace_rows": len(relation_rows),
        "simplification_trace_rows": len(simplification_rows),
        "answer_attempt_trace_rows": len(answer_attempt_rows),
        "missing_prediction_fields": missing_prediction_fields,
        "missing_retrieval_fields": missing_retrieval_fields,
        "missing_relation_fields": missing_relation_fields,
        "missing_simplification_fields": missing_simplification_fields,
        "missing_answer_attempt_fields": missing_answer_attempt_fields,
        "graph_view_mismatches": dynamic_graph_rows[:20],
        "figure_contract": figure_contract,
        "archive_contract": archive_contract,
    }


def _find_missing_fields(rows: list[dict[str, Any]], required_fields: set[str]) -> list[str]:
    return sorted(
        field
        for field in required_fields
        if rows and any(field not in row for row in rows)
    )
