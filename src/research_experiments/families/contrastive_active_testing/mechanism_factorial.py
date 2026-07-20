"""预注册的证书/验证器 2×2 机制审计模板与汇总。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CELLS = (
    ("model_certificate__model_verifier", "model", "model"),
    ("human_certificate__model_verifier", "human", "model"),
    ("model_certificate__human_or_deterministic_verifier", "model", "human_or_deterministic"),
    ("human_certificate__human_or_deterministic_verifier", "human", "human_or_deterministic"),
)


def write_factorial_template(audit_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    cases = audit.get("cases") or []
    if len(cases) != 120:
        raise ValueError("The mechanism factorial requires the frozen 120-case audit set.")
    rows = []
    for case in cases:
        for cell_id, certificate_source, verifier_source in CELLS:
            rows.append(
                {
                    "sample_id": case["sample_id"],
                    "dataset": case["dataset"],
                    "task": case.get("task"),
                    "case_class": case["case_class"],
                    "cell_id": cell_id,
                    "certificate_source": certificate_source,
                    "verifier_source": verifier_source,
                    "completed": False,
                    "initial_correct": None,
                    "final_correct": None,
                    "correct_challenger_certificate_present": None,
                    "mandatory_obligations_complete": None,
                    "commitment_direction_correct": None,
                    "verifier_outcome_correct": None,
                    "false_pass": None,
                    "false_reject": None,
                    "notes": "",
                }
            )
    payload = {
        "schema_version": "catch_cert_v2_mechanism_factorial_v1",
        "development_only": True,
        "cell_definitions": [
            {
                "cell_id": cell_id,
                "certificate_source": certificate_source,
                "verifier_source": verifier_source,
            }
            for cell_id, certificate_source, verifier_source in CELLS
        ],
        "case_count": len(cases),
        "row_count": len(rows),
        "rows": rows,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def summarize_factorial_results(results_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    payload = json.loads(Path(results_path).read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    if not rows or not all(row.get("completed") is True for row in rows):
        raise ValueError("All 480 factorial rows must be completed before mechanism attribution.")
    cells: dict[str, dict[str, Any]] = {}
    for cell_id, _, _ in CELLS:
        selected = [row for row in rows if row.get("cell_id") == cell_id]
        cells[cell_id] = {
            "sample_count": len(selected),
            "accuracy": _rate(selected, "final_correct"),
            "wrong_to_correct": sum(not row["initial_correct"] and row["final_correct"] for row in selected),
            "correct_to_wrong": sum(row["initial_correct"] and not row["final_correct"] for row in selected),
            "correct_challenger_certificate_recall": _rate(selected, "correct_challenger_certificate_present"),
            "mandatory_obligation_coverage": _rate(selected, "mandatory_obligations_complete"),
            "commitment_direction_accuracy": _rate(selected, "commitment_direction_correct"),
            "verifier_outcome_accuracy": _rate(selected, "verifier_outcome_correct"),
            "false_pass_rate": _rate(selected, "false_pass"),
            "false_reject_rate": _rate(selected, "false_reject"),
        }
    oracle = cells["human_certificate__human_or_deterministic_verifier"]["accuracy"]
    model_cert_human_verify = cells["model_certificate__human_or_deterministic_verifier"]["accuracy"]
    human_cert_model_verify = cells["human_certificate__model_verifier"]["accuracy"]
    end_to_end = cells["model_certificate__model_verifier"]["accuracy"]
    summary = {
        "schema_version": "catch_cert_v2_mechanism_factorial_summary_v1",
        "development_only": True,
        "cells": cells,
        "attribution": {
            "designer_loss": oracle - model_cert_human_verify,
            "verifier_loss": oracle - human_cert_model_verify,
            "end_to_end_loss": oracle - end_to_end,
            "interaction_or_decoder_residual": (
                oracle - end_to_end - (oracle - model_cert_human_verify) - (oracle - human_cert_model_verify)
            ),
        },
        "paper_branch": (
            "designer_bottleneck"
            if human_cert_model_verify > end_to_end
            else "homogeneous_verifier_boundary"
            if human_cert_model_verify <= end_to_end and human_cert_model_verify < oracle
            else "positive_method_candidate"
        ),
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _rate(rows: list[dict[str, Any]], field: str) -> float:
    values = [bool(row[field]) for row in rows if row.get(field) is not None]
    return sum(values) / len(values) if values else 0.0
