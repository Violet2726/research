"""从 CATCH-v3 原始 turn 独立重算预测与成本契约。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from research_experiments.families.contrastive_active_testing.algorithms import (
    build_stage_decision,
    decide_direct_judges,
)
from research_experiments.families.contrastive_active_testing.icv import (
    ContrastCoordinate,
    IcvWitnessParseResult,
    decode_icv,
)


def audit_v3_artifact_recomputation(
    *,
    turns: list[dict[str, Any]],
    routers: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    seed: int = 42,
) -> dict[str, Any]:
    """Recompute every v3 method outcome and cost without trusting predictions."""

    turns_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    predictions_by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in turns:
        turns_by_sample[str(row.get("sample_id") or "")].append(row)
    for row in predictions:
        predictions_by_sample[str(row.get("sample_id") or "")][str(row.get("method_name") or "")] = row
    violations: list[dict[str, Any]] = []
    audited = 0
    audited_ids: set[str] = set()
    for router in routers:
        if router.get("protocol_version") != "catch_v3":
            continue
        sample_id = str(router.get("sample_id") or "")
        sample_turns = turns_by_sample.get(sample_id, [])
        actual = predictions_by_sample.get(sample_id, {})
        stage_rows = [row for row in sample_turns if row.get("role") == "stage_a_solver"]
        resample_rows = [row for row in sample_turns if row.get("role") == "independent_resample"]
        stage = build_stage_decision(stage_rows, seed=seed, sample_id=sample_id)
        adaptive = build_stage_decision([*stage_rows, *resample_rows], seed=seed, sample_id=sample_id)
        expected: dict[str, tuple[str, list[dict[str, Any]]]] = {
            "sc_5": (stage.anchor_answer, stage_rows),
            "adaptive_sc_8": (adaptive.anchor_answer or stage.anchor_answer, [*stage_rows, *resample_rows]),
        }
        catch_rows = [
            row for row in sample_turns if row.get("role") in {"icv_selector", "icv_witness"}
        ]
        catch_answer = stage.anchor_answer
        if stage.triggered and router.get("eligible_challengers"):
            coordinates = tuple(
                _coordinate_from_dict(item)
                for item in router.get("validated_contrasts") or []
                if item.get("challenger_key") in set(router.get("eligible_challengers") or [])
            )
            panels = tuple(
                IcvWitnessParseResult(
                    bool(item.get("top_level_valid")),
                    dict(item.get("observations") or {}),
                    len(coordinates),
                    int(item.get("valid_coordinate_count") or 0),
                    int(item.get("decisive_coordinate_count") or 0),
                    (),
                )
                for item in router.get("witness_panels") or []
            )
            catch_answer = decode_icv(stage, coordinates, panels).answer
        expected["catch"] = (catch_answer, [*stage_rows, *catch_rows])
        if "direct_judge_3" in actual:
            selected = list(router.get("direct_judge_selections") or [])
            direct_answer = decide_direct_judges(stage, selected)[0] if stage.triggered else stage.anchor_answer
            direct_rows = [row for row in sample_turns if row.get("role") == "direct_judge"]
            expected["direct_judge_3"] = (direct_answer, [*stage_rows, *direct_rows])
        if "pair_judge_3" in actual:
            selected = list(router.get("pair_judge_selections") or [])
            pair_answer = decide_direct_judges(stage, selected)[0] if stage.triggered else stage.anchor_answer
            pair_rows = [row for row in sample_turns if row.get("role") == "pair_judge"]
            expected["pair_judge_3"] = (pair_answer, [*stage_rows, *pair_rows])

        for method, (answer, logical_rows) in expected.items():
            row = actual.get(method)
            if row is None:
                violations.append({"sample_id": sample_id, "method": method, "reason": "missing_prediction"})
                continue
            checks = {
                "prediction": str(row.get("prediction") or "") == str(answer or ""),
                "candidate_oracle": bool(row.get("candidate_oracle_correct"))
                == bool(router.get("candidate_oracle_correct")),
                "target_oracle": bool(row.get("target_oracle_correct"))
                == bool(router.get("target_oracle_correct")),
                "logical_calls": int(row.get("logical_calls_per_question") or 0) == len(logical_rows),
                "tokens": float(row.get("total_tokens_per_question") or 0)
                == sum(float(item.get("actual_total_tokens") or item.get("total_tokens") or 0) for item in logical_rows),
                "network_attempts": int(row.get("network_attempts_per_question") or 0)
                == sum(int(item.get("network_attempt_count") or 0) for item in logical_rows),
            }
            if method == "catch":
                checks["override"] = bool(row.get("override_accepted")) == bool(
                    (router.get("decision") or {}).get("override_accepted")
                )
                checks["resolver"] = str(row.get("resolver") or "") == str(
                    (router.get("decision") or {}).get("resolver") or ""
                )
            for field, passed in checks.items():
                if not passed:
                    violations.append(
                        {"sample_id": sample_id, "method": method, "reason": f"recomputed_{field}_mismatch"}
                    )
        audited += 1
        audited_ids.add(sample_id)
    missing_router_ids = set(predictions_by_sample) - audited_ids
    for sample_id in sorted(missing_router_ids):
        violations.append({"sample_id": sample_id, "reason": "missing_v3_router_for_prediction"})
    return {
        "passed": not violations,
        "audited_sample_count": audited,
        "violation_count": len(violations),
        "violations": violations,
    }


def _coordinate_from_dict(item: dict[str, Any]) -> ContrastCoordinate:
    return ContrastCoordinate(
        contrast_id=str(item["contrast_id"]),
        pair_id=str(item["pair_id"]),
        anchor_key=str(item["anchor_key"]),
        challenger_key=str(item["challenger_key"]),
        left_candidate_key=str(item["left_candidate_key"]),
        right_candidate_key=str(item["right_candidate_key"]),
        left_unit_ids=tuple(item.get("left_unit_ids") or ()),
        right_unit_ids=tuple(item.get("right_unit_ids") or ()),
        left_text=str(item.get("left_text") or ""),
        right_text=str(item.get("right_text") or ""),
        left_span=tuple(item.get("left_span") or (0, 0)),
        right_span=tuple(item.get("right_span") or (0, 0)),
        sha256=str(item.get("sha256") or ""),
    )
