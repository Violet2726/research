"""执行 D4 候选盲 SourceIR v3 编译器 smoke 的受限开发工具。"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from research_experiments.core.execution.cache import CACHE_KEY_POLICY_VERSION, RequestCacheRouter
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import RequestThrottle
from research_experiments.core.execution.runner_common import execute_cached_request
from research_experiments.families.contrastive_active_testing.certificates_v2 import build_source_span_graph
from research_experiments.families.contrastive_active_testing.config import (
    load_phase_benchmarks,
    load_protocol_config,
    phase_metadata,
)
from research_experiments.families.contrastive_active_testing.d4_contract import (
    D4_MAINLINE_PROTOCOL_VERSION,
    D4_SOURCE_COMPILER_SMOKE_FAILED,
    D4_SOURCE_COMPILER_SMOKE_PASSED,
)
from research_experiments.families.contrastive_active_testing.kernel_d4 import (
    D4_IR_SCHEMA,
    D4_PROMPT_VERSION,
    answer_contract_for_sample,
    capability_registry,
    metamorphic_checks_passed,
    parse_source_ir_v3,
    route_for_sample,
    run_metamorphic_checks,
    solve_source_ir,
)
from research_experiments.families.contrastive_active_testing.kernel_prompts import (
    build_d4_source_compiler_messages,
)
from research_experiments.families.contrastive_active_testing.run.d4_ledger import D4CompletionLedger
from research_experiments.families.contrastive_active_testing.run.execute import _select_phase_samples
from research_experiments.workspace.layout import default_cache_root

SMOKE_SCHEMA = "catch_d4_source_compiler_smoke_v1"
SMOKE_SELECTION_SEED = 42
SMOKE_COMPILERS_PER_SAMPLE = 3
SMOKE_SAMPLES_PER_CAPABILITY = 5


def select_d4_compiler_smoke_samples(experiment) -> list[Any]:
    """Choose exactly five public-development rows for each frozen new capability."""

    phase = phase_metadata(experiment, "development")
    candidates: list[Any] = []
    for benchmark in load_phase_benchmarks(experiment, "development"):
        candidates.extend(_select_phase_samples(benchmark, phase, "development"))
    capabilities = tuple(capability_registry()["development_gate"]["capabilities"])
    return fixed_capability_smoke_selection(
        candidates,
        capability_ids=capabilities,
        seed=SMOKE_SELECTION_SEED,
    )


def fixed_capability_smoke_selection(
    samples: list[Any],
    *,
    capability_ids: tuple[str, ...] | list[str],
    seed: int,
) -> list[Any]:
    """Apply the fixed hash order independently within each capability."""

    groups: dict[str, list[Any]] = {str(capability): [] for capability in capability_ids}
    for sample in samples:
        capability_id = route_for_sample(sample).capability_id
        if capability_id in groups:
            groups[capability_id].append(sample)
    selected: list[Any] = []
    for capability_id in sorted(groups):
        available = sorted(
            groups[capability_id],
            key=lambda item: _smoke_selection_key(seed, capability_id, item),
        )
        if len(available) < SMOKE_SAMPLES_PER_CAPABILITY:
            raise ValueError(
                "D4 compiler smoke has insufficient public development rows for "
                f"{capability_id}: required={SMOKE_SAMPLES_PER_CAPABILITY} available={len(available)}."
            )
        selected.extend(available[:SMOKE_SAMPLES_PER_CAPABILITY])
    return sorted(
        selected,
        key=lambda item: (
            route_for_sample(item).capability_id,
            _smoke_selection_key(seed, route_for_sample(item).capability_id, item),
        ),
    )


def run_d4_source_compiler_smoke(
    *,
    experiment,
    backbone,
    run_dir: str | Path,
    cache_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the predeclared 45 x 3 source-only smoke, without Stage-A or scoring."""

    if str(experiment.raw.get("kernel_revision") or "") != "d4_proof_carrying_v1":
        raise ValueError("The source-only compiler smoke requires the D4 experiment.")
    configured_status = str(experiment.raw.get("source_compiler_smoke_status") or "")
    configured_result_path = Path(
        str(experiment.raw.get("source_compiler_smoke_result_path") or "")
    )
    configured_result_sha = str(
        experiment.raw.get("source_compiler_smoke_result_sha256") or ""
    )
    if (
        configured_status
        in {D4_SOURCE_COMPILER_SMOKE_PASSED, D4_SOURCE_COMPILER_SMOKE_FAILED}
        and configured_result_path.is_file()
        and hashlib.sha256(configured_result_path.read_bytes()).hexdigest()
        == configured_result_sha
    ):
        raise RuntimeError(
            "The preregistered D4 source-compiler smoke is terminal and cannot be rerun "
            "in another directory."
        )
    protocol = load_protocol_config(experiment.protocol)
    if (protocol.solver_max_tokens, protocol.role_max_tokens, protocol.judge_max_tokens) != (65_536, 65_536, 32_768):
        raise ValueError("The D4 compiler smoke requires the frozen 65536/65536/32768 protocol.")
    selected = select_d4_compiler_smoke_samples(experiment)
    expected_capabilities = tuple(capability_registry()["development_gate"]["capabilities"])
    _validate_smoke_selection(selected, expected_capabilities)
    selection_hash = _selection_hash(selected)
    protocol_hash = hashlib.sha256(
        json.dumps(
            {
                "protocol": asdict(protocol),
                "d4_mainline_protocol_version": D4_MAINLINE_PROTOCOL_VERSION,
                "d4_ir_schema": D4_IR_SCHEMA,
                "d4_prompt_version": D4_PROMPT_VERSION,
                "smoke_schema": SMOKE_SCHEMA,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    root = Path(run_dir).resolve()
    manifest_path = root / "manifest.json"
    turns_path = root / "turns.jsonl"
    result_path = root / "result.json"
    root.mkdir(parents=True, exist_ok=True)
    manifest = _prepare_or_validate_manifest(
        manifest_path,
        experiment=experiment,
        backbone=backbone,
        protocol=protocol,
        selected=selected,
        selection_hash=selection_hash,
        protocol_hash=protocol_hash,
        cache_policy="global_validated_response_v3",
    )
    ledger = D4CompletionLedger(root / "d4_completion_ledger.jsonl")
    load_dotenv(".env.local", override=False)
    provider = OpenAICompatibleProvider(backbone)
    router = RequestCacheRouter(cache_root or default_cache_root())
    throttle = RequestThrottle.for_model(
        backbone,
        max_concurrent_requests=experiment.max_concurrent_requests,
        requests_per_minute=experiment.requests_per_minute_limit,
    )
    try:
        jobs = [
            (sample, compiler_index)
            for sample in selected
            for compiler_index in range(1, SMOKE_COMPILERS_PER_SAMPLE + 1)
        ]
        workers = max(1, min(int(experiment.max_concurrent_requests), len(jobs)))
        rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="d4-source-smoke") as executor:
            future_to_job = {
                executor.submit(
                    _run_compiler_turn,
                    sample=sample,
                    compiler_index=compiler_index,
                    run_id=str(manifest["run_id"]),
                    backbone=backbone,
                    protocol=protocol,
                    provider=provider,
                    cache=router.for_request_target(
                        provider=backbone.provider,
                        request_model=backbone.model_id,
                        dataset=sample.dataset,
                    ),
                    throttle=throttle,
                    ledger=ledger,
                ): (sample, compiler_index)
                for sample, compiler_index in jobs
            }
            for future in as_completed(future_to_job):
                sample, compiler_index = future_to_job[future]
                try:
                    rows.append(future.result())
                except BaseException as exc:  # durable failure record; never silently retry a smoke turn
                    failure = _unexpected_failure_row(
                        sample=sample,
                        compiler_index=compiler_index,
                        run_id=str(manifest["run_id"]),
                        error=exc,
                    )
                    ledger.record(failure)
                    rows.append(failure)
        rows.sort(key=lambda row: (str(row["capability_id"]), str(row["dataset"]), str(row["sample_id"]), int(row["agent_id"])))
        turns_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        result = assess_d4_source_compiler_smoke(rows, expected_capabilities=expected_capabilities)
        result.update(
            {
                "schema": SMOKE_SCHEMA,
                "d4_mainline_protocol_version": D4_MAINLINE_PROTOCOL_VERSION,
                "run_id": manifest["run_id"],
                "run_dir": root.as_posix(),
                "selection_sha256": selection_hash,
                "protocol_sha256": protocol_hash,
                "cache_policy": "global_validated_response_v3",
                "cache_key_policy": CACHE_KEY_POLICY_VERSION,
                "stage_a_executed": False,
                "gold_accuracy_computed": False,
            }
        )
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest.update(
            {
                "run_status": "passed" if result["passed"] else "failed",
                "completed_at": datetime.now(UTC).isoformat(),
                "result_path": result_path.relative_to(root).as_posix(),
                "turns_path": turns_path.relative_to(root).as_posix(),
            }
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    finally:
        router.close()
        provider.close()


def assess_d4_source_compiler_smoke(
    rows: list[dict[str, Any]],
    *,
    expected_capabilities: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Apply the preregistered smoke stopping rules without using gold labels."""

    expected = {str(item) for item in expected_capabilities}
    by_sample: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_sample.setdefault(
            (str(row.get("capability_id") or ""), str(row.get("dataset") or ""), str(row.get("sample_id") or "")),
            [],
        ).append(row)
    successful_capabilities: set[str] = set()
    for (capability, _dataset, _sample_id), group in by_sample.items():
        if (
            len(group) == SMOKE_COMPILERS_PER_SAMPLE
            and all(row.get("source_ir_v3_status") == "ok" and row.get("verification_passed") is True for row in group)
            and len({str(row.get("canonical_answer") or "") for row in group}) == 1
        ):
            successful_capabilities.add(capability)
    source_ir_pass_count = sum(row.get("source_ir_v3_status") == "ok" for row in rows)
    false_pass_count = sum(
        row.get("verification_passed") is True
        and (
            not str(row.get("reference_checker_status") or "").startswith("PASSED_")
            or str(dict(row.get("concrete_witness_status") or {}).get("status") or "") != "PASSED"
            or not bool(row.get("metamorphic_checks_passed"))
        )
        for row in rows
    )
    leakage_count = sum(bool(row.get("candidate_or_gold_leakage")) for row in rows)
    conditions = {
        "exact_135_compiler_turns": len(rows) == 135,
        "no_candidate_or_gold_leakage": leakage_count == 0,
        "at_least_122_source_ir_v3_passes": source_ir_pass_count >= 122,
        "every_capability_has_one_three_way_verified_agreement": successful_capabilities == expected,
        "no_reference_or_metamorphic_false_pass": false_pass_count == 0,
    }
    return {
        "conditions": conditions,
        "passed": all(conditions.values()),
        "summary": {
            "compiler_turn_count": len(rows),
            "source_ir_v3_pass_count": source_ir_pass_count,
            "candidate_or_gold_leakage_count": leakage_count,
            "reference_or_metamorphic_false_pass_count": false_pass_count,
            "capabilities_with_three_way_verified_agreement": sorted(successful_capabilities),
            "missing_capabilities": sorted(expected - successful_capabilities),
        },
    }


def _run_compiler_turn(
    *,
    sample,
    compiler_index: int,
    run_id: str,
    backbone,
    protocol,
    provider,
    cache,
    throttle,
    ledger: D4CompletionLedger,
) -> dict[str, Any]:
    method_name = "catch_kernel_d4_source_compiler_smoke"
    role = "d4_source_compiler"
    seed = 49_000 + compiler_index
    prior = ledger.lookup(
        sample_id=sample.sample_id,
        method_name=method_name,
        role=role,
        agent_id=compiler_index,
        seed=seed,
    )
    if prior is not None:
        return prior
    decision = route_for_sample(sample)
    graph = build_source_span_graph(sample)
    messages = build_d4_source_compiler_messages(
        sample,
        source_spans=[{"span_id": span.span_id, "text": span.text} for span in graph.spans],
        answer_contract=answer_contract_for_sample(sample),
        decision=decision,
    )
    request = execute_cached_request(
        backbone=backbone,
        provider=provider,
        cache=cache,
        throttle=throttle,
        messages=messages,
        temperature=0.7,
        top_p=1.0,
        seed=seed,
        use_response_format=True,
        max_tokens=protocol.role_max_tokens,
        response_validator=lambda response: _admit_compiler_response(
            response,
            sample=sample,
            decision=decision,
        ),
    )
    payload: dict[str, Any] | None = None
    json_error: str | None = request.request_error
    if json_error is None:
        try:
            candidate = json.loads(str(request.response_payload.get("assistant_text") or ""))
            if not isinstance(candidate, dict):
                raise ValueError("JSON output must be an object")
            payload = candidate
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            json_error = str(exc)
    ir = None
    parse_reason = json_error or "not_attempted"
    if payload is not None:
        ir, parse_reason = parse_source_ir_v3(payload, sample=sample, decision=decision)
    solver = solve_source_ir(sample, decision, ir) if ir is not None else None
    metamorphic = run_metamorphic_checks(ir, solver, sample=sample, decision=decision) if ir is not None and solver is not None else {}
    verified = bool(
        solver is not None
        and solver.status == "UNIQUE"
        and bool(solver.canonical_answer)
        and str(solver.reference_checker_status).startswith("PASSED_")
        and solver.concrete_witness_status.get("status") == "PASSED"
        and metamorphic_checks_passed(ir, solver, metamorphic)
    )
    usage = dict(request.usage)
    row = {
        "run_id": run_id,
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "capability_id": decision.capability_id,
        "kernel_id": decision.kernel_id,
        "query_operator": decision.query_operator,
        "method_name": method_name,
        "role": role,
        "agent_id": compiler_index,
        "request_seed": seed,
        "cache_key": request.cache_key,
        "cache_hit": request.cache_hit,
        "cache_origin_completion_cap": request.response_payload.get("cache_origin_completion_cap"),
        "request_completion_cap": protocol.role_max_tokens,
        "cache_key_policy": request.response_payload.get("cache_origin_key_policy"),
        "request_error": request.request_error,
        "raw_finish_reason": request.response_payload.get("finish_reason"),
        "usage_reported": dict(request.response_payload.get("usage_reported") or {}),
        "prompt_tokens": float(usage.get("prompt_tokens") or 0),
        "completion_tokens": float(usage.get("completion_tokens") or 0),
        "assistant_text": str(request.response_payload.get("assistant_text") or ""),
        "source_ir_v3_status": "ok" if ir is not None else "failed",
        "source_ir_v3_parse_reason": parse_reason,
        "source_ir_hash": ir.canonical_ir_hash if ir is not None else None,
        "solver_status": solver.status if solver is not None else "NOT_RUN",
        "solver_reason": solver.reason if solver is not None else "NOT_RUN",
        "canonical_answer": solver.canonical_answer if solver is not None else None,
        "solver_trace": list(solver.solver_trace) if solver is not None else [],
        "reference_checker_status": solver.reference_checker_status if solver is not None else "NOT_RUN",
        "concrete_witness_status": dict(solver.concrete_witness_status) if solver is not None else {"status": "NOT_RUN"},
        "metamorphic_status": metamorphic,
        "metamorphic_checks_passed": bool(
            ir is not None and solver is not None and metamorphic_checks_passed(ir, solver, metamorphic)
        ),
        "verification_passed": verified,
        "candidate_or_gold_leakage": bool(
            ir is None and str(parse_reason or "").startswith("source_ir_v3_candidate_leakage:")
        ),
        "compiler_input_fields": ["source", "answer_contract", "source_spans", "capability_id", "query_operator"],
    }
    if ir is None or not verified:
        # Malformed or locally unverifiable compiler output is a D4 protocol
        # failure and must not remain in the shared request cache.
        cache.delete(request.cache_key)
    ledger.record(row)
    return row


def _admit_compiler_response(response: dict[str, Any], *, sample, decision) -> dict[str, Any]:
    candidate = json.loads(str(response.get("assistant_text") or ""))
    if not isinstance(candidate, dict):
        raise ValueError("Compiler output must be a JSON object.")
    ir, reason = parse_source_ir_v3(candidate, sample=sample, decision=decision)
    if ir is None:
        raise ValueError(reason)
    solver = solve_source_ir(sample, decision, ir)
    metamorphic = run_metamorphic_checks(ir, solver, sample=sample, decision=decision)
    if not (
        solver.status == "UNIQUE"
        and bool(solver.canonical_answer)
        and str(solver.reference_checker_status).startswith("PASSED_")
        and solver.concrete_witness_status.get("status") == "PASSED"
        and metamorphic_checks_passed(ir, solver, metamorphic)
    ):
        raise ValueError("Compiler proof chain did not pass all local verification gates.")
    return candidate


def _prepare_or_validate_manifest(
    path: Path,
    *,
    experiment,
    backbone,
    protocol,
    selected: list[Any],
    selection_hash: str,
    protocol_hash: str,
    cache_policy: str,
) -> dict[str, Any]:
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema": SMOKE_SCHEMA,
            "selection_sha256": selection_hash,
            "protocol_sha256": protocol_hash,
            "cache_policy": cache_policy,
            "component_sha256": {
                "d4_compiler_smoke.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
        }
        if any(previous.get(key) != value for key, value in expected.items()):
            raise ValueError("D4 compiler smoke resume manifest does not match the frozen selection and protocol.")
        if previous.get("run_status") in {"passed", "failed"}:
            raise ValueError("A terminal D4 compiler smoke cannot be rerun or extended.")
        return previous
    manifest = {
        "schema": SMOKE_SCHEMA,
        "run_id": f"d4-source-smoke-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        "created_at": datetime.now(UTC).isoformat(),
        "experiment_name": experiment.name,
        "resolved_model": asdict(backbone),
        "protocol": asdict(protocol),
        "d4_mainline_protocol_version": D4_MAINLINE_PROTOCOL_VERSION,
        "protocol_sha256": protocol_hash,
        "component_sha256": {
            "d4_compiler_smoke.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "selection_sha256": selection_hash,
        "selection_seed": SMOKE_SELECTION_SEED,
        "selected_samples": [
            {
                "capability_id": route_for_sample(sample).capability_id,
                "dataset": sample.dataset,
                "sample_id": sample.sample_id,
            }
            for sample in selected
        ],
        "cache_policy": cache_policy,
        "cache_key_policy": CACHE_KEY_POLICY_VERSION,
        "stage_a_executed": False,
        "gold_accuracy_computed": False,
        "run_status": "running",
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _validate_smoke_selection(selected: list[Any], expected_capabilities: tuple[str, ...]) -> None:
    counts: dict[str, int] = {}
    for sample in selected:
        capability = route_for_sample(sample).capability_id
        counts[capability] = counts.get(capability, 0) + 1
    expected = {capability: SMOKE_SAMPLES_PER_CAPABILITY for capability in expected_capabilities}
    if counts != expected or len(selected) != 45:
        raise ValueError(f"D4 compiler smoke selection must be exactly 5 x 9 rows, got {counts}.")


def _selection_hash(selected: list[Any]) -> str:
    payload = [
        {
            "capability_id": route_for_sample(sample).capability_id,
            "dataset": sample.dataset,
            "sample_id": sample.sample_id,
        }
        for sample in selected
    ]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _smoke_selection_key(seed: int, capability_id: str, sample: Any) -> str:
    return hashlib.sha256(
        f"{seed}\0d4-source-compiler-smoke-v1\0{capability_id}\0{sample.dataset}\0{sample.sample_id}".encode()
    ).hexdigest()


def _unexpected_failure_row(*, sample, compiler_index: int, run_id: str, error: BaseException) -> dict[str, Any]:
    decision = route_for_sample(sample)
    return {
        "run_id": run_id,
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "capability_id": decision.capability_id,
        "kernel_id": decision.kernel_id,
        "query_operator": decision.query_operator,
        "method_name": "catch_kernel_d4_source_compiler_smoke",
        "role": "d4_source_compiler",
        "agent_id": compiler_index,
        "request_seed": 49_000 + compiler_index,
        "request_error": f"unexpected:{type(error).__name__}:{error}",
        "source_ir_v3_status": "failed",
        "source_ir_v3_parse_reason": "unexpected_worker_error",
        "solver_status": "NOT_RUN",
        "reference_checker_status": "NOT_RUN",
        "concrete_witness_status": {"status": "NOT_RUN"},
        "metamorphic_status": {},
        "metamorphic_checks_passed": False,
        "verification_passed": False,
        "candidate_or_gold_leakage": False,
        "compiler_input_fields": ["source", "answer_contract", "source_spans", "capability_id", "query_operator"],
    }
