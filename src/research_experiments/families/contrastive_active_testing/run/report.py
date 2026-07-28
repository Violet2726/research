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
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {},
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
    protocol_version = manifest.get("protocol_version")
    is_cert = protocol_version in {"catch_cert_v1", "catch_cert_v2", "catch_kernel_v1"}
    primary_method = (
        "catch_kernel"
        if protocol_version == "catch_kernel_v1"
        else "catch_cert_v2"
        if protocol_version == "catch_cert_v2"
        else "catch_cert"
        if is_cert
        else "catch"
    )
    lines = [
        f"# {manifest.get('paper_method_name') or 'CATCH'} 实验结果",
        "",
        f"运行状态：`{manifest.get('run_status') or 'running'}`",
        "",
        f"实验阶段：`{manifest.get('phase_name') or 'unknown'}`",
        "",
        "本报告采用 best-effort 执行策略；请求失败、协议解析失败和数据集适配失败均单独记录，不会静默计入正确率。",
        "",
        "## 数据集结果覆盖",
        "",
    ]
    readiness = manifest.get("readiness_assessment") or {}
    d3_audit = manifest.get("d3_data_audit") or {}
    if d3_audit:
        lines.extend(
            [
                "## D3 data-independence audit",
                "",
                f"- role: `{d3_audit.get('primary_confirmation_role')}`",
                f"- official BBEH Mini: `{d3_audit.get('official_mini_count', 0)}`; overlap with inspected: `{d3_audit.get('official_mini_overlap_with_inspected_count', 0)}`",
                f"- official Mini text-hash overlap with inspected: `{d3_audit.get('official_mini_text_hash_overlap_with_inspected_count', 0)}`",
                f"- selected BBEH overlap with inspected: `{d3_audit.get('selected_bbeh_inspected_overlap_count', 0)}`",
                f"- selected BBEH text-hash overlap with inspected: `{d3_audit.get('selected_bbeh_text_hash_overlap_with_inspected_count', 0)}`",
                f"- selected BBEH overlap with official Mini: `{d3_audit.get('selected_bbeh_official_mini_overlap_count', 0)}`",
                "",
            ]
        )
    if protocol_version == "catch_cert_v2" and manifest.get("phase_name") in {
        "heldout",
        "confirmation",
    }:
        unmet = list(readiness.get("unmet_conditions") or [])
        lines.extend(
            [
                "## 科研证据状态（非阻断）",
                "",
                "readiness assessment 只用于解释结果，不会终止运行、删除失败样本或阻止后续阶段。",
                "",
                f"- 诊断状态：`{readiness.get('status', 'missing')}`",
                f"- 证据解释：`{manifest.get('evidence_interpretation') or 'exploratory_diagnostic_evidence'}`",
                f"- 是否阻止执行：`{bool(readiness.get('blocks_execution', False))}`",
                f"- 推荐条件全部满足：`{bool(readiness.get('all_recommended_conditions_met', False))}`",
                f"- 未满足项数量：`{len(unmet)}`",
            ]
        )
        if unmet:
            lines.extend(f"  - `{condition}`" for condition in unmet)
        lines.append("")

    screening = metrics.get("screening") or {}
    if screening:
        lines.extend(
            [
                "### Screening 池",
                "",
                "| 数据集 | 状态 | 样本数 | SC5 | 候选 oracle | 目标 oracle | 分歧数 | Stage-A 无效 |",
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
        lines.append("尚未写入可评价的数据集结果。")
    for dataset, payload in datasets.items():
        lines.extend(
            [
                f"### {dataset}",
                "",
                f"计划样本：**{payload.get('planned', 0)}**；已尝试：**{payload.get('attempted', 0)}**；样本错误：**{payload.get('sample_errors', 0)}**。",
                "",
                "| 方法 | 可评价 | 缺失 | 完整准确率 (Complete-case) | Wilson 95% | 缺失按错 (Missing=wrong) | 平均 token/题 | token 中位数 | token P90 | 平均调用/题 | 每千 token 正确 | token/正确题 | 错改对 | 对改错 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method, row in (payload.get("methods") or {}).items():
            if method not in {
                "sc_5",
                "adaptive_sc_8",
                "fixed_sc_8",
                "solver_direct",
                "catch",
                "catch_cert",
                "catch_cert_v2",
                "catch_kernel",
                "direct_judge_3",
                "pair_judge_3",
            } and not method.startswith("catch_d3_"):
                continue
            interval = row.get("accuracy_wilson_95") or [0, 0]
            lines.append(
                f"| {method} | {row.get('evaluable', 0)} | {row.get('missing', 0)} | "
                f"{float(row.get('complete_case_accuracy') or 0):.2%} | "
                f"[{float(interval[0]):.2%}, {float(interval[1]):.2%}] | "
                f"{float(row.get('conservative_accuracy_missing_as_wrong') or 0):.2%} | "
                f"{float(row.get('mean_total_tokens') or 0):.1f} | "
                f"{float(row.get('median_total_tokens') or 0):.1f} | "
                f"{float(row.get('p90_total_tokens') or 0):.1f} | "
                f"{float(row.get('mean_calls_per_question') or 0):.2f} | "
                f"{float(row.get('correct_per_1000_tokens') or 0):.4f} | "
                f"{float(row.get('tokens_per_correct') or 0):.1f} | "
                f"{row.get('corrected', 0)} | {row.get('harmed', 0)} |"
            )
        cert = (payload.get("methods") or {}).get(primary_method) or {}
        if cert:
            transitions = cert.get("transitions") or {}
            lines.extend(
                [
                    "",
                    f"{primary_method} 机制：证书覆盖率 {float(cert.get('certificate_coverage') or cert.get('eligible_rate') or 0):.2%}；",
                    f"证书利用率 {float(cert.get('certificate_utilization') or 0):.2%}；弃权率 {float(cert.get('abstention_rate') or 0):.2%}；",
                    f"verifier false-pass={cert.get('verifier_false_pass', 0)}，false-reject={cert.get('verifier_false_reject', 0)}；",
                    f"headroom 利用率 {float(cert.get('headroom_utilization') or 0):.2%}；",
                    f"答案连接覆盖率 {float(cert.get('answer_link_coverage') or 0):.2%}；"
                    f"问题义务覆盖率 {float(cert.get('obligation_coverage') or 0):.2%}；"
                    f"adapter 执行测试数 {int(cert.get('adapter_executed_test_count') or 0)}；"
                    f"格式修复数 {int(cert.get('verifier_format_repair_count') or 0)}；",
                    f"语法/schema/类型编译有效率="
                    f"{float(cert.get('syntax_validity') or 0):.2%}/"
                    f"{float(cert.get('schema_validity') or 0):.2%}/"
                    f"{float(cert.get('typed_compilation_validity') or 0):.2%}；"
                    f"人工语义有效率={_optional_percent(cert.get('semantic_validity'))}；"
                    f"契约正确率={_optional_percent(cert.get('contract_accuracy'))}；"
                    f"验证辖域覆盖率={float(cert.get('verifier_jurisdiction_coverage') or 0):.2%}；"
                    f"证明完整率={float(cert.get('proof_completeness') or 0):.2%}；",
                    f"结构义务完整率={_optional_percent(cert.get('structural_obligation_completeness'))}；"
                    f"provenance 有效率={_optional_percent(cert.get('provenance_validity'))}；"
                    f"entailment 有效率={_optional_percent(cert.get('entailment_validity'))}；"
                    f"adapter EXECUTED/CONFLICT/UNSUPPORTED/INVALID="
                    f"{int(cert.get('adapter_executed_test_count') or 0)}/"
                    f"{int(cert.get('adapter_conflict_test_count') or 0)}/"
                    f"{int(cert.get('adapter_unsupported_test_count') or 0)}/"
                    f"{int(cert.get('adapter_invalid_test_count') or 0)}；"
                    f"panel 分歧={int(cert.get('panel_disagreement_count') or 0)}；",
                    f"证明 PASS/CONFLICT/UNSUPPORTED/UNKNOWN="
                    f"{int(cert.get('proof_pass_count') or 0)}/"
                    f"{int(cert.get('proof_conflict_count') or 0)}/"
                    f"{int(cert.get('proof_unsupported_count') or 0)}/"
                    f"{int(cert.get('proof_unknown_count') or 0)}；",
                    f"wrong→correct={transitions.get('wrong_to_correct', 0)}，correct→wrong={transitions.get('correct_to_wrong', 0)}，"
                    f"wrong→wrong={transitions.get('wrong_to_wrong', 0)}，correct→correct={transitions.get('correct_to_correct', 0)}。",
                ]
            )
            if dataset == "seqbench":
                lines.extend(
                    [
                        f"seqBench：exact={float(cert.get('seqbench_exact_match') or 0):.4f}，"
                        f"progress={float(cert.get('seqbench_progress_ratio') or 0):.4f}，"
                        f"precision={float(cert.get('seqbench_precision') or 0):.4f}，"
                        f"recall={float(cert.get('seqbench_recall') or 0):.4f}，"
                        f"合法动作率={float(cert.get('seqbench_valid_action_rate') or 0):.4f}，"
                        f"执行前缀比例={float(cert.get('seqbench_execution_prefix_ratio') or 0):.4f}，"
                        f"完成有效率={float(cert.get('seqbench_completion_validity') or 0):.4f}。",
                    ]
                )
        if manifest.get("kernel_revision") == "d3_source_blind_v1":
            lines.extend(
                [
                    "",
                    "D3 audit:",
                    f"- primary_metric: `{payload.get('primary_metric') or cert.get('primary_metric')}`",
                    f"- route_counts: `{json.dumps(cert.get('d3_route_counts') or {}, ensure_ascii=False)}`",
                    f"- first_failure_counts: `{json.dumps(cert.get('d3_first_failure_counts') or {}, ensure_ascii=False)}`",
                    f"- route_quality: `{json.dumps(cert.get('d3_route_quality') or {}, ensure_ascii=False)}`",
                    f"- candidate_completion: `{cert.get('d3_candidate_completion_count', 0)}`; solver_direct: `{cert.get('d3_solver_direct_count', 0)}`",
                    f"- semantic_shadow: `{cert.get('d3_semantic_shadow_count', 0)}`",
                    f"- method cost: calls=`{cert.get('mean_calls_per_question')}`, tokens=`{cert.get('mean_total_tokens')}`, "
                    f"latency_ms=`{cert.get('mean_latency_ms_per_question')}`, cache_hits=`{cert.get('mean_cache_hits_per_question')}`, "
                    f"network_calls=`{cert.get('mean_network_calls_per_question')}`",
                    f"- override_precision_one_sided_95_lower: `{cert.get('d3_override_precision_one_sided_95_lower')}`",
                    f"- corrections/harm: `{cert.get('d3_correction_count', 0)}` / `{cert.get('d3_harm_count', 0)}`; harm upper CI: `{cert.get('d3_harm_rate_one_sided_95_upper')}`",
                    "- certificate language: conditional `source/IR -> answer`; not a proof of source-to-gold semantic equivalence.",
                ]
            )
            if dataset == "gpqa_diamond":
                lines.extend(
                    [
                        f"- GPQA domain_accuracy: `{json.dumps(cert.get('per_domain_accuracy') or {}, ensure_ascii=False)}`",
                        f"- GPQA subdomain_accuracy: `{json.dumps(cert.get('per_subdomain_accuracy') or {}, ensure_ascii=False)}`",
                        f"- GPQA quantitative/conceptual available: `{bool(cert.get('reasoning_type_stratification_available'))}`; "
                        "no heuristic labels are invented when audited labels are absent.",
                    ]
                )
        paired = payload.get("paired_complete_cases") or {}
        if paired:
            lines.extend(["", f"配对比较（主方法：{primary_method}）：", ""])
            for competitor, row in paired.items():
                lines.append(
                    f"- vs `{competitor}`：n={row.get('paired_sample_count', 0)}，"
                    f"主方法={float(row.get('catch_accuracy') or 0):.2%}，"
                    f"对照={float(row.get('competitor_accuracy') or 0):.2%}。"
                )

    paired_tests = (metrics.get("paired_statistics") or {}).get("tests") or []
    if paired_tests:
        lines.extend(
            [
                "",
                "## 配对推断",
                "",
                "| 数据集 | 对照 | 配对题数 | 准确率差 | 仅主方法正确 | 仅对照正确 | exact McNemar p | Holm p |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in paired_tests:
            holm = row.get("holm_adjusted_p")
            holm_text = f"{float(holm):.6f}" if holm is not None else "—"
            lines.append(
                f"| {row.get('dataset')} | {row.get('comparison_method')} | "
                f"{row.get('paired_question_count', 0)} | {float(row.get('mean_accuracy_delta') or 0):.2%} | "
                f"{row.get('mcnemar_b_reference_only_correct', 0)} | "
                f"{row.get('mcnemar_c_comparator_only_correct', 0)} | "
                f"{float(row.get('mcnemar_exact_p') or 0):.6f} | "
                f"{holm_text} |"
            )

    comparison_audit = metrics.get("comparison_method_audit") or {}
    if comparison_audit:
        lines.extend(["", "## 预注册对照完整性", ""])
        for dataset, row in sorted(comparison_audit.items()):
            missing = list(row.get("missing") or [])
            lines.append(
                f"- {dataset}：完整={bool(row.get('complete'))}；"
                f"已有={','.join(row.get('available') or []) or '无'}；"
                f"缺失={','.join(missing) or '无'}。"
            )

    failures = metrics.get("failures") or {}
    interval = failures.get("request_failure_rate_wilson_95") or [0, 0]
    lines.extend(
        [
            "",
            "## 执行失败与随机波动",
            "",
            f"- logical calls：**{failures.get('logical_call_count', 0)}**",
            f"- 请求失败：**{failures.get('request_failure_count', 0)}** "
            f"({float(failures.get('request_failure_rate') or 0):.2%}；Wilson 95% CI "
            f"{float(interval[0]):.2%} to {float(interval[1]):.2%})",
            f"- 结构化/答案解析失败：**{failures.get('parse_failure_count', 0)}**",
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
    cert_v2 = mechanism.get("certificate_v2") or {}
    kernel = mechanism.get("kernel") or {}
    lines.extend(
        [
            "",
            "## 机制诊断",
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
            f"- v2 answer-link coverage: `{float(cert_v2.get('mean_answer_link_coverage') or 0):.4f}`",
            f"- v2 mandatory-obligation coverage: `{float(cert_v2.get('mean_obligation_coverage') or 0):.4f}`",
            f"- v2 panel exact agreement: `{float(cert_v2.get('panel_exact_agreement_rate') or 0):.4f}`",
            f"- v2 adapter executed tests / format repairs: `{int(cert_v2.get('adapter_executed_test_count') or 0)}` / "
            f"`{int(cert_v2.get('format_repair_count') or 0)}`",
            f"- Kernel 验证辖域覆盖率：`{float(kernel.get('jurisdiction_coverage') or 0):.4f}`",
            f"- Kernel 证明完整率：`{float(kernel.get('proof_completeness') or 0):.4f}`",
            f"- Kernel 跨辖域回退次数：`{int(kernel.get('cross_jurisdiction_fallback_count') or 0)}`",
            f"- Kernel verifier 路由：`{json.dumps(kernel.get('verifier_route_counts') or {}, ensure_ascii=False)}`",
            f"- Kernel 各路由修正精度与 harm："
            f"`{json.dumps(kernel.get('verifier_route_quality') or {}, ensure_ascii=False)}`",
            f"- Kernel adapter 状态：`{json.dumps(kernel.get('adapter_status_counts') or {}, ensure_ascii=False)}`",
            f"- Kernel 证明状态：`{json.dumps(kernel.get('proof_status_counts') or {}, ensure_ascii=False)}`",
            f"- Kernel 首个失败层：`{json.dumps(kernel.get('failure_layer_counts') or {}, ensure_ascii=False)}`",
            f"- Kernel panel 分歧次数：`{int(kernel.get('panel_disagreement_count') or 0)}`",
        ]
    )
    if cert_v2.get("dropped_reason_counts"):
        lines.extend(
            [
                "",
                "v2 证书编译丢弃原因：",
                "",
                *[f"- `{reason}`: {count}" for reason, count in dict(cert_v2["dropped_reason_counts"]).items()],
            ]
        )

    costs = metrics.get("costs") or {}
    budget = execution.get("network_attempt_budget") or {}
    warnings = list(execution.get("warnings") or manifest.get("execution_warnings") or [])
    lines.extend(
        [
            "",
            "## 成本与警告",
            "",
            f"- cache 命中：`{costs.get('cache_hits', 0)}`",
            f"- 实际网络尝试：`{costs.get('physical_network_attempts', 0)}`",
            f"- 重试次数：`{costs.get('retry_attempts', 0)}`",
            f"- 实际总 token：`{costs.get('actual_total_tokens', 0)}`",
            f"- 尝试次数警告阈值：`{budget.get('configured_limit')}`；超额：`{budget.get('overage', 0)}`",
            "",
        ]
    )
    if warnings:
        lines.extend(f"- `{warning}`" for warning in warnings)
    else:
        lines.append("- 未记录配置或执行警告。")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report_path": target.as_posix(), "run_status": manifest.get("run_status")}


def _optional_percent(value: Any) -> str:
    return "待人工审计" if value is None else f"{float(value):.2%}"
