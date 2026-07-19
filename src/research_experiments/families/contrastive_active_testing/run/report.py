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
    is_cert = manifest.get("protocol_version") == "catch_cert_v1"
    primary_method = "catch_cert" if is_cert else "catch"
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
            if method not in {"sc_5", "adaptive_sc_8", "catch", "catch_cert", "direct_judge_3", "pair_judge_3"}:
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
                        f"执行前缀比例={float(cert.get('seqbench_execution_prefix_ratio') or 0):.4f}。",
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
