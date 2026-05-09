"""论文分析产物共用的 Markdown 报告模板。"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def render_frontier_report(
    summary_rows: list[dict[str, Any]],
    *,
    title: str = "Accuracy-Cost Frontier",
) -> str:
    """渲染精度-成本前沿汇总表。"""
    lines = [
        f"# {title}",
        "",
        "| Dataset | Method | Accuracy | Total Tokens | Comm Tokens | Audit Tokens | Calls / Q | Acc / 1K Tokens |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row.get('dataset')}` | `{row.get('method_name')}` | "
            f"{float(row.get('accuracy_mean') or 0.0):.4f} | "
            f"{float(row.get('total_tokens_mean') or 0.0):.2f} | "
            f"{float(row.get('communication_tokens_mean') or 0.0):.2f} | "
            f"{float(row.get('audit_tokens_mean') or 0.0):.2f} | "
            f"{float(row.get('calls_per_question_mean') or 0.0):.2f} | "
            f"{float(row.get('acc_per_1k_tokens') or 0.0):.6f} |"
        )
    return "\n".join(lines) + "\n"


def render_trigger_diagnostic_report(
    rows: list[dict[str, Any]],
    *,
    title: str = "Trigger Diagnostics",
) -> str:
    """渲染 trigger 策略诊断表。"""
    lines = [
        f"# {title}",
        "",
        "| Dataset | Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall | False Trigger Rate | Missed Beneficial Comm Rate | Avg Comm Tokens |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row.get('dataset')}` | `{row.get('method_name') or row.get('policy_name')}` | "
            f"{float(row.get('trigger_rate') or 0.0):.4f} | "
            f"{float(row.get('early_exit_rate') or 0.0):.4f} | "
            f"{float(row.get('oracle_precision') or 0.0):.4f} | "
            f"{float(row.get('oracle_recall') or 0.0):.4f} | "
            f"{float(row.get('false_trigger_rate') or 0.0):.4f} | "
            f"{float(row.get('missed_beneficial_comm_rate') or 0.0):.4f} | "
            f"{float(row.get('communication_tokens_mean') or 0.0):.2f} |"
        )
    return "\n".join(lines) + "\n"


def render_audit_diagnostic_report(
    rows: list[dict[str, Any]],
    *,
    title: str = "Audit Diagnostics",
) -> str:
    """渲染审计环节诊断表。"""
    lines = [
        f"# {title}",
        "",
        "| Dataset | Method | Resolve Rate | Abstain Rate | Wrong Overrule Rate | Minority Rescue Count | Audit Tokens |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row.get('dataset')}` | `{row.get('method_name')}` | "
            f"{float(row.get('resolve_rate') or 0.0):.4f} | "
            f"{float(row.get('abstain_rate') or 0.0):.4f} | "
            f"{float(row.get('wrong_overrule_rate') or 0.0):.4f} | "
            f"{int(row.get('minority_rescue_count') or 0)} | "
            f"{float(row.get('audit_tokens_mean') or 0.0):.2f} |"
        )
    return "\n".join(lines) + "\n"


def render_split_context_report(
    rows: list[dict[str, Any]],
    *,
    title: str = "Split-Context Communication Report",
) -> str:
    """渲染 split-context 通信实验汇总表。"""
    lines = [
        f"# {title}",
        "",
        "| Dataset | Method | Answer EM | Answer F1 | Support F1 | Joint F1 | Comm Tokens | Total Tokens |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row.get('dataset')}` | `{row.get('method_name')}` | "
            f"{float(row.get('answer_em_mean') or 0.0):.4f} | "
            f"{float(row.get('answer_f1_mean') or 0.0):.4f} | "
            f"{float(row.get('support_f1_mean') or 0.0):.4f} | "
            f"{float(row.get('joint_f1_mean') or 0.0):.4f} | "
            f"{float(row.get('communication_tokens_mean') or 0.0):.2f} | "
            f"{float(row.get('total_tokens_mean') or 0.0):.2f} |"
        )
    return "\n".join(lines) + "\n"


def write_report(path: str | Path, content: str) -> Path:
    """把报告内容写入磁盘并返回目标路径。"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    return output
