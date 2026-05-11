"""共享的中文分析附录报告模板。"""

from __future__ import annotations

from typing import Any

from research_experiments.reporting.scientific_report import render_markdown_table


def render_frontier_report(
    summary_rows: list[dict[str, Any]],
    *,
    title: str = "成本-性能前沿附录",
) -> str:
    """渲染标准成本-性能附录表。"""

    return _render_appendix_table(
        title=title,
        description="该附录表保留所有 summary 行，便于快速复核方法在成本、通信与整体效率上的相对位置。",
        headers=["数据集", "方法", "准确率", "总 token", "通信 token", "审计 token", "每题调用数", "每千 token 得分"],
        rows=[
            [
                f"`{row.get('dataset')}`",
                f"`{row.get('method_name')}`",
                f"{float(row.get('accuracy_mean') or 0.0):.4f}",
                f"{float(row.get('total_tokens_mean') or 0.0):.2f}",
                f"{float(row.get('communication_tokens_mean') or 0.0):.2f}",
                f"{float(row.get('audit_tokens_mean') or 0.0):.2f}",
                f"{float(row.get('calls_per_question_mean') or 0.0):.2f}",
                f"{float(row.get('acc_per_1k_tokens') or row.get('accuracy_per_1k_tokens') or 0.0):.6f}",
            ]
            for row in summary_rows
        ],
    )


def render_trigger_diagnostic_report(
    rows: list[dict[str, Any]],
    *,
    title: str = "触发策略诊断附录",
) -> str:
    """渲染 trigger 策略诊断表。"""

    return _render_appendix_table(
        title=title,
        description="该附录表聚焦 trigger 决策行为本身，用于判断策略是靠更高召回还是更低误触发获得收益。",
        headers=["数据集", "策略", "触发率", "早退率", "Oracle 精确率", "Oracle 召回率", "误触发率", "漏掉有益通信率", "平均通信 token"],
        rows=[
            [
                f"`{row.get('dataset')}`",
                f"`{row.get('method_name') or row.get('policy_name')}`",
                f"{float(row.get('trigger_rate') or 0.0):.4f}",
                f"{float(row.get('early_exit_rate') or 0.0):.4f}",
                f"{float(row.get('oracle_precision') or 0.0):.4f}",
                f"{float(row.get('oracle_recall') or 0.0):.4f}",
                f"{float(row.get('false_trigger_rate') or 0.0):.4f}",
                f"{float(row.get('missed_beneficial_comm_rate') or 0.0):.4f}",
                f"{float(row.get('communication_tokens_mean') or 0.0):.2f}",
            ]
            for row in rows
        ],
    )


def render_audit_diagnostic_report(
    rows: list[dict[str, Any]],
    *,
    title: str = "审计环节诊断附录",
) -> str:
    """渲染审计策略诊断表。"""

    return _render_appendix_table(
        title=title,
        description="该附录表集中展示审计是否真正解决冲突、是否过度覆盖原始答案，以及其额外审计开销。",
        headers=["数据集", "方法", "解决率", "弃权率", "错误覆盖率", "少数派挽救数", "平均审计 token"],
        rows=[
            [
                f"`{row.get('dataset')}`",
                f"`{row.get('method_name')}`",
                f"{float(row.get('resolve_rate') or 0.0):.4f}",
                f"{float(row.get('abstain_rate') or 0.0):.4f}",
                f"{float(row.get('wrong_overrule_rate') or 0.0):.4f}",
                str(int(row.get('minority_rescue_count') or 0)),
                f"{float(row.get('audit_tokens_mean') or 0.0):.2f}",
            ]
            for row in rows
        ],
    )


def render_split_context_report(
    rows: list[dict[str, Any]],
    *,
    title: str = "Split-Context 通信附录",
) -> str:
    """渲染 split-context 结果附录表。"""

    return _render_appendix_table(
        title=title,
        description="该附录表把答案质量、证据质量和联合指标并列展示，用于复核通信是否同时改善答案与证据推理。",
        headers=["数据集", "方法", "Answer EM", "Answer F1", "Support F1", "Joint F1", "通信 token", "总 token"],
        rows=[
            [
                f"`{row.get('dataset')}`",
                f"`{row.get('method_name')}`",
                f"{float(row.get('answer_em_mean') or 0.0):.4f}",
                f"{float(row.get('answer_f1_mean') or 0.0):.4f}",
                f"{float(row.get('support_f1_mean') or row.get('supporting_f1_mean') or 0.0):.4f}",
                f"{float(row.get('joint_f1_mean') or 0.0):.4f}",
                f"{float(row.get('communication_tokens_mean') or 0.0):.2f}",
                f"{float(row.get('total_tokens_mean') or 0.0):.2f}",
            ]
            for row in rows
        ],
    )


def _render_appendix_table(
    *,
    title: str,
    description: str,
    headers: list[str],
    rows: list[list[str]],
) -> str:
    lines = [f"# {title}", "", description, ""]
    lines.extend(render_markdown_table(headers=headers, rows=rows))
    lines.append("")
    return "\n".join(lines)
