"""共享的中文科研报告 Markdown 渲染工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def render_scientific_report(
    *,
    title: str,
    abstract: list[str],
    overview_items: list[tuple[str, str]],
    sections: list[dict[str, Any]],
) -> str:
    """按统一科研结构渲染中文 Markdown 报告。"""
    lines = [f"# {title}", ""]
    if abstract:
        lines.extend(["## 摘要", ""])
        lines.extend(_render_bullets(abstract))
        lines.append("")
    if overview_items:
        lines.extend(["## 实验概览", ""])
        lines.extend(_render_kv_bullets(overview_items))
        lines.append("")
    for section in sections:
        section_title = str(section.get("title") or "").strip()
        if not section_title:
            continue
        lines.extend([f"## {section_title}", ""])
        description = str(section.get("description") or "").strip()
        if description:
            lines.append(description)
            lines.append("")
        paragraphs = [str(item).strip() for item in section.get("paragraphs", []) if str(item).strip()]
        if paragraphs:
            lines.extend(paragraphs)
            lines.append("")
        bullets = [str(item).strip() for item in section.get("bullets", []) if str(item).strip()]
        if bullets:
            lines.extend(_render_bullets(bullets))
            lines.append("")
        table = section.get("table")
        if isinstance(table, dict):
            lines.extend(
                render_markdown_table(
                    headers=list(table.get("headers", [])),
                    rows=list(table.get("rows", [])),
                )
            )
            lines.append("")
        tables = section.get("tables", [])
        if isinstance(tables, list):
            for table_payload in tables:
                if not isinstance(table_payload, dict):
                    continue
                table_title = str(table_payload.get("title") or "").strip()
                if table_title:
                    lines.append(f"### {table_title}")
                    lines.append("")
                table_description = str(table_payload.get("description") or "").strip()
                if table_description:
                    lines.append(table_description)
                    lines.append("")
                lines.extend(
                    render_markdown_table(
                        headers=list(table_payload.get("headers", [])),
                        rows=list(table_payload.get("rows", [])),
                    )
                )
                lines.append("")
        cases = section.get("cases", [])
        if isinstance(cases, list) and cases:
            lines.extend(render_case_cards(cases))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_markdown_table(headers: list[str], rows: list[list[str]] | list[dict[str, Any]]) -> list[str]:
    """渲染标准 Markdown 表格。"""
    if not headers:
        return ["暂无数据。"]
    if not rows:
        return [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            "| " + " | ".join([""] * len(headers)) + " |",
        ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        if isinstance(row, dict):
            values = [str(row.get(header, "")) for header in headers]
        else:
            values = [str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def render_case_cards(cases: list[dict[str, Any]]) -> list[str]:
    """渲染典型案例小节。"""
    if not cases:
        return ["暂无可复核案例。"]
    lines: list[str] = []
    for index, case in enumerate(cases, start=1):
        lines.extend(
            [
                f"### 案例 {index}",
                "",
            ]
        )
        for label, value in case.items():
            if value in {None, ""}:
                continue
            lines.append(f"- {label}：`{value}`")
        lines.append("")
    return lines


def render_run_reproducibility_section(
    *,
    run_dir: Path,
    artifact_items: list[str],
    note_lines: list[str] | None = None,
) -> dict[str, Any]:
    """构建统一的复现与产物说明章节。"""
    bullets = [f"运行目录：`{run_dir.as_posix()}`"]
    bullets.extend(artifact_items)
    if note_lines:
        bullets.extend(note_lines)
    return {"title": "复现与产物说明", "bullets": bullets}


def format_float(value: Any, digits: int = 4) -> str:
    """稳定格式化浮点数。"""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def format_percent(value: Any, digits: int = 2) -> str:
    """把 0-1 比例格式化为百分比。"""
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return ""


def _render_bullets(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items]


def _render_kv_bullets(items: list[tuple[str, str]]) -> list[str]:
    return [f"- {label}：`{value}`" for label, value in items]
