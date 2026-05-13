"""测试中常用的文件落盘辅助函数。"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    """把对象按 UTF-8 JSON 写入磁盘，并确保父目录存在。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """把行记录写成 UTF-8 JSONL 文件，并确保父目录存在。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def touch_figure_contract(root: Path, report_name: str = "report.md") -> None:
    """为报告/归档相关测试构造最小 figure 与 archive 合同。"""

    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_rows = []
    for figure_id, title, metric in [
        ("frontier_overall", "Frontier", "Accuracy"),
        ("efficiency_rank_overall", "Efficiency Rank", "Accuracy per 1K tokens"),
        ("score_by_dataset", "Score by Dataset", "Accuracy"),
    ]:
        (figures_dir / f"{figure_id}.svg").write_text(
            "<svg xmlns=\"http://www.w3.org/2000/svg\"><text>test</text></svg>\n",
            encoding="utf-8",
        )
        (figures_dir / f"{figure_id}.csv").write_text("label,value\nexample,1\n", encoding="utf-8")
        figure_rows.append(
            {
                "figure_id": figure_id,
                "title": title,
                "caption": "test figure",
                "svg_path": f"figures/{figure_id}.svg",
                "csv_path": f"figures/{figure_id}.csv",
                "source_kind": "test",
                "dataset_scope": "overall",
                "primary_metric": metric,
            }
        )
    write_json(
        root / "figure_manifest.json",
        {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "figure_count": len(figure_rows),
            "figures": figure_rows,
        },
    )
    (root / report_name).write_text(
        "# report\n\n![Frontier](figures/frontier_overall.svg)\n",
        encoding="utf-8",
    )
    write_json(
        root / "archive_manifest.json",
        {
            "schema_version": 1,
            "generated_at": "2026-01-01T00:00:00+00:00",
            "run_id": "test-run",
            "remote_repo": None,
            "remote_prefix": "family/experiment/phase/test-run",
            "artifacts_packaged": False,
            "visible_files": [
                report_name,
                "figure_manifest.json",
                "figures/frontier_overall.svg",
                "figures/frontier_overall.csv",
                "figures/efficiency_rank_overall.svg",
                "figures/efficiency_rank_overall.csv",
                "figures/score_by_dataset.svg",
                "figures/score_by_dataset.csv",
            ],
            "archives": [],
        },
    )
