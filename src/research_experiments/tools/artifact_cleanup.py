"""自动清理无效运行记录与无效报告。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import argparse
import json
import re
import shutil

from research_experiments.cli_support.output import configure_utf8_stdio, emit_json
from research_experiments.workspace.layout import workspace_layout
from research_experiments.families.registry import validator_map


RUN_ID_PATTERN = re.compile(r"20\d{6}T\d{6}Z-[A-Za-z0-9._:-]+")

RunValidator = Callable[[str | Path], dict[str, object]]

RUN_VALIDATORS: dict[str, RunValidator] = validator_map()


@dataclass(frozen=True)
class RunStatus:
    """单个运行目录的校验结果。"""

    family_name: str
    run_dir: Path
    run_id: str | None
    exists: bool
    passed: bool | None
    reason: str | None


@dataclass(frozen=True)
class ReportStatus:
    """单个报告文件的有效性判定。"""

    report_path: Path
    run_ids: tuple[str, ...]
    missing_run_ids: tuple[str, ...]
    failed_run_ids: tuple[str, ...]

    @property
    def is_invalid(self) -> bool:
        return bool(self.missing_run_ids or self.failed_run_ids)


@dataclass(frozen=True)
class CleanupSummary:
    """一次清理动作的汇总。"""

    dry_run: bool
    invalid_runs: tuple[RunStatus, ...]
    invalid_reports: tuple[ReportStatus, ...]
    removed_run_dirs: tuple[str, ...]
    removed_report_paths: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""
    layout = workspace_layout()
    parser = argparse.ArgumentParser(description="Delete invalid run records and invalid reports.")
    parser.add_argument("--workspace-root", default=".", help="Workspace root. Defaults to the current directory.")
    parser.add_argument("--runs-root", default=layout.runs_root.as_posix(), help="Runs root relative to workspace root.")
    parser.add_argument("--reports-root", default=layout.reports_root.as_posix(), help="Reports root relative to workspace root.")
    parser.add_argument(
        "--revalidate-runs",
        action="store_true",
        help="Recompute run validity with the current validator when possible. By default, existing run_validation.json is trusted first.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print cleanup candidates without deleting them.")
    parser.add_argument("--json", action="store_true", help="Print the cleanup summary as JSON.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """命令行入口。"""
    configure_utf8_stdio()
    args = build_parser().parse_args(argv)
    workspace_root = Path(args.workspace_root).resolve()
    runs_root = workspace_root / args.runs_root
    reports_root = workspace_root / args.reports_root
    summary = cleanup_invalid_artifacts(
        workspace_root=workspace_root,
        runs_root=runs_root,
        reports_root=reports_root,
        dry_run=args.dry_run,
        revalidate_runs=args.revalidate_runs,
    )
    if args.json:
        emit_json(_summary_to_dict(summary))
        return
    _print_summary(summary, workspace_root)


def cleanup_invalid_artifacts(
    workspace_root: Path,
    runs_root: Path,
    reports_root: Path,
    dry_run: bool = False,
    revalidate_runs: bool = False,
) -> CleanupSummary:
    """扫描并删除无效运行记录与无效报告。"""
    run_statuses = collect_run_statuses(
        workspace_root=workspace_root,
        runs_root=runs_root,
        revalidate_runs=revalidate_runs,
    )
    invalid_runs = tuple(status for status in run_statuses if status.reason is not None)
    run_status_by_id = {status.run_id: status for status in run_statuses if status.run_id}
    report_statuses = collect_report_statuses(reports_root=reports_root, run_status_by_id=run_status_by_id)
    invalid_reports = tuple(status for status in report_statuses if status.is_invalid)

    removed_run_dirs: list[str] = []
    removed_report_paths: list[str] = []

    if not dry_run:
        for status in invalid_reports:
            _delete_path(status.report_path, workspace_root)
            removed_report_paths.append(status.report_path.relative_to(workspace_root).as_posix())
        for status in invalid_runs:
            _delete_path(status.run_dir, workspace_root)
            removed_run_dirs.append(status.run_dir.relative_to(workspace_root).as_posix())

    return CleanupSummary(
        dry_run=dry_run,
        invalid_runs=invalid_runs,
        invalid_reports=invalid_reports,
        removed_run_dirs=tuple(removed_run_dirs),
        removed_report_paths=tuple(removed_report_paths),
    )


def collect_run_statuses(
    workspace_root: Path,
    runs_root: Path,
    revalidate_runs: bool = False,
) -> list[RunStatus]:
    """收集所有运行目录的有效性状态。"""
    statuses: list[RunStatus] = []
    if not runs_root.exists():
        return statuses

    for manifest_path in sorted(runs_root.rglob("manifest.json")):
        run_dir = manifest_path.parent
        try:
            family_name = run_dir.relative_to(runs_root).parts[0]
        except IndexError:
            family_name = ""
        run_id = _load_run_id(manifest_path)
        persisted = None if revalidate_runs else _load_persisted_validation(run_dir)
        if persisted is not None:
            statuses.append(
                RunStatus(
                    family_name=family_name,
                    run_dir=run_dir,
                    run_id=run_id,
                    exists=True,
                    passed=persisted["passed"],
                    reason=persisted["reason"],
                )
            )
            continue
        validator = RUN_VALIDATORS.get(family_name)
        if validator is None:
            statuses.append(
                RunStatus(
                    family_name=family_name,
                    run_dir=run_dir,
                    run_id=run_id,
                    exists=True,
                    passed=None,
                    reason=f"unknown_family:{family_name or 'root'}",
                )
            )
            continue
        try:
            validation = validator(run_dir)
        except Exception as exc:  # pragma: no cover - defensive fallback
            statuses.append(
                RunStatus(
                    family_name=family_name,
                    run_dir=run_dir,
                    run_id=run_id,
                    exists=True,
                    passed=False,
                    reason=f"validation_error:{exc.__class__.__name__}",
                )
            )
            continue

        passed_value = validation.get("passed")
        passed = bool(passed_value) if passed_value is not None else None
        missing_files = validation.get("missing_files") or []
        reason = None
        if passed is False:
            reason = "validation_failed"
        elif missing_files:
            reason = "missing_files"

        statuses.append(
            RunStatus(
                family_name=family_name,
                run_dir=run_dir,
                run_id=run_id,
                exists=True,
                passed=passed,
                reason=reason,
            )
        )
    return statuses


def collect_report_statuses(reports_root: Path, run_status_by_id: dict[str, RunStatus]) -> list[ReportStatus]:
    """根据报告中出现的 run_id 判定报告是否有效。"""
    statuses: list[ReportStatus] = []
    if not reports_root.exists():
        return statuses

    for report_path in sorted(path for path in reports_root.rglob("*") if path.is_file()):
        text = report_path.read_text(encoding="utf-8", errors="ignore")
        run_ids = tuple(sorted(set(RUN_ID_PATTERN.findall(text))))
        if not run_ids:
            statuses.append(
                ReportStatus(
                    report_path=report_path,
                    run_ids=(),
                    missing_run_ids=(),
                    failed_run_ids=(),
                )
            )
            continue

        missing = [run_id for run_id in run_ids if run_id not in run_status_by_id]
        failed = [
            run_id
            for run_id in run_ids
            if run_id in run_status_by_id and run_status_by_id[run_id].passed is False
        ]
        statuses.append(
            ReportStatus(
                report_path=report_path,
                run_ids=run_ids,
                missing_run_ids=tuple(missing),
                failed_run_ids=tuple(failed),
            )
        )
    return statuses


def _delete_path(path: Path, workspace_root: Path) -> None:
    """安全删除工作区内的路径。"""
    resolved = path.resolve()
    try:
        resolved.relative_to(workspace_root)
    except ValueError as exc:  # pragma: no cover - safety guard
        raise ValueError(f"Refusing to delete path outside workspace: {resolved}") from exc

    if not resolved.exists():
        return
    if resolved.is_dir():
        shutil.rmtree(resolved)
    else:
        resolved.unlink()


def _load_run_id(manifest_path: Path) -> str | None:
    """从 manifest 中读取 run_id。"""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    run_id = manifest.get("run_id")
    return str(run_id) if run_id else None


def _load_persisted_validation(run_dir: Path) -> dict[str, bool | str | None] | None:
    """优先读取运行目录自带的 run_validation.json。"""
    validation_path = run_dir / "run_validation.json"
    if not validation_path.exists():
        return None
    try:
        payload = json.loads(validation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"passed": False, "reason": "invalid_run_validation_json"}

    passed_value = payload.get("passed")
    passed = bool(passed_value) if passed_value is not None else None
    missing_files = payload.get("missing_files") or []
    reason = None
    if passed is False:
        reason = "validation_failed"
    elif missing_files:
        reason = "missing_files"
    return {"passed": passed, "reason": reason}


def _summary_to_dict(summary: CleanupSummary) -> dict[str, object]:
    """把清理汇总转换为 JSON 友好的结构。"""
    return {
        "dry_run": summary.dry_run,
        "invalid_runs": [
            {
                "family_name": status.family_name,
                "run_dir": status.run_dir.as_posix(),
                "run_id": status.run_id,
                "passed": status.passed,
                "reason": status.reason,
            }
            for status in summary.invalid_runs
        ],
        "invalid_reports": [
            {
                "report_path": status.report_path.as_posix(),
                "run_ids": list(status.run_ids),
                "missing_run_ids": list(status.missing_run_ids),
                "failed_run_ids": list(status.failed_run_ids),
            }
            for status in summary.invalid_reports
        ],
        "removed_run_dirs": list(summary.removed_run_dirs),
        "removed_report_paths": list(summary.removed_report_paths),
    }


def _print_summary(summary: CleanupSummary, workspace_root: Path) -> None:
    """输出简洁的人类可读摘要。"""
    mode = "DRY-RUN" if summary.dry_run else "DELETE"
    print(f"[{mode}] invalid runs: {len(summary.invalid_runs)}")
    for status in summary.invalid_runs:
        rel = status.run_dir.relative_to(workspace_root).as_posix()
        print(f"  - {rel} ({status.reason})")

    print(f"[{mode}] invalid reports: {len(summary.invalid_reports)}")
    for status in summary.invalid_reports:
        rel = status.report_path.relative_to(workspace_root).as_posix()
        if status.missing_run_ids:
            print(f"  - {rel} (missing run_ids: {', '.join(status.missing_run_ids)})")
        elif status.failed_run_ids:
            print(f"  - {rel} (failed run_ids: {', '.join(status.failed_run_ids)})")

    if not summary.dry_run:
        print(f"removed run dirs: {len(summary.removed_run_dirs)}")
        print(f"removed reports: {len(summary.removed_report_paths)}")

