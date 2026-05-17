"""统一维护各类实验矩阵的配置规格。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import tomllib


TRACK_SAME_CONTEXT = "same_context"
TRACK_SPLIT_CONTEXT = "split_context"
TRACK_GRAPH_REASONING = "graph_reasoning"
TRACK_TABLE_REASONING = "table_reasoning"
TRACK_TOPOLOGY_COLLABORATION = "topology_collaboration"

EVIDENCE_HEADLINE = "headline"
EVIDENCE_SUPPORTING = "supporting"
EVIDENCE_DIAGNOSTIC = "diagnostic"
EVIDENCE_REFERENCE = "reference"

ENTRY_ROLE_CANONICAL = "canonical"
ENTRY_ROLE_ABLATION = "ablation"
ENTRY_ROLE_SCALING = "scaling"
ENTRY_ROLE_CONTROL = "control"
ENTRY_ROLE_REFERENCE = "reference"

ANALYSIS_MODE_PRIMARY_SUMMARY = "primary_summary"
ANALYSIS_MODE_SCALING_SUMMARY = "scaling_summary"

COMPARISON_SCOPE_GLOBAL = "global"
COMPARISON_SCOPE_WITHIN_TRACK_ONLY = "within_track_only"

MATRIX_ID_FAITHFUL = "faithful"
MATRIX_ID_REPRODUCTION = "reproduction"
DEFAULT_MATRIX_ID = MATRIX_ID_FAITHFUL

MATRIX_RUN_KIND_FAITHFUL = "faithful_matrix"
MATRIX_RUN_KIND_REPRODUCTION = "reproduction_matrix"


@dataclass(frozen=True)
class MatrixProfileSpec:
    """单个矩阵 profile 的注册信息。"""

    matrix_id: str
    matrix_kind: str
    config_path: Path
    default_reports_root_name: str


@dataclass(frozen=True)
class ExperimentMatrixSpec:
    """单个实验入口在某个矩阵中的比较规格。"""

    matrix_id: str
    track_name: str
    entry_role: str
    primary_method_name: str
    primary_metric_field: str
    primary_metric_label: str
    comparison_scope: str
    analysis_mode: str
    evaluation_track: str
    evidence_tier: str
    best_no_comm_candidates: tuple[str, ...]
    full_comm_reference: str | None = None
    full_context_reference: str | None = None
    token_gate_basis: str = "none"


MATRIX_PROFILE_SPECS: dict[str, MatrixProfileSpec] = {
    MATRIX_ID_FAITHFUL: MatrixProfileSpec(
        matrix_id=MATRIX_ID_FAITHFUL,
        matrix_kind=MATRIX_RUN_KIND_FAITHFUL,
        config_path=Path("configs/core/matrix/faithful_matrix.toml"),
        default_reports_root_name=MATRIX_RUN_KIND_FAITHFUL,
    ),
    MATRIX_ID_REPRODUCTION: MatrixProfileSpec(
        matrix_id=MATRIX_ID_REPRODUCTION,
        matrix_kind=MATRIX_RUN_KIND_REPRODUCTION,
        config_path=Path("configs/core/matrix/reproduction_matrix.toml"),
        default_reports_root_name=MATRIX_RUN_KIND_REPRODUCTION,
    ),
}


def get_matrix_profile(matrix_id: str = DEFAULT_MATRIX_ID) -> MatrixProfileSpec:
    """返回某个矩阵 profile 的注册信息。"""

    try:
        return MATRIX_PROFILE_SPECS[matrix_id]
    except KeyError as exc:
        raise KeyError(f"Unknown matrix_id: {matrix_id}") from exc


@lru_cache(maxsize=None)
def _load_matrix_spec_rows(matrix_id: str) -> tuple[tuple[str, ExperimentMatrixSpec], ...]:
    profile = get_matrix_profile(matrix_id)
    with profile.config_path.open("rb") as handle:
        payload = tomllib.load(handle)
    rows: list[tuple[str, ExperimentMatrixSpec]] = []
    for entry in payload.get("entries", []):
        config_path = str(entry["config_path"])
        track_name = str(entry.get("track_name") or entry.get("evaluation_track") or "")
        entry_role = str(entry.get("entry_role") or entry.get("evidence_tier") or "")
        primary_method_name = str(entry.get("primary_method_name") or "")
        primary_metric_field = str(entry.get("primary_metric_field") or "accuracy_mean")
        primary_metric_label = str(entry.get("primary_metric_label") or "accuracy")
        comparison_scope = str(entry.get("comparison_scope") or COMPARISON_SCOPE_GLOBAL)
        analysis_mode = str(entry.get("analysis_mode") or ANALYSIS_MODE_PRIMARY_SUMMARY)
        rows.append(
            (
                config_path,
                ExperimentMatrixSpec(
                    matrix_id=matrix_id,
                    track_name=track_name,
                    entry_role=entry_role,
                    primary_method_name=primary_method_name,
                    primary_metric_field=primary_metric_field,
                    primary_metric_label=primary_metric_label,
                    comparison_scope=comparison_scope,
                    analysis_mode=analysis_mode,
                    evaluation_track=str(entry.get("evaluation_track") or track_name),
                    evidence_tier=str(entry.get("evidence_tier") or entry_role),
                    best_no_comm_candidates=tuple(str(item) for item in entry.get("best_no_comm_candidates", [])),
                    full_comm_reference=_optional_str(entry, "full_comm_reference"),
                    full_context_reference=_optional_str(entry, "full_context_reference"),
                    token_gate_basis=str(entry.get("token_gate_basis", "none")),
                ),
            )
        )
    return tuple(rows)


def ordered_matrix_config_paths(matrix_id: str = DEFAULT_MATRIX_ID) -> tuple[str, ...]:
    """返回指定矩阵的正式执行顺序。"""

    return tuple(config_path for config_path, _ in _load_matrix_spec_rows(matrix_id))


def get_experiment_matrix_spec(config_path: str, matrix_id: str = DEFAULT_MATRIX_ID) -> ExperimentMatrixSpec:
    """按实验配置路径返回矩阵规格。"""

    for registered_path, spec in _load_matrix_spec_rows(matrix_id):
        if registered_path == config_path:
            return spec
    raise KeyError(f"Missing matrix spec for config {config_path} in matrix {matrix_id}")


def all_matrix_ids() -> tuple[str, ...]:
    """返回当前支持的矩阵 profile 列表。"""

    return tuple(sorted(MATRIX_PROFILE_SPECS))


def _optional_str(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
