"""矩阵 profile 与条目构建入口。"""

from research_experiments.matrix.matrix_specs import (
    DEFAULT_MATRIX_ID,
    MATRIX_ID_FAITHFUL,
    MATRIX_ID_REPRODUCTION,
    MATRIX_PROFILE_SPECS,
    MatrixProfileSpec,
    all_matrix_ids,
    get_experiment_matrix_spec,
    get_matrix_profile,
    ordered_matrix_config_paths,
    referenced_method_names,
)
from research_experiments.matrix.orchestrator import DiscoveredConfig, RuntimeOverrides, build_run_matrix, discover_phase_configs

__all__ = [
    "DEFAULT_MATRIX_ID",
    "DiscoveredConfig",
    "MATRIX_ID_FAITHFUL",
    "MATRIX_ID_REPRODUCTION",
    "MATRIX_PROFILE_SPECS",
    "MatrixProfileSpec",
    "RuntimeOverrides",
    "all_matrix_ids",
    "build_run_matrix",
    "discover_phase_configs",
    "get_experiment_matrix_spec",
    "get_matrix_profile",
    "ordered_matrix_config_paths",
    "referenced_method_names",
]

