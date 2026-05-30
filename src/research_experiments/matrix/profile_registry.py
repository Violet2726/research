"""矩阵 profile 注册入口。"""

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

__all__ = [
    "DEFAULT_MATRIX_ID",
    "MATRIX_ID_FAITHFUL",
    "MATRIX_ID_REPRODUCTION",
    "MATRIX_PROFILE_SPECS",
    "MatrixProfileSpec",
    "all_matrix_ids",
    "get_experiment_matrix_spec",
    "get_matrix_profile",
    "ordered_matrix_config_paths",
    "referenced_method_names",
]
