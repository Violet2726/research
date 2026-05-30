"""矩阵编排公共入口。"""

from research_experiments.matrix.faithful_matrix import (
    MatrixBuild,
    MatrixEntry,
    OrchestratorPaths,
    ReviewResult,
    RuntimeOverrides,
    apply_runtime_overrides,
    assert_matrix_succeeded,
    build_run_matrix,
    collect_blocking_entries,
    discover_phase_configs,
    resume_faithful_matrix,
    resume_matrix,
    review_run_health,
    run_faithful_matrix,
    run_matrix,
)

__all__ = [
    "MatrixBuild",
    "MatrixEntry",
    "OrchestratorPaths",
    "ReviewResult",
    "RuntimeOverrides",
    "apply_runtime_overrides",
    "assert_matrix_succeeded",
    "build_run_matrix",
    "collect_blocking_entries",
    "discover_phase_configs",
    "resume_faithful_matrix",
    "resume_matrix",
    "review_run_health",
    "run_faithful_matrix",
    "run_matrix",
]
