from __future__ import annotations

from collections import Counter

from research_experiments.matrix.faithful_matrix import RuntimeOverrides, build_run_matrix


def test_build_run_matrix_counts_expected_for_reproduction() -> None:
    overrides = RuntimeOverrides()
    matrix = build_run_matrix(overrides, matrix_id="reproduction")
    semantic_counts = Counter(entry.status for entry in matrix.semantic_entries)

    assert matrix.matrix_id == "reproduction"
    assert matrix.matrix_kind == "reproduction_matrix"
    assert len(matrix.semantic_entries) == 7
    assert semantic_counts["pending"] == 7
    assert matrix.counts["semantic_unique_targets"] == 7
    track_names = {entry.experiment_name: entry.track_name for entry in matrix.semantic_entries}
    assert track_names["dmad_reasoning_main"] == "same_context"
    assert track_names["dog_graph_main"] == "graph_reasoning"
    assert track_names["table_critic_main"] == "table_reasoning"
    assert track_names["colmad_realmistake_main"] == "oversight_protocol"
    assert track_names["macnet_paper_main"] == "topology_collaboration"
    scaling_entry = next(entry for entry in matrix.semantic_entries if entry.experiment_name == "macnet_scaling_study")
    assert scaling_entry.entry_role == "scaling"
    assert scaling_entry.analysis_mode == "scaling_summary"
