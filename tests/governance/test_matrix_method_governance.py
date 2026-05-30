"""约束 faithful matrix 对标准 comparator 的引用，防止口径再次漂移。"""

from __future__ import annotations

from research_experiments.families.shared.standard_method_names import (
    COT_1,
    MV_3,
    MAD_3A_R1,
    MAD_FIXED_R3,
    MV_6,
    SC_5,
    SC_6,
    VANILLA_MAD_R1_FINAL_VOTE,
)
from research_experiments.matrix.matrix_specs import (
    MATRIX_ID_FAITHFUL,
    get_experiment_matrix_spec,
    ordered_matrix_config_paths,
    referenced_method_names,
)


def test_faithful_matrix_standard_method_references_stay_aligned() -> None:
    specs = {
        config_path: get_experiment_matrix_spec(config_path, MATRIX_ID_FAITHFUL)
        for config_path in ordered_matrix_config_paths(MATRIX_ID_FAITHFUL)
    }

    single_agent_core = specs["configs/families/single_agent/experiments/same_context_core_benchmarks.toml"]
    assert single_agent_core.primary_method_name == SC_5
    assert single_agent_core.best_no_comm_candidates == (COT_1, SC_5)

    single_agent_main = specs["configs/families/single_agent/experiments/same_context_main_table.toml"]
    assert single_agent_main.primary_method_name == SC_5
    assert single_agent_main.best_no_comm_candidates == (COT_1, SC_5)

    multi_agent = specs["configs/families/multi_agent/experiments/same_context_controlled_debate.toml"]
    assert multi_agent.primary_method_name == MAD_3A_R1
    assert multi_agent.best_no_comm_candidates == (MV_6,)

    imad = specs["configs/families/imad/experiments/imad_same_context_main.toml"]
    assert imad.best_no_comm_candidates == (MV_6,)
    assert imad.full_comm_reference == MAD_FIXED_R3

    free_mad = specs["configs/families/free_mad_lite/experiments/free_mad_lite_mechanism_validation.toml"]
    assert free_mad.best_no_comm_candidates == (MV_3,)
    assert free_mad.full_comm_reference == VANILLA_MAD_R1_FINAL_VOTE

    for config_path in (
        "configs/families/budget_comm/experiments/dala_lite_same_context_main.toml",
        "configs/families/budget_comm/experiments/dala_lite_split_context_main.toml",
        "configs/families/cue/experiments/cue_black_box_utility_main.toml",
        "configs/families/sid_lite/experiments/sid_lite_mechanism_validation.toml",
    ):
        assert specs[config_path].best_no_comm_candidates == (MV_3,)

    for config_path in (
        "configs/families/selective_comm/experiments/trigger_early_exit_main.toml",
        "configs/families/selective_comm/experiments/voc_trigger_main.toml",
    ):
        assert specs[config_path].best_no_comm_candidates == (MV_3, MV_6, SC_6)


def test_faithful_matrix_does_not_reference_removed_aliases() -> None:
    removed_aliases = {"mv_3_reuse", "mv_3_initial", "cot_1_math512", "sc_5_math512", "cot", "cot_sc", "sbp_sc", "pot_sc"}
    for config_path in ordered_matrix_config_paths(MATRIX_ID_FAITHFUL):
        spec = get_experiment_matrix_spec(config_path, MATRIX_ID_FAITHFUL)
        assert removed_aliases.isdisjoint(referenced_method_names(spec)), config_path
