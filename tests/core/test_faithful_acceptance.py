"""覆盖 faithful acceptance 汇总与门槛判定逻辑的测试。"""

from __future__ import annotations

from experiment_core.faithful_acceptance import build_acceptance_summary


def test_build_acceptance_summary_classifies_same_and_split_tracks() -> None:
    analysis_payload = {
        "rows": [
            {
                "family": "budget_comm",
                "config_path": "configs/budget_comm/experiments/dala_lite_same_context_v1.toml",
                "experiment_name": "dala_lite_same_context_v1",
                "evaluation_track": "same_context",
                "dataset": "overall",
                "primary_method_name": "dala_lite",
                "faithful_score": 0.71,
                "delta_vs_best_no_comm": -0.01,
                "full_comm_reference": "all_to_all_full",
                "token_ratio_vs_full_comm": 0.95,
                "communication_token_ratio_vs_full_comm": 0.2,
                "delta_vs_full_comm": -0.02,
                "delta_vs_full_context": None,
                "stage_ceiling_gap": 0.01,
                "engineering_noise_gap": 0.03,
            },
            {
                "family": "comm_necessary",
                "config_path": "configs/comm_necessary/experiments/hotpotqa_split_evidence_v1.toml",
                "experiment_name": "hotpotqa_split_evidence_v1",
                "evaluation_track": "split_context",
                "dataset": "overall",
                "primary_method_name": "full_packet_exchange",
                "faithful_score": 0.5,
                "best_no_comm_score": 0.3,
                "full_context_score": 0.6,
                "delta_vs_best_no_comm": 0.2,
                "full_context_reference": "full_context_single",
                "token_ratio_vs_full_comm": None,
                "delta_vs_full_comm": None,
                "delta_vs_full_context": -0.1,
                "stage_ceiling_gap": 0.0,
                "engineering_noise_gap": 0.02,
            },
            {
                "family": "selective_comm",
                "config_path": "configs/selective_comm/experiments/trigger_early_exit_v1.toml",
                "experiment_name": "trigger_early_exit_v1",
                "evaluation_track": "same_context",
                "dataset": "overall",
                "primary_method_name": "hybrid_trigger",
                "faithful_score": 0.75,
                "delta_vs_best_no_comm": -0.05,
                "full_comm_reference": "always_communicate",
                "token_ratio_vs_full_comm": 0.9,
                "communication_token_ratio_vs_full_comm": 0.7,
                "delta_vs_full_comm": -0.01,
                "delta_vs_full_context": None,
                "stage_ceiling_gap": 0.02,
                "engineering_noise_gap": -0.03,
            },
        ]
    }

    summary = build_acceptance_summary(analysis_payload)

    assert summary["counts"]["evaluated"] == 3
    assert summary["counts"]["accepted_same_context"] == 1
    assert summary["counts"]["accepted_split_context"] == 1
    assert summary["counts"]["negative_control_family"] == 1
