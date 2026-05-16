from __future__ import annotations

from research_experiments.families.macnet.config import load_experiment_config, load_protocol_config
from research_experiments.families.macnet.run.sample import _build_metrics
from research_experiments.families.macnet.topologies import build_topology


def test_load_macnet_experiment_and_protocol() -> None:
    experiment = load_experiment_config("configs/families/macnet/experiments/macnet_paper_main.toml")
    protocol = load_protocol_config(experiment.protocol)

    assert experiment.name == "macnet_paper_main"
    assert experiment.experiment_kind == "paper"
    assert [method.name for method in experiment.methods] == [
        "single_agent_cot",
        "macnet_chain",
        "macnet_star",
        "macnet_tree",
        "macnet_mesh",
        "macnet_layer",
        "macnet_random",
    ]
    assert protocol.profile_asset_path.as_posix() == "macnet/srdd-profile-repo.zip"


def test_topology_builder_returns_valid_dag_statistics() -> None:
    topology = build_topology("star", node_count=4, direction_mode="convergent", seed=42)

    assert topology.node_count == 4
    assert topology.sink_nodes == [3]
    assert topology.source_nodes == [0, 1, 2]
    assert len(topology.edges) == 3
    assert topology.dag_depth >= 1


def test_build_metrics_keeps_node_scale_and_direction() -> None:
    metrics = _build_metrics(
        [
            {
                "dataset": "mmlu",
                "method_name": "macnet_chain",
                "node_scale": 4,
                "topology_direction_mode": "divergent",
                "score": 1.0,
                "total_tokens_per_question": 100.0,
                "communication_tokens_per_question": 20.0,
                "calls_per_question": 7,
                "artifact_revision_count": 3,
                "inbound_instruction_count": 3,
                "max_context_tokens_observed": 40.0,
            },
            {
                "dataset": "mmlu",
                "method_name": "single_agent_cot",
                "node_scale": 1,
                "topology_direction_mode": "none",
                "score": 0.0,
                "total_tokens_per_question": 20.0,
                "communication_tokens_per_question": 0.0,
                "calls_per_question": 1,
                "artifact_revision_count": 0,
                "inbound_instruction_count": 0,
                "max_context_tokens_observed": 20.0,
            },
        ],
        model_name="xiaomimimo/mimo-v2.5",
    )
    chain_row = next(row for row in metrics["summary"] if row["method_name"] == "macnet_chain")

    assert chain_row["node_scale"] == 4
    assert chain_row["topology_direction_mode"] == "divergent"
    assert chain_row["quality_mean"] == 1.0
