from __future__ import annotations

from pathlib import Path

from research_experiments.matrix.reproduction_analysis import build_reproduction_analysis, render_reproduction_analysis
from research_experiments.reporting.reproduction_landscape import build_reproduction_landscape_payload, render_reproduction_landscape
from research_experiments.reporting.reproduction_package import build_reproduction_package_payload, render_reproduction_package
from research_experiments.workspace.layout import default_reports_root
from testsupport.filesystem import write_json


def test_build_reproduction_package_payload_splits_canonical_auxiliary_and_scaling() -> None:
    analysis = {
        "matrix_id": "reproduction",
        "phase_name": "count100",
        "model_ref": "xiaomimimo/mimo-v2.5",
        "counts": {"completed": 3},
        "entries": [
            {
                "family": "dog_graph",
                "experiment_name": "dog_graph_main",
                "track_name": "graph_reasoning",
                "entry_role": "canonical",
                "analysis_mode": "primary_summary",
                "overall_row": {
                    "family": "dog_graph",
                    "experiment_name": "dog_graph_main",
                    "track_name": "graph_reasoning",
                    "entry_role": "canonical",
                    "primary_method_name": "dog_graph_paper",
                    "primary_metric_label": "accuracy",
                    "primary_metric_value": 0.77,
                    "total_tokens_mean": 3200.0,
                    "calls_per_question_mean": 5.0,
                },
            },
            {
                "family": "dog_graph",
                "experiment_name": "dog_graph_static_ablation",
                "track_name": "graph_reasoning",
                "entry_role": "ablation",
                "analysis_mode": "primary_summary",
                "overall_row": {
                    "family": "dog_graph",
                    "experiment_name": "dog_graph_static_ablation",
                    "track_name": "graph_reasoning",
                    "entry_role": "ablation",
                    "primary_method_name": "dog_graph_r1",
                    "primary_metric_label": "accuracy",
                    "primary_metric_value": 0.42,
                    "total_tokens_mean": 900.0,
                    "calls_per_question_mean": 2.0,
                },
            },
            {
                "family": "macnet",
                "experiment_name": "macnet_scaling_study",
                "track_name": "topology_collaboration",
                "entry_role": "scaling",
                "analysis_mode": "scaling_summary",
                "overall_row": None,
                "scaling_summary": {
                    "series": [
                        {
                            "method_name": "macnet_random",
                            "topology_direction_mode": "divergent",
                            "scales": [{"node_scale": 4, "quality_mean": 0.6, "total_tokens_mean": 1800.0}],
                        }
                    ]
                },
            },
        ],
    }

    package = build_reproduction_package_payload(analysis)

    assert package["canonical_board"][0]["experiment_name"] == "dog_graph_main"
    assert package["auxiliary_board"][0]["experiment_name"] == "dog_graph_static_ablation"
    assert package["scaling_sections"][0]["experiment_name"] == "macnet_scaling_study"


def test_build_reproduction_landscape_payload_groups_only_within_track() -> None:
    analysis = {
        "matrix_id": "reproduction",
        "phase_name": "count100",
        "model_ref": "xiaomimimo/mimo-v2.5",
        "counts": {"completed": 2},
        "entries": [
            {
                "family": "table_critic",
                "experiment_name": "table_critic_main",
                "track_name": "table_reasoning",
                "entry_role": "canonical",
                "overall_row": {
                    "family": "table_critic",
                    "experiment_name": "table_critic_main",
                    "track_name": "table_reasoning",
                    "entry_role": "canonical",
                    "primary_method_name": "table_critic_paper",
                    "primary_metric_label": "accuracy",
                    "primary_metric_value": 0.895,
                    "total_tokens_mean": 5583.06,
                },
            },
            {
                "family": "macnet",
                "experiment_name": "macnet_paper_main",
                "track_name": "topology_collaboration",
                "entry_role": "canonical",
                "overall_row": {
                    "family": "macnet",
                    "experiment_name": "macnet_paper_main",
                    "track_name": "topology_collaboration",
                    "entry_role": "canonical",
                    "primary_method_name": "macnet_random",
                    "primary_metric_label": "quality",
                    "primary_metric_value": 0.64,
                    "total_tokens_mean": 2100.0,
                },
            },
        ],
    }

    payload = build_reproduction_landscape_payload(analysis)

    assert "table_reasoning" in payload["track_boards"]
    assert "topology_collaboration" in payload["track_boards"]
    assert "global_total_board" not in payload


def test_render_reproduction_outputs_markdown_and_json(tmp_path: Path) -> None:
    state_dir = tmp_path / "matrix_state_demo"
    run_dir = tmp_path / "dog_graph_run"
    run_dir.mkdir(parents=True)
    write_json(
        state_dir / "state.json",
        {
            "matrix_id": "reproduction",
            "matrix_kind": "reproduction_matrix",
            "overrides": {"phase_name": "count100", "model_ref": "xiaomimimo/mimo-v2.5"},
            "counts": {"completed": 1, "semantic_unique_targets": 1},
            "semantic_entries": [
                {
                    "family": "dog_graph",
                    "config_path": "configs/families/dog_graph/experiments/dog_graph_main.toml",
                    "experiment_name": "dog_graph_main",
                    "status": "completed",
                    "run_dir": run_dir.as_posix(),
                },
                {
                    "family": "macnet",
                    "config_path": "configs/families/macnet/experiments/macnet_scaling_study.toml",
                    "experiment_name": "macnet_scaling_study",
                    "status": "running",
                    "run_dir": None,
                }
            ],
            "entries": [
                {
                    "family": "dog_graph",
                    "config_path": "configs/families/dog_graph/experiments/dog_graph_main.toml",
                    "experiment_name": "dog_graph_main",
                    "status": "completed",
                    "run_dir": run_dir.as_posix(),
                },
                {
                    "family": "macnet",
                    "config_path": "configs/families/macnet/experiments/macnet_scaling_study.toml",
                    "experiment_name": "macnet_scaling_study",
                    "status": "running",
                    "run_dir": None,
                }
            ],
        },
    )
    write_json(
        run_dir / "metrics.json",
        {
            "summary": [
                {
                    "dataset": "overall",
                    "method_name": "tog_iterative_baseline",
                    "accuracy_mean": 0.75,
                    "total_tokens_mean": 4000.0,
                },
                {
                    "dataset": "overall",
                    "method_name": "dog_graph_paper",
                    "accuracy_mean": 0.77,
                    "total_tokens_mean": 4800.0,
                },
            ]
        },
    )
    write_json(run_dir / "scaling_summary.json", {"series": []})

    published_analysis = tmp_path / "published" / "analysis.md"
    published_package = tmp_path / "published" / "package.md"
    published_landscape = tmp_path / "published" / "landscape.md"
    default_analysis_path = Path(default_reports_root("reproduction_matrix")) / f"{state_dir.name}-reproduction.md"
    default_package_path = Path(default_reports_root("reproduction_matrix")) / f"{state_dir.name}-reproduction_package.md"
    default_landscape_path = Path(default_reports_root("reproduction_matrix")) / f"{state_dir.name}-reproduction_landscape.md"

    assert not default_analysis_path.exists()
    assert not default_package_path.exists()
    assert not default_landscape_path.exists()

    analysis_paths = render_reproduction_analysis(state_dir, output_root=state_dir, published_path=published_analysis)
    package_paths = render_reproduction_package(state_dir, output_root=state_dir, published_path=published_package)
    landscape_paths = render_reproduction_landscape(state_dir, output_root=state_dir, published_path=published_landscape)

    assert Path(analysis_paths["json_path"]).exists()
    assert Path(package_paths["package_json"]).exists()
    assert Path(landscape_paths["json_path"]).exists()
    assert (state_dir / "reproduction_analysis.json").exists()
    assert (state_dir / "reproduction_package.json").exists()
    assert (state_dir / "reproduction_landscape.json").exists()
    assert not default_analysis_path.exists()
    assert not default_package_path.exists()
    assert not default_landscape_path.exists()
