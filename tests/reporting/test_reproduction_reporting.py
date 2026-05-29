from __future__ import annotations

from pathlib import Path

from testsupport.filesystem import write_json

from research_experiments.matrix.reproduction_analysis import render_reproduction_analysis
from research_experiments.reporting.reproduction_landscape import (
    build_reproduction_landscape_payload,
    render_reproduction_landscape,
)
from research_experiments.reporting.reproduction_package import (
    build_reproduction_package_payload,
    render_reproduction_package,
)
from research_experiments.workspace.layout import default_reports_root


def test_build_reproduction_package_payload_splits_canonical_auxiliary_and_scaling() -> None:
    analysis = {
        "matrix_id": "reproduction",
        "phase_name": "count100",
        "model_ref": "xiaomimimo/mimo-v2.5",
        "counts": {"completed": 3},
        "entries": [
            {
                "family": "colmad",
                "experiment_name": "colmad_realmistake_main",
                "track_name": "oversight_protocol",
                "entry_role": "canonical",
                "analysis_mode": "primary_summary",
                "overall_row": {
                    "family": "colmad",
                    "experiment_name": "colmad_realmistake_main",
                    "track_name": "oversight_protocol",
                    "entry_role": "canonical",
                    "primary_method_name": "colmad_collaborative",
                    "primary_metric_label": "accuracy",
                    "primary_metric_value": 0.85,
                    "total_tokens_mean": 3200.0,
                    "calls_per_question_mean": 5.0,
                },
            },
            {
                "family": "dmad",
                "experiment_name": "dmad_reasoning_main",
                "track_name": "same_context",
                "entry_role": "canonical",
                "analysis_mode": "primary_summary",
                "overall_row": {
                    "family": "dmad",
                    "experiment_name": "dmad_reasoning_main",
                    "track_name": "same_context",
                    "entry_role": "canonical",
                    "primary_method_name": "dmad_cot_sbp_pot",
                    "primary_metric_label": "accuracy",
                    "primary_metric_value": 0.78,
                    "total_tokens_mean": 2800.0,
                    "calls_per_question_mean": 4.0,
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

    assert package["canonical_board"][0]["experiment_name"] == "colmad_realmistake_main"
    assert package["canonical_board"][1]["experiment_name"] == "dmad_reasoning_main"
    assert package["scaling_sections"][0]["experiment_name"] == "macnet_scaling_study"


def test_build_reproduction_landscape_payload_groups_only_within_track() -> None:
    analysis = {
        "matrix_id": "reproduction",
        "phase_name": "count100",
        "model_ref": "xiaomimimo/mimo-v2.5",
        "counts": {"completed": 2},
        "entries": [
            {
                "family": "colmad",
                "experiment_name": "colmad_realmistake_main",
                "track_name": "oversight_protocol",
                "entry_role": "canonical",
                "overall_row": {
                    "family": "colmad",
                    "experiment_name": "colmad_realmistake_main",
                    "track_name": "oversight_protocol",
                    "entry_role": "canonical",
                    "primary_method_name": "colmad_collaborative",
                    "primary_metric_label": "accuracy",
                    "primary_metric_value": 0.85,
                    "total_tokens_mean": 3200.0,
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

    assert "oversight_protocol" in payload["track_boards"]
    assert "topology_collaboration" in payload["track_boards"]
    assert "global_total_board" not in payload


def test_render_reproduction_outputs_markdown_and_json(tmp_path: Path) -> None:
    state_dir = tmp_path / "matrix_state_demo"
    run_dir = tmp_path / "colmad_run"
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
                    "family": "colmad",
                    "config_path": "configs/families/colmad/experiments/colmad_realmistake_main.toml",
                    "experiment_name": "colmad_realmistake_main",
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
                    "family": "colmad",
                    "config_path": "configs/families/colmad/experiments/colmad_realmistake_main.toml",
                    "experiment_name": "colmad_realmistake_main",
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
                    "method_name": "colmad_collaborative",
                    "accuracy_mean": 0.85,
                    "total_tokens_mean": 3200.0,
                },
                {
                    "dataset": "overall",
                    "method_name": "colmad_independent",
                    "accuracy_mean": 0.78,
                    "total_tokens_mean": 2800.0,
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
