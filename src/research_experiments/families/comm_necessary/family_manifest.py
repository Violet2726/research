"""`comm_necessary` family manifest。"""

from research_experiments.families.manifest_helpers import make_family_manifest

MANIFEST = make_family_manifest(
    family_name="comm_necessary",
    prototype="packet_belief_update",
    config_loader_path="research_experiments.families.comm_necessary.config:load_experiment_config",
    model_resolver_path="research_experiments.families.shared.config_loading:resolve_model",
    runner_path="research_experiments.families.comm_necessary.run.execute:run_experiment",
    validator_path="research_experiments.families.comm_necessary.run.validate:validate_run",
    summarizer_path="research_experiments.families.comm_necessary.run.report:summarize_run",
    report_renderer_path="research_experiments.families.comm_necessary.run.report:render_report",
    cli_main_path="research_experiments.families.comm_necessary.spec:main",
    metrics_view_path="metrics.json",
    prediction_records_path="final_predictions.jsonl",
    turn_record_paths=("stage_a_turns.jsonl", "stage_b_turns.jsonl"),
    extra_view_paths=(
        "sample_views.jsonl",
        "message_packets.jsonl",
        "diagnostics.json",
        "paper_summary.csv",
        "hotpot_predictions",
    ),
)
