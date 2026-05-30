"""`sid_lite` family manifest。"""

from research_experiments.families.manifest_helpers import make_family_manifest

MANIFEST = make_family_manifest(
    family_name="sid_lite",
    prototype="packet_belief_update",
    config_loader_path="research_experiments.families.sid_lite.config:load_experiment_config",
    model_resolver_path="research_experiments.families.shared.config_loading:resolve_model",
    runner_path="research_experiments.families.sid_lite.run.execute:run_experiment",
    validator_path="research_experiments.families.sid_lite.run.validate:validate_run",
    summarizer_path="research_experiments.families.sid_lite.run.report:summarize_run",
    report_renderer_path="research_experiments.families.sid_lite.run.report:render_report",
    cli_main_path="research_experiments.families.sid_lite.spec:main",
    metrics_view_path="metrics.json",
    prediction_records_path="final_predictions.jsonl",
    turn_record_paths=("stage_a_turns.jsonl", "message_packets.jsonl", "belief_updates.jsonl"),
    extra_view_paths=("diagnostics.json", "paper_summary.csv", "run_summary.json"),
)
