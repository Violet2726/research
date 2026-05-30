"""`free_mad_lite` family manifest。"""

from research_experiments.families.manifest_helpers import make_family_manifest

MANIFEST = make_family_manifest(
    family_name="free_mad_lite",
    prototype="debate_rounds",
    config_loader_path="research_experiments.families.free_mad_lite.config:load_experiment_config",
    model_resolver_path="research_experiments.families.shared.config_loading:resolve_model",
    runner_path="research_experiments.families.free_mad_lite.run.execute:run_experiment",
    validator_path="research_experiments.families.free_mad_lite.run.validate:validate_run",
    summarizer_path="research_experiments.families.free_mad_lite.run.report:summarize_run",
    report_renderer_path="research_experiments.families.free_mad_lite.run.report:render_report",
    cli_main_path="research_experiments.families.free_mad_lite.spec:main",
    metrics_view_path="metrics.json",
    prediction_records_path="final_predictions.jsonl",
    turn_record_paths=("agent_turns.jsonl", "debate_messages.jsonl"),
    extra_view_paths=("diagnostics.json", "trajectory_scores.jsonl", "paper_summary.csv", "run_summary.json"),
)
