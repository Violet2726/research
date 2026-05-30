"""`imad` family manifest。"""

from research_experiments.families.manifest_helpers import make_family_manifest

MANIFEST = make_family_manifest(
    family_name="imad",
    prototype="debate_rounds",
    config_loader_path="research_experiments.families.imad.config:load_experiment_config",
    model_resolver_path="research_experiments.families.shared.config_loading:resolve_model",
    runner_path="research_experiments.families.imad.run.execute:run_experiment",
    validator_path="research_experiments.families.imad.run.validate:validate_run",
    summarizer_path="research_experiments.families.imad.run.report:summarize_run",
    report_renderer_path="research_experiments.families.imad.run.report:render_report",
    cli_main_path="research_experiments.families.imad.spec:main",
    metrics_view_path="metrics.json",
    prediction_records_path="final_predictions.jsonl",
    turn_record_paths=("agent_turns.jsonl", "debate_messages.jsonl", "round_diagnostics.jsonl"),
    extra_view_paths=("cost_breakdown.json", "stability_diagnostics.json", "run_summary.json"),
)
