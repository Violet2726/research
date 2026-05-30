"""`madjudge` family manifest。"""

from research_experiments.families.manifest_helpers import make_family_manifest

MANIFEST = make_family_manifest(
    family_name="madjudge",
    prototype="debate_rounds",
    config_loader_path="research_experiments.families.madjudge.config:load_experiment_config",
    model_resolver_path="research_experiments.families.shared.config_loading:resolve_model",
    runner_path="research_experiments.families.madjudge.run.execute:run_experiment",
    validator_path="research_experiments.families.madjudge.run.validate:validate_run",
    summarizer_path="research_experiments.families.madjudge.run.report:summarize_run",
    report_renderer_path="research_experiments.families.madjudge.run.report:render_report",
    cli_main_path="research_experiments.families.madjudge.spec:main",
    metrics_view_path="metrics.json",
    prediction_records_path="predictions.jsonl",
    turn_record_paths=("turns.jsonl", "debate_messages.jsonl"),
    extra_view_paths=("cost_breakdown.json", "debate_diagnostics.json"),
)
