"""`colmad` family manifest。"""

from research_experiments.families.manifest_helpers import make_family_manifest

MANIFEST = make_family_manifest(
    family_name="colmad",
    prototype="topology_or_graph",
    config_loader_path="research_experiments.families.colmad.config:load_experiment_config",
    model_resolver_path="research_experiments.families.shared.config_loading:resolve_model",
    runner_path="research_experiments.families.colmad.run.execute:run_experiment",
    validator_path="research_experiments.families.colmad.run.validate:validate_run",
    summarizer_path="research_experiments.families.colmad.run.report:summarize_run",
    report_renderer_path="research_experiments.families.colmad.run.report:render_report",
    cli_main_path="research_experiments.families.colmad.spec:main",
    metrics_view_path="metrics.json",
    prediction_records_path="final_predictions.jsonl",
    turn_record_paths=("debate_trace.jsonl", "judge_trace.jsonl"),
    extra_view_paths=("protocol_diagnostics.json", "run_summary.json"),
)
