"""`macnet` family manifest。"""

from research_experiments.families.manifest_helpers import make_family_manifest

MANIFEST = make_family_manifest(
    family_name="macnet",
    prototype="topology_or_graph",
    config_loader_path="research_experiments.families.macnet.config:load_experiment_config",
    model_resolver_path="research_experiments.families.shared.config_loading:resolve_model",
    runner_path="research_experiments.families.macnet.run.execute:run_experiment",
    validator_path="research_experiments.families.macnet.run.validate:validate_run",
    summarizer_path="research_experiments.families.macnet.run.report:summarize_run",
    report_renderer_path="research_experiments.families.macnet.run.report:render_report",
    cli_main_path="research_experiments.families.macnet.spec:main",
    metrics_view_path="metrics.json",
    prediction_records_path="final_predictions.jsonl",
    turn_record_paths=("artifact_trace.jsonl", "instruction_trace.jsonl"),
    extra_view_paths=("topology_manifest.json", "scaling_summary.json"),
)
