"""`single_agent` family manifest。"""

from research_experiments.families.manifest_helpers import make_family_manifest

MANIFEST = make_family_manifest(
    family_name="single_agent",
    prototype="independent_sampling",
    config_loader_path="research_experiments.families.single_agent.config:load_experiment_config",
    model_resolver_path="research_experiments.families.shared.config_loading:resolve_model",
    runner_path="research_experiments.families.single_agent.run.execute:run_experiment",
    validator_path="research_experiments.families.single_agent.run.validate:validate_run",
    summarizer_path="research_experiments.families.single_agent.run.report:summarize_run",
    report_renderer_path="research_experiments.families.single_agent.run.report:render_report",
    cli_main_path="research_experiments.families.single_agent.spec:main",
    metrics_view_path="metrics.json",
    prediction_records_path="predictions.jsonl",
    turn_record_paths=("raw_responses.jsonl",),
    extra_view_paths=("paper_tables.md", "run_summary.json"),
)
