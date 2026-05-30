"""`multi_agent` family manifest。"""

from research_experiments.families.manifest_helpers import make_family_manifest

MANIFEST = make_family_manifest(
    family_name="multi_agent",
    prototype="debate_rounds",
    config_loader_path="research_experiments.families.multi_agent.config:load_experiment_config",
    model_resolver_path="research_experiments.families.shared.config_loading:resolve_model",
    runner_path="research_experiments.families.multi_agent.run.execute:run_experiment",
    validator_path="research_experiments.families.multi_agent.run.validate:validate_run",
    summarizer_path="research_experiments.families.multi_agent.run.report:summarize_run",
    report_renderer_path="research_experiments.families.multi_agent.run.report:render_report",
    cli_main_path="research_experiments.families.multi_agent.spec:main",
    metrics_view_path="metrics.json",
    prediction_records_path="final_predictions.jsonl",
    turn_record_paths=("agent_turns.jsonl", "debate_messages.jsonl"),
    extra_view_paths=("cost_breakdown.json", "debate_diagnostics.json", "run_summary.json"),
)
