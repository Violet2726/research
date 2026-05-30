"""`econ` family manifest。"""

from research_experiments.families.manifest_helpers import make_family_manifest

MANIFEST = make_family_manifest(
    family_name="econ",
    prototype="packet_belief_update",
    config_loader_path="research_experiments.families.econ.config:load_experiment_config",
    model_resolver_path="research_experiments.families.shared.config_loading:resolve_model",
    runner_path="research_experiments.families.econ.run.execute:run_experiment",
    validator_path="research_experiments.families.econ.run.validate:validate_run",
    summarizer_path="research_experiments.families.econ.run.report:summarize_run",
    report_renderer_path="research_experiments.families.econ.run.report:render_report",
    cli_main_path="research_experiments.families.econ.spec:main",
    metrics_view_path="metrics.json",
    prediction_records_path="final_predictions.jsonl",
    turn_record_paths=(
        "agent_turns.jsonl",
        "communication_trace.jsonl",
        "belief_trace.jsonl",
        "equilibrium_trace.jsonl",
    ),
)
