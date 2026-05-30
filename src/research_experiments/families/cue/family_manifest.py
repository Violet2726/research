"""`cue` family manifest。"""

from research_experiments.families.manifest_helpers import make_family_manifest

MANIFEST = make_family_manifest(
    family_name="cue",
    prototype="shared_stage_policy",
    config_loader_path="research_experiments.families.cue.config:load_experiment_config",
    model_resolver_path="research_experiments.families.shared.config_loading:resolve_model",
    runner_path="research_experiments.families.cue.run.execute:run_experiment",
    validator_path="research_experiments.families.cue.run.validate:validate_run",
    summarizer_path="research_experiments.families.cue.run.report:summarize_run",
    report_renderer_path="research_experiments.families.cue.run.report:render_report",
    cli_main_path="research_experiments.families.cue.spec:main",
    metrics_view_path="policy_metrics.json",
    prediction_records_path="policy_predictions.jsonl",
    turn_record_paths=(
        "stage_a_turns.jsonl",
        "communication_turns.jsonl",
        "audit_turns.jsonl",
        "control_turns.jsonl",
    ),
    extra_view_paths=("policy_diagnostics.json", "oracle_trigger_eval.json"),
)
