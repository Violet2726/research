"""`budget_comm` family manifest。"""

from research_experiments.families.manifest_helpers import make_family_manifest

MANIFEST = make_family_manifest(
    family_name="budget_comm",
    prototype="packet_belief_update",
    config_loader_path="research_experiments.families.budget_comm.config:load_experiment_config",
    model_resolver_path="research_experiments.families.shared.config_loading:resolve_model",
    runner_path="research_experiments.families.budget_comm.run.execute:run_experiment",
    validator_path="research_experiments.families.budget_comm.run.validate:validate_run",
    summarizer_path="research_experiments.families.budget_comm.run.report:summarize_run",
    report_renderer_path="research_experiments.families.budget_comm.run.report:render_report",
    cli_main_path="research_experiments.families.budget_comm.spec:main",
    metrics_view_path="metrics.json",
    prediction_records_path="final_predictions.jsonl",
    turn_record_paths=("stage_a_turns.jsonl",),
    extra_view_paths=(
        "sample_views.jsonl",
        "candidate_packets.jsonl",
        "auction_decisions.jsonl",
        "belief_updates.jsonl",
        "budget_diagnostics.json",
        "paper_summary.csv",
    ),
)
