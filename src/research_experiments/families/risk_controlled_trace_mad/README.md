# MAD Innovation / historical H-SGSA

H-SGSA v5 is retained only to replay and diagnose its recorded BBEH trajectory. Its candidate board was formed before sample-aware BBEH canonicalization, so the historical positive route is unconfirmable and no live H-SGSA confirmation, Pareto, or SOTA claim is permitted.

The offline normalization-impact audit reads only the recorded Stage-A and resample turns:

`research_cli experiment --family risk_controlled_trace_mad replay-dev --experiment configs/families/risk_controlled_trace_mad/experiments/mad_innovation.toml`

Use the unified report renderer for an existing historical run:

`research_cli experiment --family risk_controlled_trace_mad render-report --run-dir <run-dir>`

The active 100-to-200 gated research line is `disagreement_guided_crux_reconstruction`.
