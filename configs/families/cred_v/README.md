# cred_v configs

`cred_v_main.toml` is the stable CRED-RFS mainline. It uses SC-anchored
free-text Stage A reasoning with `cred_rfs_vote_5_anchor` and
`cred_rfs_repair_only_v6`. v6 keeps deterministic Math/Hotpot repair and
removes GPQA/MMLU/StrategyQA semantic promotion from the forward path.

`cred_v_rfs_v3_pairwise_ablation.toml` retains the former GPQA 3/3 unanimous
pairwise selector. It is now an ablation/negative-evidence entry because the
latest count300 replication showed the signal was not stable.

`cred_v_rfs_v7_shadow_evidence_select.toml` logs cross-view evidence selector
signals without changing final predictions. Shadow rules must pass
pre-registered precision and harm gates on count100 and count300 before any
future promotion experiment is proposed.

`cred_v_rfs_v8_repair_bank.toml` is the next forward candidate. It keeps the
v6 final-answer discipline and only adds scorer-safe Math repair, HotpotQA
context-supported span repair, and deterministic multiple-choice option-text
format repair.

`cred_v_rfs_v9_certificate_shadow.toml` is shadow-only. It records cross-view
certificate signals for weak-split MC/StrategyQA samples, but its final
prediction remains the v8/v6 deterministic-repair answer.

`cred_v_error_bucket_audit_v1.toml` is a diagnostic entry for producing
`diagnostics/error_bucket_audit.json`, which groups wrong-majority, oracle-gap,
repair, and shadow-harm samples for mechanism discovery.

`cred_v_rfs_v5_evidence_repair.toml` is retained as a negative/ablation
entry. The count300 audit showed `math_equivalence_repair_v2` harmed
scorer-canonical ASCII `pi` answers, so that mode is disabled in the protocol.
Forward runs should treat v6 as the stable baseline unless a new
pre-registered repair gate is added.

`cred_v_legacy.toml` keeps the retired verifier and ACS baselines for explicit
failure analysis. Legacy methods are not part of the default mainline.
