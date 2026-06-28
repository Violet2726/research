# cred_v configs

`cred_v_main.toml` is the stable CRED-RFS mainline. It uses SC-anchored
free-text Stage A reasoning with `cred_rfs_vote_5_anchor` and
`cred_rfs_safe_select_v3`.

`cred_v_rfs_v5_evidence_repair.toml` is the next forward experiment. It keeps
the v3 safe selector as a baseline and adds `cred_rfs_evidence_repair_v5`,
which only expands deterministic Math/Hotpot evidence repair plus the existing
GPQA 3/3 unanimous duel gate.

`cred_v_legacy.toml` keeps the retired verifier and ACS baselines for explicit
failure analysis. Legacy methods are not part of the default mainline.
