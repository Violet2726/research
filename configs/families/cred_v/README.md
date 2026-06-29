# cred_v configs

`cred_v_main.toml` is the stable CRED-RFS mainline. It uses SC-anchored
free-text Stage A reasoning with `cred_rfs_vote_5_anchor` and
`cred_rfs_safe_select_v3`.

`cred_v_rfs_v5_evidence_repair.toml` is retained as a negative/ablation
entry. The count300 audit showed `math_equivalence_repair_v2` harmed
scorer-canonical ASCII `pi` answers, so that mode is disabled in the protocol.
Forward runs should treat v3 as the stable baseline unless a new pre-registered
repair gate is added.

`cred_v_legacy.toml` keeps the retired verifier and ACS baselines for explicit
failure analysis. Legacy methods are not part of the default mainline.
