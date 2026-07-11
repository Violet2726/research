# BRD-MAD research mainline

## Status and scope

`adaptive_sparse_mad`, `cred_mad`, and `cred_v` are frozen historical mechanisms and controls. Their code and archived results remain reproducible, but no V10/V11-style iteration should be added there. The active innovation family is `blind_reconstructive_mad` (BRD-MAD).

This change is evidence-led. Earlier runs show that vanilla MAD produces only a small gain over the strongest SC baseline while consuming roughly three times the tokens; A-SMAD and CRED-MAD have not produced validated corrections; and CRED-V's candidate oracle greatly exceeds its selected answer. Thus the causal bottleneck is safe use of a correct minority, not merely generating more candidates.

## BRD-MAD hypothesis

BRD-MAD tests blind review--reconstruct debate under a fixed Stage-A budget:

1. Five `sc_5`-aligned free-CoT samples create candidate families.
2. A five-way agreement exits immediately.
3. For disagreement, reviewers see one anonymous representative per family, with random labels and hidden support counts.
4. Three mutually invisible reviewers falsify candidates and reconstruct from scratch.
5. Only existing candidates are eligible. A 4-1 split needs 3/3 minority review support; other patterns need 2/3. New answers are retained only as shadows.

The target decomposition is:

`ΔAcc = P(anchor wrong and correct override) − P(anchor correct and wrong override)`.

The IID quorum expression `3e² − 2e³` is reported only as a reference; the implementation estimates actual reviewer error correlation, effective reviewer count, and realized quorum error.

## Pre-registered experiment flow

1. Run unit tests and a fake-provider smoke test.
2. Run Qwen-Flash pilot on the shared `count100_seed42` prefixes of Omni-MATH-2 Filtered and BBEH. `count20_seed42` is a prefix of this pilot split.
3. Continue only if there are zero request/protocol failures, both primary sets show at least a 3pp candidate-oracle gap, BRD has net positive corrections versus both `sc_5` and `conditional_resample_3`, at least 20 overrides, and override precision is at least 2/3.
4. Freeze prompt/configuration/split hashes, then run Qwen-Flash, Qwen-Turbo, and MiMo on `full1000_seed42` Omni-MATH-2 Filtered, `full4520_seed42` BBEH, and `full198_seed42` GPQA Diamond. Qwen uses 1000 RPM/1000 concurrency; MiMo is limited to 18 RPM/8 concurrency.
5. Do not inspect locked per-method accuracy until all locked runs complete.

The baseline family contains matched `cot_1`, `sc_3`, `sc_5`, `mad_3a_r1`, `mad_3a_r2`, and `mad_5a_r1` configurations for the pilot and locked suites. Qwen-Turbo is run by overriding the Qwen configuration's model reference; MiMo uses the explicit 18-RPM configuration.

## Reporting rule

For Omni-MATH-2 use exact accuracy; for BBEH use task harmonic mean as primary and micro accuracy as secondary; GPQA is transfer continuity only. Reports include paired 10,000-sample bootstrap CIs, McNemar tests, and Holm correction. The phrase “fixed-backbone, fixed-reasoning-budget method SOTA” is allowed only when BRD has positive corrected 95% CIs over conditional resampling on both primary sets, non-negative point estimates for all three backbones, and top accuracy without exceeding the strongest competitor's average token use.
