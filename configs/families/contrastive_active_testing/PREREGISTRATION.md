# CATCH v1 preregistration

## Fixed inference protocol

- Model: MiMo-v2.5, `thinking.disabled`, temperature 0.7, top-p 1; seeds are recorded observations, not determinism claims.
- Shared Stage-A: five solver calls, cap 16,384.  Invalid canonical answers do not vote or trigger.
- Adaptive comparator: three independent resamples on the same valid disagreement event.
- CATCH configuration: one designer plus two blinded witnesses, role cap 4,096.  The development grid shares the one
  designer and runs one witness pair for each `d_min`; every evaluated grid cell is still charged its own designer plus
  matching witness pair as exactly three logical intervention calls.
- DirectJudge-3: three label-permuted candidate-restricted judges on development and held-out only.
- Cost endpoint: actual reported tokens per question.  Retry attempts are separately capped and reported.

## Development selection

The grid is `d_min in {2,3,4}` and `margin in {1,2}`.  A global configuration, never a per-sample configuration, is
chosen by maximum corrected-minus-harmed subject to precision at least 65%, net corrections at least 3, and selected
code-packet coverage at least 40%.  Ties use higher precision, higher `d_min`, higher margin, then stable SHA-256.  The
CLI freezes only a gate-passing, validation-passing candidate.

Development hard gates additionally require zero terminal request failures, structured parse at least 99.5%, candidate
oracle micro headroom at least 5pp over SC5, CATCH micro at least 3pp over adaptive-SC8 and 2pp over DirectJudge-3, and
mean actual tokens no greater than adaptive-SC8.  Task harmonic is reported but not used for development feasibility.

## Held-out and stopping rule

Held-out uses the immutable decoder file and `catch-heldout-v1`.  It requires the inherited quality gates, BBEH task
harmonic at least 2pp over adaptive-SC8, micro accuracy no lower than DirectJudge-3, corrected greater than harmed, the
one-sided exact 95% override-precision lower bound above 0.5, and no token excess.  Any failure stops confirmation.

Once a held-out response exists, no prompt, parser, validator, threshold, split, or configuration may be changed and
re-run as the same study.  A negative result is reported as a mechanism boundary rather than repaired against held-out
labels.

## Human audit and confirmation

Two annotators, blinded to gold and support counts, independently label 100 seed-42 coordinates for source
decidability, span-to-commitment entailment, final-answer leakage, and outcome coverage.  Confirmation requires
adjudicated decidability and entailment at least 90%, leakage at most 5%, and a recorded agreement statistic.

Confirmation is hard-capped at 62,000 real network attempts including retries.  The registered BBEH remainder contains
4,220 samples.  BBEH task harmonic is primary; 10,000 task-stratified paired bootstrap replicates, exact paired McNemar,
Holm correction, exact override precision, and paired token bootstrap are reported.  No cross-model or global-SOTA
claim is permitted by this registration.

