# CATCH-v3 / CATCH-ICV preregistration

Status: final mechanism attempt, registered before the first `catch-dev-v3`
intervention response.  A structural-preflight or dev100 failure permanently
retires this line; there is no v4 prompt or decoder repair.

## Scope and estimand

CATCH-ICV studies one fixed MiMo-v2.5 endpoint with thinking disabled, five
shared homogeneous Stage-A samples, and at most three intervention calls.  It
does not claim cross-model generalization or global SOTA.  The estimand is the
paired accuracy and actual-token difference relative to adaptive-SC8 under the
frozen dev100/heldout200 manifests.

The immutable zero-network replay must precede every v3 structural preflight.
Its production implementation must report candidate-oracle minus SC5 at least
5 percentage points, target-oracle minus SC5 at least 8 points, and target-
oracle minus adaptive-SC8 at least 5 points.  The registered v1 replay is
43/45/61/58 percent for SC5/adaptive/candidate oracle/target oracle, with 83
answer-class disagreements.

## Mechanism

Only valid answers from the sample-aware answer contract enter voting.  The
plurality answer is the anchor; stable hashing breaks ties.  At most the two
highest-support challenger classes form pair-local anonymous targets.  Each
class exposes one content-hash-selected representative reasoning trace.

The runner deterministically segments normalized reasoning into indexed local
units.  The selector can emit only pair IDs and one-to-three consecutive unit
IDs per side.  It emits no question, outcome, quote, offset, explanation,
candidate identity, vote count, or answer label.  A challenger is eligible
only with exactly three non-overlapping, non-leaking coordinates.  Otherwise
CATCH abstains after the selector and does not call witnesses.

Two blinded witnesses receive the option-free source and anonymous left/right
statements.  Coordinate order and sides are independently permuted.  The only
verdicts are `LEFT_ONLY`, `RIGHT_ONLY`, `BOTH`, and `NEITHER`; the latter two
are erasures.  Unknown, duplicate, missing, or malformed coordinate rows erase
that coordinate, while an unrecoverable top-level object fails the panel.

For each panel and challenger, let `n_c` and `n_a` be decisive coordinates
supporting challenger and anchor.  The frozen pass rule is `n_c >= 2` and
`n_c > n_a`.  The same unique challenger must pass both panels.  Every other
state returns the anchor.  No generated answer can enter the output set.

## Baselines and gates

Development and heldout run adaptive-SC8, DirectJudge-3, PairJudge-3, and
CATCH-ICV.  PairJudge sees exactly the anchor-plus-top-two target set and is the
primary complexity control.  All three baselines have three post-Stage-A
calls when disagreement triggers.

The structural-preflight command always terminates after its 20 selected
disagreement samples.  It never continues into dev100.  Machine thresholds are
100% selector top-level parsing, 100% ID/ownership validity, zero automatic
leakage, at least 60% three-coordinate coverage; then 100% witness top-level
parsing, at least 95% valid rows, 80% decisive verdicts, 90% usable dual-panel
pairs, and 70% inverse-permutation agreement.

Forty accepted coordinates are sampled with seed 42 for two annotators blind
to gold, votes, and candidate answers.  After adjudication, decidability,
mutual exclusivity, and atomic/context sufficiency must each be at least 90%,
answer leakage must be zero, and Cohen's kappa must be at least 0.6 except for
the zero-base-rate leakage item. The runner verifies the exact 40 coordinate
hashes and recomputes rates from adjudicated item-level labels; the reported
kappa pools the decidable, mutually-exclusive, and atomic binary judgments.

Dev100 requires zero terminal failures, at least 99.5% structured parsing,
40% code-packet coverage, the replay headroom conditions, CATCH at least 3
points above adaptive-SC8 and 2 points above the stronger judge, at least three
net corrections, at least 65% override precision, and no larger mean actual
token count than adaptive-SC8.  No prompt, coordinate count, threshold, or
decoder grid is searched.

Heldout200 is one shot after an immutable protocol freeze.  It requires BBEH
task-harmonic CATCH at least 2 points above adaptive-SC8, micro CATCH no lower
than either judge, corrected greater than harmed, one-sided exact 95% override
precision lower bound above 0.5, and no larger actual-token cost.  Failure
stops confirmation and yields a mechanism-boundary result.

## Formal claims

Let `T` be the anchor plus at most two challengers.  Any target-restricted
selector satisfies `P(Y_hat = Y) <= P(Y in T)`.  If Stage-A vote shares are
treated as a plug-in posterior, choosing the two highest-share challengers
maximizes target coverage under a two-challenger budget; this is conditional
on that proxy and is not a calibration claim.

For a reliable three-coordinate challenger signature, `e + u <= 1` (one
observation error or erasure) is sufficient for the fixed panel rule.  With
`s` semantic signature errors the sufficient condition is `s + e + u <= 1`.
The blind human audit therefore measures a theorem assumption rather than an
auxiliary presentation metric.

The two same-model panels are not assumed independent.  If their false-pass
probabilities are `q1,q2` and Bernoulli correlation is `rho`, their joint
false-pass probability is
`q1*q2 + rho*sqrt(q1(1-q1)q2(1-q2))`.  Reports must include empirical panel
correlation and side-swap agreement.

Finally, because CATCH changes SC5 only on override,
`Acc(CATCH)-Acc(SC5) = P(wrong->correct)-P(correct->wrong)`.  Override
precision above one half is equivalent to positive net corrections.

## Run discipline

The provider audit is live and cache-bypassed.  All other commands are finite
one-shot processes that write terminal `progress.json`, `run_validation.json`,
termination reason, and archive integrity before exiting.  No process polls,
wakes, or continues this conversation.  The user reports completion before
artifacts are read.
