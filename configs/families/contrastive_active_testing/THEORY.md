# CATCH theoretical contract

The active statements below describe the CATCH-v3 indexed-contrast decoder and
its explicit assumptions. They do not assert that an LLM selector identifies
semantically sound contrasts or that homogeneous witnesses are independent.
The variable-distance discussion later in this file documents the retired
v1/v2 motivation; v3 does not use its threshold grid.

## Target-set barrier and conditional top-two rule

Let `T` be the anchor and at most two selected challengers. Because every v3
output belongs to `T`, the event `{Y_hat=Y}` is a subset of `{Y in T}`. Hence
`P(Y_hat=Y) <= P(Y in T)`. This is sharper than candidate-oracle coverage and
is why both bounds are mandatory gate diagnostics.

Suppose, conditionally, that normalized Stage-A support induces a plug-in
posterior `pi(c)` and the anchor is already included. For any two-challenger
set `S`, target coverage under this posterior is
`pi(anchor) + sum_{c in S} pi(c)`. If `S` omits a challenger `i` with greater
posterior mass than an included `j`, exchanging `j` for `i` cannot decrease
the sum. Repeating the exchange yields the two highest-mass challengers. This
proves conditional optimality under the proxy; it does not prove calibration.

## Frozen three-coordinate repetition code

For a true challenger, classify each of three coordinates as correct support,
an adversarial error supporting the anchor, or an erasure. Let their counts be
`3-e-u`, `e`, and `u`. The panel rule requires at least two challenger supports
and strictly more challenger than anchor support. If `e+u <= 1`, then
`3-e-u >= 2`; moreover, if `e=1` then challenger support is 2 and anchor
support is 1, while if `e=0` anchor support is 0. Both inequalities hold.
Thus one error or one erasure is correctable. If `s` selected coordinates have
an incorrect semantic signature before observation, they are adversarial
errors, giving the sufficient condition `s+e+u <= 1`. The converse is not
claimed: some patterns outside the radius can still decode correctly.

## Candidate barrier

Let `C` be the valid Stage-A candidate set and `Y` the correct answer.  Since CATCH is candidate restricted,

`P(final != Y) = P(Y not in C) + P(Y in C, final != Y)`.

The events are disjoint and exhaustive, so `P(final = Y) <= P(Y in C)`.  No selector improvement can recover a correct
answer that the five shared samples never produced.  Candidate-oracle coverage is therefore a necessary headroom
diagnostic, not an optional upper-bound plot.

## Archived general coding motivation (not the v3 decoder)

For true codeword `h*`, competitor `h`, and their `d` differing effective coordinates, suppose the received vector is
corrupted on `e` of those coordinates.  Each uncorrupted coordinate contributes `+1` to

`dist(w,h) - dist(w,h*)`,

and each corrupted coordinate can contribute no less than `-1`.  Hence

`dist(w,h) - dist(w,h*) >= (d-e)-e = d-2e`.

If `e < d/2`, the true codeword is strictly nearer.  If `s` coordinates contain incorrect candidate commitments before
measurement, treating those as adversarial corruptions gives the sufficient condition `s+e < d/2`.  CATCH computes
distance only on differing coordinates backed by non-overlapping evidence spans, preventing duplicated trace text from
artificially increasing `d`.

## Stochastic and correlated errors

For conditionally bounded coordinate errors with mean at most `eta < 1/2`, Hoeffding/Azuma gives the pairwise tail

`P(error against one competitor) <= exp(-2 d (1/2-eta)^2)`.

A union bound covers the remaining candidate classes.  This result is conditional on the stated bounded-error process;
it is not used as an empirical independence claim.

For two witness false-pass indicators with probabilities `q1`, `q2` and correlation `rho`, the covariance identity is

`P(both false-pass) = q1*q2 + rho*sqrt(q1(1-q1)q2(1-q2))`.

The product reduction holds only when `rho=0`.  The run reports both panels separately so shared-model bias remains
visible.

## Selective net gain

CATCH differs from SC5 only on override events.  Partitioning by the direction of those changes yields

`Acc(CATCH)-Acc(SC5) = P(wrong->correct) - P(correct->wrong)`.

Conditional override precision above one half is equivalent to positive net corrections.  Held-out inference uses a
one-sided exact Clopper-Pearson lower bound, not a normal approximation.

## Archived counting motivation and active failure modes

With at most `q` outcomes per test, `T` tests encode at most `q^T` distinct complete signatures.  Injective separation
of `K` hypotheses therefore requires `T >= ceil(log_q K)`.  Error tolerance requires positive additional Hamming
distance; mere injectivity is insufficient.

CATCH-ICV must abstain or fail when the correct answer is absent from the
target set, the selector cannot form three non-overlapping contrasts, selected
groups are not semantically mutually exclusive or source-decidable, witness
errors are too correlated, structured outputs fail, or more than one
challenger passes. Indexed units establish exact provenance only; the frozen
blind human audit estimates decidability, exclusivity, atomic/context
sufficiency, and leakage. The earlier finite-outcome counting argument is
retained only as the design history of v1/v2; v3's operative guarantee is the
three-coordinate proposition above.
