# CATCH theoretical contract

The statements below describe the deterministic decoder and its explicit assumptions.  They do not assert that an LLM
designer produces semantically sound commitments or that homogeneous witnesses are independent.

## Candidate barrier

Let `C` be the valid Stage-A candidate set and `Y` the correct answer.  Since CATCH is candidate restricted,

`P(final != Y) = P(Y not in C) + P(Y in C, final != Y)`.

The events are disjoint and exhaustive, so `P(final = Y) <= P(Y in C)`.  No selector improvement can recover a correct
answer that the five shared samples never produced.  Candidate-oracle coverage is therefore a necessary headroom
diagnostic, not an optional upper-bound plot.

## Deterministic decoding radius

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

## Counting bound and failure modes

With at most `q` outcomes per test, `T` tests encode at most `q^T` distinct complete signatures.  Injective separation
of `K` hypotheses therefore requires `T >= ceil(log_q K)`.  Error tolerance requires positive additional Hamming
distance; mere injectivity is insufficient.

CATCH must abstain or fail when the correct answer is absent, candidate signatures are indistinguishable, commitments
are semantically unsound, witness errors are too correlated, structured outputs fail, or more than one challenger
passes.  Exact trace offsets establish provenance only; the frozen human audit estimates semantic entailment.

