# Literature and novelty audit

The active novelty boundary is CATCH-ICV, not the retired generated-question
mechanism. It replaces free diagnostic questions, free outcomes, copied
quotes, and offsets with a selector that can emit only program-generated
evidence IDs. This choice follows the empirical warning that solving ability
does not imply good information-query generation (QuestBench), that verifier
decomposition can itself be misaligned (Lu et al., ACL 2025,
<https://aclanthology.org/2025.acl-long.254/>), and that under practical test-
time budgets additional solving may outperform generative verification
(*When To Solve, When To Verify*, <https://arxiv.org/abs/2504.01005>).

The principal matched objection is tested by PairJudge-3: it receives exactly
the same anchor-plus-top-two target set and the same three-call envelope. If it
matches or exceeds CATCH-ICV, local active measurement is not reported as a
positive mechanism contribution. Independent side permutations address known
pairwise position bias (Shi et al., IJCNLP 2025,
<https://aclanthology.org/2025.ijcnlp-long.18/>). CATCH-ICV also differs from
external-evidence fact-checking systems such as GAVEL
(<https://aclanthology.org/2026.findings-acl.1789/>): it is training-free,
tool-free, candidate-restricted, same-model, and fixed-budget.

CATCH is positioned against primary sources rather than generic “multi-agent” comparisons:

- Self-consistency: Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in Language Models*, ICLR 2023,
  <https://openreview.net/pdf?id=1PL1NIMMrw>.
- Test-time compute allocation: Snell et al., *Scaling LLM Test-Time Compute Optimally can be More Effective than
  Scaling Model Parameters*, ICLR 2025, <https://openreview.net/pdf?id=4FWAwZtd2n>.
- Debate reliability: Smit et al., *Should We Be Going MAD?*, ICML 2024,
  <https://proceedings.mlr.press/v235/smit24a.html>.
- Homogeneous debate diversity/confidence: Zhu et al., ACL Findings 2026,
  <https://aclanthology.org/2026.findings-acl.1694/>.
- Critical-divergence auditing: *AgentAuditor*, <https://arxiv.org/abs/2602.09341>.
- Masked/backward verification: *ProCo*, <https://arxiv.org/abs/2405.14092>, and *FOBAR*,
  <https://aclanthology.org/2024.findings-acl.397/>.
- Contrastive verification questions: *TOOLVERIFIER*, <https://aclanthology.org/2024.findings-emnlp.289.pdf>.
- Question acquisition limitation: *QuestBench*,
  <https://deepmind.google/research/publications/questbench-can-llms-ask-the-right-question-to-acquire-information-in-reasoning-tasks/>.
- Active hypothesis testing and experimental design: <https://arxiv.org/abs/1901.06795> and
  <https://proceedings.mlr.press/v139/katz-samuels21a.html>.
- Chernoff-style active hypothesis testing is a design motivation, not an
  optimality claim: <https://proceedings.mlr.press/v151/mukherjee22a.html>.
- Error-correcting output codes: <https://arxiv.org/abs/cs/9501101>.
- Selective classification and conformal risk control:
  <https://proceedings.neurips.cc/paper_files/paper/7073-selective-classification-for-deep-neural-networks.pdf> and
  <https://openreview.net/forum?id=33XGfHLtZg>.

The nearest components exist separately: candidate auditing, decomposed
verification, pairwise comparison, active hypothesis testing, repetition
codes, and selective abstention. The registered v3 novelty claim is narrower:
a training-free black-box composition in which the program first exposes only
indexed, non-leaking trace units; a selector may choose IDs but cannot generate
measurements; witnesses never see candidate identities, votes, answers, or
full traces; and a fixed candidate-restricted repetition decoder—not an LLM
judge—controls overrides. DirectJudge-3 and target-matched PairJudge-3 test the
main complexity objection. Machine leakage checks, independent left/right
permutations, panel agreement, and a blind semantic audit test whether the
claimed information isolation and signature assumptions actually hold.

The generated-question, generated-outcome, max-min codebook, `first-four`,
`signature-shuffle`, and unblinded/vote-aware ablations belong to the frozen
v1/v2 mechanism. They are not v3 components and cannot be revived after the
one-shot v3 preflight begins.
