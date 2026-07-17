# Literature and novelty audit

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
- Error-correcting output codes: <https://arxiv.org/abs/cs/9501101>.
- Selective classification and conformal risk control:
  <https://proceedings.neurips.cc/paper_files/paper/7073-selective-classification-for-deep-neural-networks.pdf> and
  <https://openreview.net/forum?id=33XGfHLtZg>.

The nearest components exist separately: candidate auditing, masked verification, contrastive questions, active test
selection, error-correcting codes, and abstention.  The registered novelty claim is their training-free black-box
composition in which candidate traces commit to a finite code, witnesses never see candidate-side information, a
program chooses tests by effective code distance, and a deterministic candidate-restricted decoder—not an LLM judge—
controls overrides.  DirectJudge-3, blindness leakage, vote leakage, first-four, single-witness, and signature-shuffle
controls are predeclared to test whether each claimed component matters.

