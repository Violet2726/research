# SGSA-MAD

SGSA-MAD freezes BRD-MAD V1 as a negative result and tests a risk-controlled
generative synthesis selector. Five Stage-A calls exactly match `sc_5`. Three
blind GSA reviewers share one physical panel: `gsa_quorum_3` applies a 2/3
counterfactual selector while `sgsa_unanimous_3` requires 3/3 support for the
same existing candidate. Novel answers remain shadow-only.

Every reviewer receives all candidate headers and final answers. The 6000-character
board budget is divided evenly across candidate rationales, retaining both the
beginning and conclusion of every truncated rationale. Reviewer visible reasoning
is limited to one sentence of at most 40 words and the request ceiling is 8192
tokens. The `count100_seed42` gate requires zero final request/protocol
failures and prevents full runs when safety or paired-accuracy conditions fail.

All datasets use the repository-wide split contract: `countN_seed42` is the
same seed-42 prefix for every N, and `fullN_seed42` is the complete dataset.
The `count100_seed42` run uses this split for both Omni-MATH-2 and BBEH;
`count20_seed42` is contained in both count100 sets.
BBEH uses micro accuracy for every `countN_seed42` run and official task harmonic
accuracy only for `full4520_seed42`.

```powershell
uv run research_cli experiment --family selective_gsa_mad inspect-experiment --experiment configs/families/selective_gsa_mad/experiments/sgsa_mad.toml
uv run research_cli experiment --family selective_gsa_mad render-report --run-dir local/runs/selective_gsa_mad/sgsa_mad/count100_seed42/<run-id>
```
