# cred_mad

CRED-MAD is the Contractual Refutation Evidence Debate family. The current
configuration is the v5 shrink: debate is triggered only for weak/split initial
votes, with one refutation target and strict verified override gating.

Use `cred_v` for the verifier-centric SOTA mainline.

The default entry point is:

```powershell
uv run research_cli experiment --family cred_mad inspect-experiment --experiment configs/families/cred_mad/experiments/cred_mad_main.toml
```
