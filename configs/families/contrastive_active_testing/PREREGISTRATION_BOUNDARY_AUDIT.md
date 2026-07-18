# CATCH-v3 cross-domain mechanism-boundary audit

This study is a post-failure exploration, not a confirmatory gate. The frozen
BBEH v3 structural preflight failed at 75% coordinate validity, 40% eligible
packet coverage, and one automated answer leak; its raw packet-coverage upper
bound was 55%. It cannot be rerun, retuned, or rescued by this audit.

The audit uses the unchanged CATCH-v3 selector, validator, two-witness packet,
and fixed 2-of-3 decoder on BBEH, MuSR, seqBench, and GPQA Diamond. Each dataset
has a deterministic 100-item Stage-A screening pool. Up to twenty Stage-A
disagreements are selected without gold and receive adaptive-SC8, CATCH-ICV,
DirectJudge-3, and target-matched PairJudge-3. Results are conditional on
disagreement and may not be presented as benchmark-level headline accuracy.

MuSR is allocated 34/33/33 across murder, object-placement, and team-allocation
domains. GPQA uses largest-remainder allocation by high-level domain with a
minimum of one item per nonempty domain. seqBench allocates by `(B,N)` cell with
a minimum of one item, then interleaves logical-depth deciles. BBEH reuses the
failed v3 preflight's fixed seed-42 twenty-item manifest.

The physical-attempt cap is 3,000 including retries and is separate from the
62,000-attempt confirmation budget. Dataset, prompt, schema, configuration,
screening, and disagreement-selection hashes are recorded before intervention
results are interpreted. No heldout or confirmation command is dispatched.

Interpretation is preregistered as follows: low eligibility indicates contrast
formation failure; adequate eligibility with low decisive/agreement rates
indicates witness measurement failure; corrected not exceeding harmed indicates
measurement/correctness misalignment; PairJudge matching or exceeding CATCH
denies a complexity benefit. A signal limited to one dataset is a task boundary,
and even a multi-dataset signal remains exploratory.
