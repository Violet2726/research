# D4 sealed-data contracts

This document defines the input contracts for data that may eventually enter a
D4 calibration or confirmation run. The current repository contains loaders and
validators, but no real confirmation asset or activation evidence.

All stages use the sole executable protocol `catch_kernel_d4_mainline_v3`
(`protocols/catch_kernel_d4_v3.toml`). The completed compiler smoke is frozen as
`failed_blocking_downstream`, so selection inspection, provider audit, the
300-record validation, calibration, and confirmation currently stop before any
model request.

## Common rules

- Calibration, human-audit, and confirmation rows must be split before the
  method is frozen and before the confirmation text/gold is disclosed.
- A manifest contains IDs, strata, and hashes only. It must not contain
  question text, options, latent graphs, or gold answers.
- Every manifest is hash-linked to the source asset and the independent audit
artifacts. The runner recomputes these links and recomputes per-row hashes
when it materializes the sealed confirmation split.
- The independent 300-record tagged validation uses the project-wide validated
  response cache. Exact provider/model/dataset/prompt/seed identities may be
  reused across experiments; interruption recovery remains bound to the same
  run ledger. Protocol and selection hashes remain frozen in the manifest.
- All linked paths must be relative to the manifest directory. Absolute paths
  and `..` traversal are rejected.
- `development`, `human_audit`, and `confirmation` counts and strata must be
  written into the experiment TOML before `sealed_data_ready=true` is enabled.
  Counts are not inferred from the manifest after the fact.

## Independent output-protocol validation split

The post-selection tagged-text validation is a separate 300-record development
artifact, not calibration or confirmation data. It uses exactly 100 records
each from `bbeh_extension`, `musr_x`, and `supergpqa_science`, all under the
manifest split name `protocol_validation`. The blocked executable template is
`configs/families/contrastive_active_testing/experiments/catch_kernel_d4_protocol_independent_validation_tagged_v3.toml`.

Before enabling that template, the custodian must install all three source
assets and manifests, fill the exact counts/strata and selection SHA-256 values,
and set `sealed_data_ready=true`. The runner loads the complete source asset,
selects only the signed manifest IDs, recomputes question/source/latent hashes,
requires the selected count to equal the preregistered limit, and rejects any
post-sealing exclusion split. The resulting certificate additionally binds the
source run, provider audit, sealed-manifest hashes, and exact 100/100/100
selection hashes.

After installing the three assets and manifests, but before enabling the run,
compute the values to freeze without any model calls:

```powershell
research_cli experiment --family contrastive_active_testing `
  inspect-kernel-d4-protocol-validation-selection `
  --experiment configs/families/contrastive_active_testing/experiments/catch_kernel_d4_protocol_independent_validation_tagged_v3.toml `
  --output local/audits/catch_kernel_d4_protocol_validation_selection.json
```

Copy the emitted `selection_hashes` into the experiment TOML, review the
hash-only artifact, and only then set `sealed_data_ready=true`. Immediately
before the formal validation run, refresh `kernel-d4-provider-audit`.

The later one-shot confirmation is a distinct split and asset namespace. Its
three executable benchmark placeholders are
`configs/core/shared/benchmarks/d4_confirmation/bbeh_extension.toml`,
`musr_x.toml`, and `supergpqa_science.toml`; it never falls back to the public
BBEH, MuSR, or GPQA development pools and never reuses the protocol-validation
split as confirmation data.

## BBEH-extension records

Use loader `bbeh_extension_jsonl`. Each record must contain:

```json
{
  "record_id": "stable-custodian-id",
  "task": "frozen-task-name",
  "input": "question text with its answer contract",
  "target": "gold answer",
  "provenance": {
    "source_id": "upstream-or-generator-id",
    "source_sha256": "64 lowercase hex characters"
  }
}
```

`record_id` is unique, `task` is non-empty, and the provenance hash is
mandatory. The loader stores a canonical source-record hash in metadata; the
manifest stores that hash without exposing the record.

The data custodian must also provide a duplicate/near-duplicate audit against
the already inspected BBEH pools and disclose the generator, source license,
and any transformed upstream material.

## MuSR-X records

Use loader `musr_x_jsonl`. Each record must contain a unique `record_id`, one
of `murder_mysteries`, `object_placements`, or `team_allocation`, a unique
64-hex `latent_graph_sha256`, a narrative, a question, at least two unique
choices, and an in-range `answer_index`.

The v3 latent-first manifest additionally hashes and links:

- the rendered narrative/question/gold asset;
- the independent rendering and quality-validation audit;
- the pinned generator commit and generation lock;
- the quality-validation protocol.

The v2 manifest is intentionally rejected because it did not bind the rendered
asset to the latent partition.

## SuperGPQA Science records

Use `supergpqa_science_jsonl` or `supergpqa_science_parquet` on a prefiltered
asset. The official dataset exposes `uuid`, `question`, `options`, `answer`,
`answer_letter`, `discipline`, `field`, `subfield`, `difficulty`, and
`is_calculation`; the loader requires `discipline=Science`,
`field` in `{Physics, Chemistry, Biology}`, text-only questions, and a consistent
answer letter/text pair. The official schema can be inspected in the
[SuperGPQA dataset card](https://huggingface.co/datasets/m-a-p/SuperGPQA).

The custodian must hash and report exact and MinHash/near-duplicate screening
against GPQA, SciBench, and any other disclosed source datasets. This is a
duplicate screen, not a proof that model-training contamination is absent.

## Sealing workflow

The family CLI exposes two custodian-side commands:

- `seal-kernel-d4-text-data` for BBEH-extension and SuperGPQA Science;
- `seal-kernel-d4-latent-data` for MuSR-X.

Both commands require a counts JSON file, frozen strata, protocol hash inputs,
an independent custodian ID, and linked audit files. They fail closed if the
generated manifest cannot be recomputed immediately from its source files.
