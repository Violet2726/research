"""从两个 count300_seed42 run 拟合并冻结 RCTA 全局路由。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from research_experiments.families.risk_controlled_trace_mad.algorithms import validate_feature_vector
from research_experiments.families.risk_controlled_trace_mad.router import build_router_artifact

COMPETITOR_METHODS = (
    "cot_1",
    "sc_3",
    "sc_5",
    "sc_7",
    "sc_9",
    "adaptive_sc_9",
    "gsa_trace_1",
    "mad_5a_r1",
    "confidence_mad_5a_r1",
)


def records_from_runs(run_dirs: list[str | Path]) -> tuple[list[dict[str, Any]], list[str]]:
    if len(run_dirs) != 2:
        raise ValueError("Exactly two count300 run directories are required.")
    records: list[dict[str, Any]] = []
    hashes: list[str] = []
    observed_models: set[str] = set()
    for raw_dir in run_dirs:
        root = Path(raw_dir)
        manifest_path = root / "manifest.json"
        predictions_path = root / "views" / "predictions.jsonl"
        validation_path = root / "run_validation.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(manifest.get("phase_name") or manifest.get("phase")) != "count300_seed42":
            raise ValueError(f"Router fitting only accepts count300_seed42 runs: {root}")
        model_name = str((manifest.get("resolved_model") or manifest.get("backbone") or {}).get("name") or "")
        if not model_name:
            raise ValueError(f"Run is missing resolved model name: {root}")
        observed_models.add(model_name)
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if (
            validation.get("passed") is not True
            or int(validation.get("request_failures") or 0) != 0
            or int(validation.get("protocol_failures") or 0) != 0
            or int(validation.get("schema_failures") or 0) != 0
        ):
            raise ValueError(f"Router fitting requires a clean, validated count300 run: {root}")
        rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        method_rows = {
            (str(row["dataset"]), str(row["sample_id"]), str(row["method_name"])): row
            for row in rows
        }
        shadow = [row for row in rows if row.get("method_name") == "rcta_certificate_shadow_1"]
        for row in shadow:
            vector = row.get("feature_vector")
            if not isinstance(vector, dict):
                if row.get("triggered"):
                    raise ValueError(f"Missing feature vector for triggered sample {row.get('sample_id')}")
                continue
            validate_feature_vector(vector)
            anchor_score = float(row.get("initial_vote_score") or 0.0)
            synthesis_score = float(row.get("synthesis_score") or 0.0)
            key = (str(row["dataset"]), str(row["sample_id"]))
            sc9 = method_rows.get((*key, "sc_9"))
            gsa = method_rows.get((*key, "gsa_trace_1"))
            if sc9 is None or gsa is None:
                raise ValueError(f"Missing paired sc_9/gsa_trace_1 row for {key}")
            competitors = {}
            for method_name in COMPETITOR_METHODS:
                paired = method_rows.get((*key, method_name))
                if paired is not None:
                    competitors[method_name] = {
                        "score": float(paired.get("score") or 0.0),
                        "tokens": float(paired.get("total_tokens_per_question") or 0.0),
                    }
            records.append(
                {
                    "dataset": key[0],
                    "model_name": model_name,
                    "sample_id": key[1],
                    "feature_vector": vector,
                    "gain_label": int(anchor_score < 1.0 and synthesis_score == 1.0),
                    "harm_label": int(anchor_score == 1.0 and synthesis_score < 1.0),
                    "anchor_score": anchor_score,
                    "synthesis_score": synthesis_score,
                    "sc9_score": float(sc9.get("score") or 0.0),
                    "gsa_score": float(gsa.get("score") or 0.0),
                    "rcta_tokens": float(row.get("total_tokens_per_question") or 0.0),
                    "competitors": competitors,
                }
            )
        hashes.append(_run_hash(manifest_path, predictions_path, validation_path))
    if len(observed_models) != 2:
        raise ValueError("Router fitting requires two distinct backbone models.")
    cells = {(item["dataset"], item["model_name"]) for item in records}
    if len(cells) != 4:
        raise ValueError(f"Expected exactly four dataset×backbone cells, got {sorted(cells)}")
    return records, hashes


def fit_and_write(run_dirs: list[str | Path], output: str | Path) -> dict[str, Any]:
    records, hashes = records_from_runs(run_dirs)
    artifact = build_router_artifact(records, input_run_hashes=hashes)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifact


def _run_hash(manifest_path: Path, predictions_path: Path, validation_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(manifest_path.read_bytes())
    digest.update(predictions_path.read_bytes())
    digest.update(validation_path.read_bytes())
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit the frozen RCTA global risk router.")
    parser.add_argument("--run-dir", action="append", required=True, help="Repeat exactly twice for Qwen and MiMo count300 runs.")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps(fit_and_write(args.run_dir, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
