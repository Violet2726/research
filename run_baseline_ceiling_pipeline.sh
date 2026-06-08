#!/usr/bin/env bash

set -euo pipefail

if [[ -f .env.local ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env.local
  set +a
fi

export PYTHONUTF8="${PYTHONUTF8:-1}"

MODEL_REF="${MODEL_REF:-xiaomimimo/mimo-v2.5}"
SELECTION_DIR="${BASELINE_SELECTION_DIR:-local/reports/single_agent/baseline_ceiling_selection}"
REFERENCE_AUDIT_DIR="${BASELINE_REFERENCE_AUDIT_DIR:-local/reports/single_agent/baseline_ceiling_reference_audit}"
SUMMARY_DIR="${BASELINE_SUMMARY_DIR:-local/reports/single_agent/baseline_ceiling_summary}"
RUN_REFERENCE_AUDIT="${BASELINE_RUN_REFERENCE_AUDIT:-1}"

SCREEN_CURRENT="configs/families/single_agent/experiments/baseline_ceiling_v1_current_prompt.toml"
SCREEN_UNIFIED="configs/families/single_agent/experiments/baseline_ceiling_v1_unified_control.toml"
SCREEN_ZERO="configs/families/single_agent/experiments/baseline_ceiling_v1_zero_shot_cot.toml"

KNOWN_REFERENCE_RUN_DIRS=(
  "local/runs/single_agent/same_context_core_benchmarks/count100/20260531T133337Z-xiaomimimo-mimo-v2.5"
  "local/runs/single_agent/same_context_main_table/count100/20260531T204302Z-xiaomimimo-mimo-v2.5"
  "local/runs/adaptive_sparse_mad/same_context_competition_math_stage_a_v4/count100/20260608T024322Z-xiaomimimo-mimo-v2.5"
  "local/runs/adaptive_sparse_mad/same_context_full_counterfactual_v1/count100/20260608T065426Z-xiaomimimo-mimo-v2.5"
)

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

run_experiment() {
  local experiment_path="$1"
  local phase_name="$2"
  log "开始运行 ${phase_name} | ${experiment_path}"
  local run_dir
  run_dir="$(uv run research_cli experiment --family single_agent run --experiment "$experiment_path" --phase "$phase_name" --model "$MODEL_REF" | tail -n 1)"
  log "完成 ${phase_name} | ${experiment_path} -> ${run_dir}"
  printf '%s\n' "$run_dir"
}

log "Baseline ceiling pipeline 启动，模型：${MODEL_REF}"

screen_current_run="$(run_experiment "$SCREEN_CURRENT" "count20")"
screen_unified_run="$(run_experiment "$SCREEN_UNIFIED" "count20")"
screen_zero_run="$(run_experiment "$SCREEN_ZERO" "count20")"

if [[ "${RUN_REFERENCE_AUDIT,,}" =~ ^(1|true|yes|on)$ ]]; then
  reference_audit_args=()
  for run_dir in "${KNOWN_REFERENCE_RUN_DIRS[@]}"; do
    if [[ -d "$run_dir" ]]; then
      reference_audit_args+=(--run-dir "$run_dir")
    fi
  done
  if [[ ${#reference_audit_args[@]} -gt 0 ]]; then
    log "开始生成 reference audit"
    uv run python -m research_experiments.families.single_agent.ceiling_audit reference-audit \
      "${reference_audit_args[@]}" \
      --output-dir "$REFERENCE_AUDIT_DIR"
    log "reference audit 完成：${REFERENCE_AUDIT_DIR}"
  else
    log "未发现本地 reference runs，跳过 reference audit"
  fi
fi

log "开始执行 count20 screening 选择"
selection_json="$(uv run python -m research_experiments.families.single_agent.ceiling_audit select-screening \
  --run-dir "$screen_current_run" \
  --run-dir "$screen_unified_run" \
  --run-dir "$screen_zero_run" \
  --output-dir "$SELECTION_DIR" | tail -n 1)"
log "screening 选择完成：${selection_json}"

mapfile -t count100_experiments < <(
  uv run python - "$selection_json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for row in payload.get("generated_count100_configs", []):
    print(row["experiment_config"])
PY
)

if [[ ${#count100_experiments[@]} -eq 0 ]]; then
  log "筛选结果没有生成 count100 配置，停止执行"
  exit 1
fi

optimized_run_dirs=()
for experiment_path in "${count100_experiments[@]}"; do
  optimized_run_dirs+=("$(run_experiment "$experiment_path" "count100")")
done

reference_summary_args=()
for run_dir in "${KNOWN_REFERENCE_RUN_DIRS[@]}"; do
  if [[ -d "$run_dir" ]]; then
    reference_summary_args+=(--reference-run-dir "$run_dir")
  fi
done

if [[ ${#reference_summary_args[@]} -gt 0 ]]; then
  optimized_summary_args=()
  for run_dir in "${optimized_run_dirs[@]}"; do
    optimized_summary_args+=(--optimized-run-dir "$run_dir")
  done
  log "开始生成 ceiling summary"
  uv run python -m research_experiments.families.single_agent.ceiling_audit summarize-ceiling \
    "${reference_summary_args[@]}" \
    "${optimized_summary_args[@]}" \
    --selection-json "$selection_json" \
    --output-dir "$SUMMARY_DIR"
  log "ceiling summary 完成：${SUMMARY_DIR}"
else
  log "未发现本地 reference runs，跳过 ceiling summary"
fi

log "Baseline ceiling pipeline 全部完成"
