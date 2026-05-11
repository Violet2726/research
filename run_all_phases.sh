#!/bin/bash

set -euo pipefail

if [[ -f .env.local ]]; then
  set -a
  source .env.local
  set +a
fi

run_phase() {
  local phase="$1"
  local reference_state="${2-}"
  export FAITHFUL_PHASE="$phase"
  if [[ -n "$reference_state" ]]; then
    export FAITHFUL_REFERENCE_STATE="$reference_state"
  else
    unset FAITHFUL_REFERENCE_STATE || true
  fi
  uv run python - <<'PY' | tail -n 1
import os

from research_experiments.matrix.faithful_matrix import RuntimeOverrides, assert_matrix_succeeded, run_faithful_matrix

kwargs = {}
reference_state = os.environ.get("FAITHFUL_REFERENCE_STATE")
if reference_state:
    kwargs["reference_state_path_or_root"] = reference_state

run_dir = run_faithful_matrix(
    RuntimeOverrides(phase_name=os.environ["FAITHFUL_PHASE"]),
    **kwargs,
)
assert_matrix_succeeded(run_dir)
print(run_dir.as_posix())
PY
}

echo "开始运行 faithful_matrix 四个阶段..."

echo "[$(date)] 开始运行 smoke20 阶段..."
SMOKE20_DIR="$(run_phase smoke20)"
echo "[$(date)] smoke20 阶段完成: $SMOKE20_DIR"

echo "[$(date)] 开始运行 pilot100 阶段..."
PILOT100_DIR="$(run_phase pilot100 "$SMOKE20_DIR")"
echo "[$(date)] pilot100 阶段完成: $PILOT100_DIR"

echo "[$(date)] 开始运行 confirmatory300 阶段..."
CONFIRM300_DIR="$(run_phase confirmatory300 "$PILOT100_DIR")"
echo "[$(date)] confirmatory300 阶段完成: $CONFIRM300_DIR"

echo "[$(date)] 开始运行 confirmatory500 阶段..."
CONFIRM500_DIR="$(run_phase confirmatory500 "$CONFIRM300_DIR")"
echo "[$(date)] confirmatory500 阶段完成: $CONFIRM500_DIR"

auto_push_flag="${RESEARCH_AUTO_PUSH_CACHE_SNAPSHOT:-}"
if [[ -n "${RESEARCH_CACHE_HF_REPO:-}" ]] && [[ "${auto_push_flag,,}" =~ ^(1|true|yes|on)$ ]]; then
  cache_root="${RESEARCH_CACHE_ROOT:-local/cache}"
  echo "[$(date)] 开始推送 cache 最新快照到 Hugging Face: $cache_root"
  uv run cache_archive_cli push-latest --cache-root "$cache_root" --repo "$RESEARCH_CACHE_HF_REPO" --json >/dev/null
  echo "[$(date)] cache 快照推送完成: $RESEARCH_CACHE_HF_REPO"
fi

echo "[$(date)] 所有阶段运行完成。"
