#!/bin/bash

set -euo pipefail

if [[ -f .env.local ]]; then
  set -a
  source .env.local
  set +a
fi

STATE_ROOT="runs/faithful_matrix_iterative"

run_phase() {
  local phase="$1"
  local reference_state="${2-}"
  export FAITHFUL_PHASE="$phase"
  export FAITHFUL_STATE_ROOT="$STATE_ROOT"
  if [[ -n "$reference_state" ]]; then
    export FAITHFUL_REFERENCE_STATE="$reference_state"
  else
    unset FAITHFUL_REFERENCE_STATE || true
  fi
  uv run python - <<'PY' | tail -n 1
import os

from experiment_core.matrix.faithful_matrix import RuntimeOverrides, run_faithful_matrix

kwargs = {"state_root": os.environ["FAITHFUL_STATE_ROOT"]}
reference_state = os.environ.get("FAITHFUL_REFERENCE_STATE")
if reference_state:
    kwargs["reference_state_path_or_root"] = reference_state

run_dir = run_faithful_matrix(
    RuntimeOverrides(phase_name=os.environ["FAITHFUL_PHASE"]),
    **kwargs,
)
print(run_dir.as_posix())
PY
}

echo "开始运行 faithful_matrix 三个阶段..."

echo "[$(date)] 开始运行 smoke20 阶段..."
SMOKE20_DIR="$(run_phase smoke20)"
echo "[$(date)] smoke20 阶段完成: $SMOKE20_DIR"

echo "[$(date)] 开始运行 pilot100 阶段..."
PILOT100_DIR="$(run_phase pilot100 "$SMOKE20_DIR")"
echo "[$(date)] pilot100 阶段完成: $PILOT100_DIR"

echo "[$(date)] 开始运行 confirmatory300 阶段..."
CONFIRM300_DIR="$(run_phase confirmatory300 "$PILOT100_DIR")"
echo "[$(date)] confirmatory300 阶段完成: $CONFIRM300_DIR"

echo "[$(date)] 所有阶段运行完成。"
