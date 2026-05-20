#!/bin/bash

set -euo pipefail

eval "$(
  uv run python - <<'PY'
from research_experiments.core.execution.rate_limits import standard_runtime_limits

limits = standard_runtime_limits()
print(f"REPRO_MAX_CONCURRENT_REQUESTS={limits['max_concurrent_requests']}")
print(f"REPRO_REQUESTS_PER_MINUTE_LIMIT={limits['requests_per_minute_limit']}")
print(f"REPRO_TOKENS_PER_MINUTE_LIMIT={limits['tokens_per_minute_limit']}")
PY
)"

if [[ -f .env.local ]]; then
  set -a
  source .env.local
  set +a
fi

run_phase() {
  local phase="$1"
  local reference_state="${2-}"
  export REPRO_PHASE="$phase"
  export REPRO_MAX_CONCURRENT_REQUESTS
  export REPRO_REQUESTS_PER_MINUTE_LIMIT
  export REPRO_TOKENS_PER_MINUTE_LIMIT
  if [[ -n "$reference_state" ]]; then
    export REPRO_REFERENCE_STATE="$reference_state"
  else
    unset REPRO_REFERENCE_STATE || true
  fi
  uv run python - <<'PY' | tail -n 1
import os

from research_experiments.matrix.faithful_matrix import RuntimeOverrides, assert_matrix_succeeded, run_matrix

kwargs = {}
reference_state = os.environ.get("REPRO_REFERENCE_STATE")
if reference_state:
    kwargs["reference_state_path_or_root"] = reference_state

run_dir = run_matrix(
    "reproduction",
    RuntimeOverrides(
        phase_name=os.environ["REPRO_PHASE"],
        max_concurrent_requests=int(os.environ["REPRO_MAX_CONCURRENT_REQUESTS"]),
        requests_per_minute_limit=int(os.environ["REPRO_REQUESTS_PER_MINUTE_LIMIT"]),
        tokens_per_minute_limit=int(os.environ["REPRO_TOKENS_PER_MINUTE_LIMIT"]),
    ),
    **kwargs,
)
assert_matrix_succeeded(run_dir)
print(run_dir.as_posix())
PY
}

echo "开始运行 reproduction_matrix 三个阶段..."
echo "使用限流: max_concurrent_requests=$REPRO_MAX_CONCURRENT_REQUESTS, requests_per_minute_limit=$REPRO_REQUESTS_PER_MINUTE_LIMIT, tokens_per_minute_limit=$REPRO_TOKENS_PER_MINUTE_LIMIT"

echo "[$(date)] 开始运行 count20 阶段..."
COUNT20_DIR="$(run_phase count20)"
echo "[$(date)] count20 阶段完成: $COUNT20_DIR"

echo "[$(date)] 开始运行 count100 阶段..."
COUNT100_DIR="$(run_phase count100 "$COUNT20_DIR")"
echo "[$(date)] count100 阶段完成: $COUNT100_DIR"

echo "[$(date)] 开始运行 count300 阶段..."
COUNT300_DIR="$(run_phase count300 "$COUNT100_DIR")"
echo "[$(date)] count300 阶段完成: $COUNT300_DIR"

auto_push_flag="${RESEARCH_AUTO_PUSH_CACHE_SNAPSHOT:-}"
if [[ -n "${RESEARCH_CACHE_HF_REPO:-}" ]] && [[ "${auto_push_flag,,}" =~ ^(1|true|yes|on)$ ]]; then
  cache_root="${RESEARCH_CACHE_ROOT:-local/cache}"
  echo "[$(date)] 开始推送 cache 最新快照到 Hugging Face: $cache_root"
  uv run cache_archive_cli push-latest --cache-root "$cache_root" --repo "$RESEARCH_CACHE_HF_REPO" --json >/dev/null
  echo "[$(date)] cache 快照推送完成: $RESEARCH_CACHE_HF_REPO"
fi

echo "[$(date)] 所有阶段运行完成。"
