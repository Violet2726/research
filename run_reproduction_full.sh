#!/bin/bash

set -euo pipefail

REPRO_MAX_CONCURRENT_REQUESTS=80
REPRO_REQUESTS_PER_MINUTE_LIMIT=95
REPRO_TOKENS_PER_MINUTE_LIMIT=1000000

if [[ -f .env.local ]]; then
  set -a
  source .env.local
  set +a
fi

export REPRO_MAX_CONCURRENT_REQUESTS
export REPRO_REQUESTS_PER_MINUTE_LIMIT
export REPRO_TOKENS_PER_MINUTE_LIMIT

echo "开始运行 reproduction_matrix full 阶段..."
echo "使用限流: max_concurrent_requests=$REPRO_MAX_CONCURRENT_REQUESTS, requests_per_minute_limit=$REPRO_REQUESTS_PER_MINUTE_LIMIT, tokens_per_minute_limit=$REPRO_TOKENS_PER_MINUTE_LIMIT"

uv run python - <<'PY' | tail -n 1
import os

from research_experiments.matrix.faithful_matrix import RuntimeOverrides, assert_matrix_succeeded, run_matrix

run_dir = run_matrix(
    "reproduction",
    RuntimeOverrides(
        phase_name="full",
        max_concurrent_requests=int(os.environ["REPRO_MAX_CONCURRENT_REQUESTS"]),
        requests_per_minute_limit=int(os.environ["REPRO_REQUESTS_PER_MINUTE_LIMIT"]),
        tokens_per_minute_limit=int(os.environ["REPRO_TOKENS_PER_MINUTE_LIMIT"]),
    ),
)
assert_matrix_succeeded(run_dir)
print(run_dir.as_posix())
PY
