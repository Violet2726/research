#!/usr/bin/env bash

set -euo pipefail

if [[ -f .env.local ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env.local
  set +a
fi

initial_reference_state="${RESEARCH_INITIAL_REFERENCE_STATE:-}"

phases=("$@")
if [[ ${#phases[@]} -eq 0 ]]; then
  # phases=(count20 count100 count300)
  phases=(count20 count100)
fi

run_phase() {
  local phase="$1"
  local reference_state="${2-}"
  local cli_args=(research_cli matrix run --matrix reproduction --phase "$phase")
  if [[ -n "$reference_state" ]]; then
    cli_args+=(--reference-state-path "$reference_state")
  fi

  local run_dir
  run_dir="$(uv run "${cli_args[@]}" | tail -n 1)"
  uv run research_cli matrix assert-success --state-path "$run_dir" --json >/dev/null
  printf '%s\n' "$run_dir"
}

push_cache_if_needed() {
  local auto_push_flag="${RESEARCH_AUTO_PUSH_CACHE_SNAPSHOT:-}"
  if [[ -z "${RESEARCH_CACHE_HF_REPO:-}" ]] || [[ ! "${auto_push_flag,,}" =~ ^(1|true|yes|on)$ ]]; then
    return
  fi

  local cache_root="${RESEARCH_CACHE_ROOT:-local/cache}"
  echo "[$(date)] 开始推送 cache 最新快照到 Hugging Face: $cache_root"
  uv run research_cli tools cache-archive push-latest \
    --cache-root "$cache_root" \
    --repo "$RESEARCH_CACHE_HF_REPO" \
    --json >/dev/null
  echo "[$(date)] cache 快照推送完成: $RESEARCH_CACHE_HF_REPO"
}

echo "开始运行 reproduction_matrix 阶段序列..."

previous_run_dir="$initial_reference_state"
for phase in "${phases[@]}"; do
  echo "[$(date)] 开始运行 $phase 阶段..."
  if [[ -n "$previous_run_dir" ]]; then
    previous_run_dir="$(run_phase "$phase" "$previous_run_dir")"
  else
    previous_run_dir="$(run_phase "$phase")"
  fi
  echo "[$(date)] $phase 阶段完成: $previous_run_dir"
done

push_cache_if_needed

echo "[$(date)] 所有阶段运行完成。"
