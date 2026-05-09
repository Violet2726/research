#!/bin/bash

set -a && source .env.local && set +a

echo "开始运行 faithful_matrix 三个阶段..."

# 阶段 1: smoke20 (无需参考)
echo "[$(date)] 开始运行 smoke20 阶段..."
SMOKE20_DIR=$(uv run python -c "from experiment_core.matrix.faithful_matrix import RuntimeOverrides, run_faithful_matrix; run_dir = run_faithful_matrix(RuntimeOverrides(phase_name='smoke20'), state_root=r'runs/faithful_matrix_iterative'); print(run_dir.as_posix())" 2>/dev/null | tail -1)
echo "[$(date)] smoke20 阶段完成: $SMOKE20_DIR"

# 阶段 2: pilot100 (参考 smoke20)
echo "[$(date)] 开始运行 pilot100 阶段..."
PILOT100_DIR=$(uv run python -c "from experiment_core.matrix.faithful_matrix import RuntimeOverrides, run_faithful_matrix; run_dir = run_faithful_matrix(RuntimeOverrides(phase_name='pilot100'), state_root=r'runs/faithful_matrix_iterative', reference_state_path_or_root=r'$SMOKE20_DIR'); print(run_dir.as_posix())" 2>/dev/null | tail -1)
echo "[$(date)] pilot100 阶段完成: $PILOT100_DIR"

# 阶段 3: confirmatory300 (参考 pilot100)
echo "[$(date)] 开始运行 confirmatory300 阶段..."
CONFIRM300_DIR=$(uv run python -c "from experiment_core.matrix.faithful_matrix import RuntimeOverrides, run_faithful_matrix; run_dir = run_faithful_matrix(RuntimeOverrides(phase_name='confirmatory300'), state_root=r'runs/faithful_matrix_iterative', reference_state_path_or_root=r'$PILOT100_DIR'); print(run_dir.as_posix())" 2>/dev/null | tail -1)
echo "[$(date)] confirmatory300 阶段完成: $CONFIRM300_DIR"

echo "[$(date)] 所有阶段运行完成！"
