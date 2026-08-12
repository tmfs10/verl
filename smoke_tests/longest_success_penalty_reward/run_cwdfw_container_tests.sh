#!/usr/bin/env bash
set -euo pipefail

# Run grouped-reward and validation-isolation tests in the exact live image.

SSH_TARGET="${SSH_TARGET:-dfw}"
ACCOUNT="${ACCOUNT:-nemotron_reason_code}"
PARTITION="${PARTITION:-interactive}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
LOCAL_VERL_ROOT="${LOCAL_VERL_ROOT:-/home/siddjain/workspace/verl/verl_main}"
REMOTE_STAGE_BASE="${REMOTE_STAGE_BASE:-/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/setup/longest_success_penalty_reward}"
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/siddjain/containers/verl_vllm012_flashattn_20260321.sqsh}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-/home/siddjain/data/smoke_tests/longest_success_penalty_reward/container_tests_$RUN_TAG}"
REMOTE_VERL_ROOT="$REMOTE_STAGE_BASE/container_tests_$RUN_TAG/verl_main"

[[ "$PARTITION" == interactive ]] || { echo "[error] tests require interactive" >&2; exit 2; }
mkdir -p "$LOCAL_LOG_DIR"

ssh "$SSH_TARGET" "mkdir -p '$REMOTE_VERL_ROOT'"
rsync -a \
  --exclude='.git/' \
  --exclude='.pytest_cache/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  "$LOCAL_VERL_ROOT/" "$SSH_TARGET:$REMOTE_VERL_ROOT/"

remote_command=$(cat <<'EOF'
set -euo pipefail
cd /workspace/verl
export PYTHONPATH=/workspace/verl:/workspace/verl/recipe
export WANDB_MODE=disabled
python -m pytest -q \
  tests/workers/reward_manager/test_batch_reward_shortest_success_on_cpu.py \
  tests/workers/reward_manager/test_batch_reward_uniform_outcome_on_cpu.py \
  tests/trainer/test_main_ppo_validation_reward_config_on_cpu.py
EOF
)

printf '%s\n' "$remote_command" > "$LOCAL_LOG_DIR/container_command.sh"
remote_command_b64=$(printf '%s' "$remote_command" | base64 -w0)
ssh "$SSH_TARGET" bash -s -- \
  "$ACCOUNT" "$PARTITION" "$TIME_LIMIT" "longest-reward-tests-$RUN_TAG" \
  "$CONTAINER" "$REMOTE_VERL_ROOT" "$remote_command_b64" 2>&1 <<'REMOTE' | tee "$LOCAL_LOG_DIR/pytest.log"
set -euo pipefail
account=$1
partition=$2
time_limit=$3
job_name=$4
container=$5
remote_verl_root=$6
command_b64=$7

srun \
  --account="$account" \
  --partition="$partition" \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:8 \
  --exclusive \
  --time="$time_limit" \
  --job-name="$job_name" \
  --container-image="$container" \
  --container-mounts="$remote_verl_root:/workspace/verl" \
  --no-container-remap-root \
  bash -lc "$(printf '%s' "$command_b64" | base64 -d)" &
srun_pid=$!

job_id=""
for _ in $(seq 1 120); do
  job_id=$(squeue -h -u "$USER" -n "$job_name" -o '%A' | head -n 1)
  [[ -n "$job_id" ]] && break
  kill -0 "$srun_pid" 2>/dev/null || break
  sleep 0.25
done
if [[ -z "$job_id" ]]; then
  wait "$srun_pid" || true
  echo "[error] could not resolve Slurm job ID" >&2
  exit 1
fi
scontrol update JobId="$job_id" Requeue=0
effective_job=$(scontrol show job -o "$job_id")
grep -q 'Requeue=0' <<<"$effective_job" || { scancel "$job_id"; exit 1; }
echo "[test] job=$job_id requeue=0"
wait "$srun_pid"
REMOTE

grep -Eq '[0-9]+ passed' "$LOCAL_LOG_DIR/pytest.log" || {
  echo "[error] focused test log contains no pass summary" >&2
  exit 1
}
echo "[test] PASS output=$LOCAL_LOG_DIR"
