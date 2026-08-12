#!/usr/bin/env bash
set -euo pipefail

# Run the focused SV-OPSD suite in the same container used by the live smoke.
# The host checkout is mounted directly, so this never copies or mutates code.

SSH_TARGET="${SSH_TARGET:-dfw}"
ACCOUNT="${ACCOUNT:-nemotron_reason_code}"
PARTITION="${PARTITION:-interactive}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
LOCAL_VERL_ROOT="${LOCAL_VERL_ROOT:-/home/siddjain/workspace/verl/verl_main}"
REMOTE_STAGE_BASE="${REMOTE_STAGE_BASE:-/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/setup/svopsd_openthoughts}"
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/siddjain/containers/verl_vllm012_flashattn_20260321.sqsh}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-/home/siddjain/data/smoke_tests/svopsd_openthoughts/container_tests_$RUN_TAG}"
REMOTE_VERL_ROOT="$REMOTE_STAGE_BASE/container_tests_$RUN_TAG/verl_main"

[[ "$PARTITION" == "interactive" ]] || {
  echo "[error] focused tests must use the interactive partition" >&2
  exit 2
}
[[ "$TIME_LIMIT" =~ ^00:([0-5][0-9]):00$ ]] || {
  echo "[error] focused-test time limit must remain below one hour" >&2
  exit 2
}
mkdir -p "$LOCAL_LOG_DIR"

# The workstation and cluster-login workspaces are not the same filesystem.
# Stage the exact live dirty tree to a new run-specific root; never mutate or
# rely on an older cluster-side checkout.
ssh "$SSH_TARGET" "mkdir -p '$REMOTE_VERL_ROOT'"
rsync -a \
  --exclude='.git/' \
  --exclude='.pytest_cache/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  "$LOCAL_VERL_ROOT/" "$SSH_TARGET:$REMOTE_VERL_ROOT/"
ssh "$SSH_TARGET" \
  "test -f '$REMOTE_VERL_ROOT/tests/recipe/opsd/test_steering.py' && test -f '$REMOTE_VERL_ROOT/recipe/opsd/steering.py'"

remote_command=$(cat <<EOF
set -euo pipefail
cd /workspace/verl
export PYTHONPATH=/workspace/verl:/workspace/verl/recipe
export WANDB_MODE=disabled
python -m pytest -q \
  tests/recipe/opsd/test_policy_gradient_steering.py \
  tests/recipe/opsd/test_steering.py \
  tests/recipe/opsd/test_teacher_utils.py \
  tests/recipe/opsd/test_dp_actor.py \
  tests/recipe/opsd/test_opsd_loss.py \
  tests/recipe/opsd/test_opsd_audit.py \
  tests/recipe/opsd/test_opsd_config.py \
  tests/recipe/opsd/test_opsd_trainer.py
EOF
)

printf '%s\n' "$remote_command" > "$LOCAL_LOG_DIR/container_command.sh"
echo "[test] output=$LOCAL_LOG_DIR"
remote_command_b64=$(printf '%s' "$remote_command" | base64 -w0)
ssh "$SSH_TARGET" bash -s -- \
  "$ACCOUNT" \
  "$PARTITION" \
  "$TIME_LIMIT" \
  "svopsd-tests-$RUN_TAG" \
  "$CONTAINER" \
  "$REMOTE_VERL_ROOT" \
  "$remote_command_b64" \
  2>&1 <<'REMOTE' | tee "$LOCAL_LOG_DIR/pytest.log"
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
  if [[ -n "$job_id" ]]; then
    break
  fi
  if ! kill -0 "$srun_pid" 2>/dev/null; then
    break
  fi
  sleep 0.25
done
if [[ -z "$job_id" ]]; then
  wait "$srun_pid" || true
  echo "[error] could not resolve Slurm job ID for $job_name" >&2
  exit 1
fi
scontrol update JobId="$job_id" Requeue=0
effective_job=$(scontrol show job -o "$job_id")
grep -q 'Requeue=0' <<<"$effective_job" || {
  scancel "$job_id"
  echo "[error] failed to enforce Requeue=0 for job $job_id" >&2
  exit 1
}
echo "[test] job=$job_id requeue=0"
wait "$srun_pid"
REMOTE

grep -Eq '[0-9]+ passed' "$LOCAL_LOG_DIR/pytest.log" || {
  echo "[error] focused test log contains no pass summary" >&2
  exit 1
}
echo "[test] PASS"
