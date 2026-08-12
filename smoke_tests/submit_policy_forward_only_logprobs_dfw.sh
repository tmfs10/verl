#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="${SCRIPT_ROOT:-/home/siddjain/workspace/scripts/src}"
VERL_LOCAL_ROOT="${VERL_LOCAL_ROOT:-/home/siddjain/workspace/verl/verl_main}"
SKILLS_DIR="${SKILLS_DIR:-/home/siddjain/workspace/skills_latest}"
REMOTE_HOST="${REMOTE_HOST:-dfw}"
CLUSTER="${CLUSTER:-cw-dfw}"
CONFIG_DIR="${CONFIG_DIR:-/home/siddjain/workspace/scripts/nemo_configs/cluster/codegen}"
PARTITION="${PARTITION:-interactive}"
ACCOUNT="${ACCOUNT:-config}"
NODES="${NODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
EXP_NAME="${EXP_NAME:-policy-forward-only-deepseek-sglang-dfw-smoke-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/output/smoke_tests/policy_forward_only_logprobs/$EXP_NAME}"
RUN_ROOT="${RUN_ROOT:-/data/smoke_tests/policy_forward_only_logprobs/$EXP_NAME}"
REMOTE_VERL_ROOT="${REMOTE_VERL_ROOT:-/lustre/fsw/portfolios/llmservice/users/siddjain/workspace/verl/verl_main}"
NEMO_RL_ROOT="${NEMO_RL_ROOT:-/workspace/container_nemo_rl_v2}"
REMOTE_OUTPUT_ROOT="${REMOTE_OUTPUT_ROOT:-/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/lustre/fsw/portfolios/llmservice/users/siddjain/llm/data}"
LOCAL_SUBMIT_LOG_ROOT="${LOCAL_SUBMIT_LOG_ROOT:-$HOME/data/smoke_tests/policy_forward_only_logprobs/$EXP_NAME}"
CONTAINER_NAME="${CONTAINER_NAME:-/lustre/fsw/portfolios/llmservice/users/igitman/images/nemo-skills-megatron-latest.sqsh}"
MODEL="${MODEL:-/hf_models/DeepSeek-R1-0528-tp16-sglang}"
TOKENIZER="${TOKENIZER:-$MODEL}"
BACKEND="${BACKEND:-megatron}"
MAX_LENGTH="${MAX_LENGTH:-256}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PROCESS_ON_NODES="${PROCESS_ON_NODES:-8,8}"
TP="${TP:-2}"
PP="${PP:-1}"
EP="${EP:-8}"
ETP="${ETP:-1}"
CP="${CP:-1}"
VANILLA_MBRIDGE="${VANILLA_MBRIDGE:-0}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-1}"
USE_FUSED_KERNELS="${USE_FUSED_KERNELS:-0}"
PARAM_OFFLOAD="${PARAM_OFFLOAD:-0}"
INFER_MAX_TOKEN_LEN_PER_GPU="${INFER_MAX_TOKEN_LEN_PER_GPU:-1024}"
OVERRIDE_TRANSFORMER_CONFIG_JSON="${OVERRIDE_TRANSFORMER_CONFIG_JSON:-}"
OVERRIDE_MCORE_MODEL_CONFIG_JSON="${OVERRIDE_MCORE_MODEL_CONFIG_JSON:-}"
ENSURE_LIGHT_DEPS="${ENSURE_LIGHT_DEPS:-1}"
ENSURE_MODELOPT="${ENSURE_MODELOPT:-1}"
POLL_SECONDS="${POLL_SECONDS:-60}"
WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-1}"

if [[ "$NODES" -le 0 ]]; then
  echo "[error] NODES must be positive, got $NODES" >&2
  exit 1
fi
if [[ "$GPUS_PER_NODE" -ne 8 ]]; then
  echo "[error] DFW GPU jobs should request 8 GPUs per node; got GPUS_PER_NODE=$GPUS_PER_NODE" >&2
  exit 1
fi
if [[ "$PARTITION" != "interactive" ]]; then
  echo "[warn] smoke tests normally use interactive; got PARTITION=$PARTITION" >&2
fi

echo "[sync] staging VERL checkout to $REMOTE_HOST:$REMOTE_VERL_ROOT"
ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_VERL_ROOT'"
rsync -az --delete \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '.mypy_cache/' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude 'wandb/' \
  "$VERL_LOCAL_ROOT/" "$REMOTE_HOST:$REMOTE_VERL_ROOT/"

mkdir -p "$LOCAL_SUBMIT_LOG_ROOT"
submit_log="$LOCAL_SUBMIT_LOG_ROOT/submit.log"

OVERRIDE_TRANSFORMER_CONFIG_JSON_B64=""
OVERRIDE_MCORE_MODEL_CONFIG_JSON_B64=""
if [[ -n "$OVERRIDE_TRANSFORMER_CONFIG_JSON" ]]; then
  OVERRIDE_TRANSFORMER_CONFIG_JSON_B64="$(printf '%s' "$OVERRIDE_TRANSFORMER_CONFIG_JSON" | base64 -w0)"
fi
if [[ -n "$OVERRIDE_MCORE_MODEL_CONFIG_JSON" ]]; then
  OVERRIDE_MCORE_MODEL_CONFIG_JSON_B64="$(printf '%s' "$OVERRIDE_MCORE_MODEL_CONFIG_JSON" | base64 -w0)"
fi

printf -v COMMAND \
  "EXP_NAME=%q RUN_ROOT=%q VERL_ROOT=%q NEMO_RL_ROOT=%q MODEL=%q TOKENIZER=%q BACKEND=%q MAX_LENGTH=%q BATCH_SIZE=%q GPUS_PER_NODE=%q PROCESS_ON_NODES=%q TP=%q PP=%q EP=%q ETP=%q CP=%q VANILLA_MBRIDGE=%q USE_REMOVE_PADDING=%q USE_FUSED_KERNELS=%q PARAM_OFFLOAD=%q INFER_MAX_TOKEN_LEN_PER_GPU=%q OVERRIDE_TRANSFORMER_CONFIG_JSON_B64=%q OVERRIDE_MCORE_MODEL_CONFIG_JSON_B64=%q ENSURE_LIGHT_DEPS=%q ENSURE_MODELOPT=%q bash /workspace/verl/verl_main/smoke_tests/policy_forward_only_logprobs_dfw_payload.sh" \
  "$EXP_NAME" \
  "$RUN_ROOT" \
  "/workspace/verl/verl_main" \
  "$NEMO_RL_ROOT" \
  "$MODEL" \
  "$TOKENIZER" \
  "$BACKEND" \
  "$MAX_LENGTH" \
  "$BATCH_SIZE" \
  "$GPUS_PER_NODE" \
  "$PROCESS_ON_NODES" \
  "$TP" \
  "$PP" \
  "$EP" \
  "$ETP" \
  "$CP" \
  "$VANILLA_MBRIDGE" \
  "$USE_REMOVE_PADDING" \
  "$USE_FUSED_KERNELS" \
  "$PARAM_OFFLOAD" \
  "$INFER_MAX_TOKEN_LEN_PER_GPU" \
  "$OVERRIDE_TRANSFORMER_CONFIG_JSON_B64" \
  "$OVERRIDE_MCORE_MODEL_CONFIG_JSON_B64" \
  "$ENSURE_LIGHT_DEPS" \
  "$ENSURE_MODELOPT"

echo "[submit] exp=$EXP_NAME output_dir=$OUTPUT_DIR container=$CONTAINER_NAME nodes=$NODES"
cd "$SKILLS_DIR"
export PYTHONPATH="$SKILLS_DIR:${PYTHONPATH:-}"
set +e
python3 "$SCRIPT_ROOT/run_cmd_wrapper.py" \
  --cluster "$CLUSTER" \
  --config_dir "$CONFIG_DIR" \
  --expname "$EXP_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --container "$CONTAINER_NAME" \
  --partition "$PARTITION" \
  --account "$ACCOUNT" \
  --exclusive \
  --nodes "$NODES" \
  --client_gpus "$GPUS_PER_NODE" \
  --commands "$COMMAND" 2>&1 | tee "$submit_log"
rc=${PIPESTATUS[0]}
set -e
if [[ "$rc" -ne 0 ]]; then
  echo "[error] submit failed; see $submit_log" >&2
  exit "$rc"
fi

job_id="$(grep -Eo 'slurm_tunnel://nemo_run/[0-9]+' "$submit_log" | tail -1 | grep -Eo '[0-9]+' || true)"
if [[ -z "$job_id" ]]; then
  echo "[error] could not parse SLURM job id from $submit_log" >&2
  exit 1
fi

remote_output="$REMOTE_OUTPUT_ROOT/${OUTPUT_DIR#/output/}"
if [[ "$RUN_ROOT" == /data/* ]]; then
  remote_run_root="$REMOTE_DATA_ROOT/${RUN_ROOT#/data/}"
else
  remote_run_root="$RUN_ROOT"
fi
remote_output_jsonl="$remote_run_root/output.jsonl"
done_file="$remote_output/rank_0/_RUN_CMD_DONE"
log_dir="$remote_output/rank_0/logs"

if [[ "$WAIT_FOR_COMPLETION" == "0" ]]; then
  echo "[submit-only] job_id=$job_id"
  echo "[submit-only] remote_output=$remote_output"
  echo "[submit-only] log_dir=$log_dir"
  exit 0
fi

echo "[wait] job=$job_id done_file=$done_file"
while true; do
  live_state="$(ssh "$REMOTE_HOST" "squeue -h -j '$job_id' -o '%T' | head -1" || true)"
  if [[ -n "$live_state" ]]; then
    echo "[wait] job=$job_id state=$live_state"
  fi
  if [[ -z "$live_state" ]]; then
    state="$(ssh "$REMOTE_HOST" "sacct -j '$job_id' --format=State -n -P | head -1 | tr -d ' '" || true)"
    if [[ "$state" == COMPLETED* ]] && ssh "$REMOTE_HOST" "test -f '$done_file'" >/dev/null 2>&1; then
      if ssh "$REMOTE_HOST" "python3 - '$remote_output_jsonl' <<'PY'
import json
import sys

path = sys.argv[1]
rows = []
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))
if len(rows) != 1:
    raise SystemExit(f'expected 1 output row, got {len(rows)}')
row = rows[0]
if not row.get('logprobs'):
    raise SystemExit('output row has no logprobs')
if len(row['logprobs']) != len(row['scored_token_ids']):
    raise SystemExit('logprobs length does not match scored_token_ids')
print(f'validated output rows=1 scored_tokens={len(row[\"scored_token_ids\"])} path={path}')
PY"; then
        echo "[result] job=$job_id state=$state"
        echo "[result] done_file=$done_file"
        echo "[result] remote_output=$remote_output"
        echo "[result] output_jsonl=$remote_output_jsonl"
        exit 0
      fi
    fi
    echo "[error] job left queue without validated output; state=${state:-unknown} log_dir=$log_dir output_jsonl=$remote_output_jsonl" >&2
    ssh "$REMOTE_HOST" "find '$log_dir' -maxdepth 1 -type f 2>/dev/null | sort | tail -20" || true
    exit 1
  fi
  sleep "$POLL_SECONDS"
done
