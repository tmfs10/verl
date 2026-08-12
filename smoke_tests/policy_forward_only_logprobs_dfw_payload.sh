#!/usr/bin/env bash
set -euo pipefail

export WANDB_DISABLED=true
export HF_HOME="${HF_HOME:-/my_models/hf-cache}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$RUN_ROOT/hf_modules}"
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

EXP_NAME="${EXP_NAME:-manual-policy-forward-only-logprobs}"
RUN_ROOT="${RUN_ROOT:-/data/smoke_tests/policy_forward_only_logprobs/$EXP_NAME}"
VERL_ROOT="${VERL_ROOT:-/workspace/verl/verl_main}"
NEMO_RL_ROOT="${NEMO_RL_ROOT:-/workspace/container_nemo_rl_v2}"
MODEL="${MODEL:-/hf_models/DeepSeek-R1-0528-tp16-sglang}"
TOKENIZER="${TOKENIZER:-$MODEL}"
BACKEND="${BACKEND:-megatron}"
MAX_LENGTH="${MAX_LENGTH:-256}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
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
OVERRIDE_TRANSFORMER_CONFIG_JSON_B64="${OVERRIDE_TRANSFORMER_CONFIG_JSON_B64:-}"
OVERRIDE_MCORE_MODEL_CONFIG_JSON_B64="${OVERRIDE_MCORE_MODEL_CONFIG_JSON_B64:-}"
RAY_PORT="${RAY_PORT:-6379}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ENSURE_LIGHT_DEPS="${ENSURE_LIGHT_DEPS:-1}"
ENSURE_MODELOPT="${ENSURE_MODELOPT:-1}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1}"

mkdir -p "$RUN_ROOT/tmp" "$HF_MODULES_CACHE"
export TMPDIR="$RUN_ROOT/tmp"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${SLURM_JOB_ID:-manual}_${SLURM_NODEID:-0}}"
mkdir -p "$RAY_TMPDIR"

INPUT_FILE="$RUN_ROOT/input.jsonl"
OUTPUT_FILE="$RUN_ROOT/output.jsonl"
STOP_FILE="$RUN_ROOT/ray_stop"

echo "[env] hostname=$(hostname)"
echo "[env] EXP_NAME=$EXP_NAME"
echo "[env] RUN_ROOT=$RUN_ROOT"
echo "[env] VERL_ROOT=$VERL_ROOT"
echo "[env] NEMO_RL_ROOT=$NEMO_RL_ROOT"
echo "[env] MODEL=$MODEL TOKENIZER=$TOKENIZER BACKEND=$BACKEND"
echo "[env] MAX_LENGTH=$MAX_LENGTH BATCH_SIZE=$BATCH_SIZE PROCESS_ON_NODES=$PROCESS_ON_NODES"
echo "[env] TP=$TP PP=$PP EP=$EP ETP=$ETP CP=$CP VANILLA_MBRIDGE=$VANILLA_MBRIDGE"
echo "[env] SLURM_JOB_ID=${SLURM_JOB_ID:-unset} SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-unset} SLURM_PROCID=${SLURM_PROCID:-unset} SLURM_NODEID=${SLURM_NODEID:-unset} SLURM_LOCALID=${SLURM_LOCALID:-unset}"
nvidia-smi || true

if [[ -n "$OVERRIDE_TRANSFORMER_CONFIG_JSON_B64" ]]; then
  OVERRIDE_TRANSFORMER_CONFIG_JSON="$(printf '%s' "$OVERRIDE_TRANSFORMER_CONFIG_JSON_B64" | base64 -d)"
fi
if [[ -n "$OVERRIDE_MCORE_MODEL_CONFIG_JSON_B64" ]]; then
  OVERRIDE_MCORE_MODEL_CONFIG_JSON="$(printf '%s' "$OVERRIDE_MCORE_MODEL_CONFIG_JSON_B64" | base64 -d)"
fi

if [[ ! -d "$VERL_ROOT" ]]; then
  echo "[error] VERL_ROOT does not exist: $VERL_ROOT" >&2
  exit 1
fi

if [[ "${SLURM_LOCALID:-0}" != "0" ]]; then
  echo "[skip] SLURM_LOCALID=${SLURM_LOCALID:-unset}; Ray node process is launched only from local rank 0"
  exit 0
fi

mapfile -t EXPANDED_NODES < <(python3 - "${SLURM_JOB_NODELIST:-$(hostname -s)}" <<'PY'
import sys

raw = sys.argv[1]

def expand_one(item):
    if "[" not in item:
        return [item]
    prefix, rest = item.split("[", 1)
    inner, suffix = rest.split("]", 1)
    out = []
    for part in inner.split(","):
        if "-" in part:
            start, end = part.split("-", 1)
            width = max(len(start), len(end))
            for value in range(int(start), int(end) + 1):
                out.append(f"{prefix}{value:0{width}d}{suffix}")
        else:
            out.append(f"{prefix}{part}{suffix}")
    return out

items = []
depth = 0
start = 0
for idx, ch in enumerate(raw):
    if ch == "[":
        depth += 1
    elif ch == "]":
        depth -= 1
    elif ch == "," and depth == 0:
        items.append(raw[start:idx])
        start = idx + 1
items.append(raw[start:])
for item in items:
    for node in expand_one(item):
        print(node)
PY
)

NNODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-${#EXPANDED_NODES[@]}}}"
if [[ "$NNODES" -le 0 ]]; then
  NNODES=1
fi
HEAD_NODE="${EXPANDED_NODES[0]:-$(hostname -s)}"
HEAD_ADDR="${HEAD_ADDR:-$HEAD_NODE}"
NODE_RANK="${SLURM_NODEID:-0}"
echo "[dist] node_rank=$NODE_RANK nnodes=$NNODES head=$HEAD_ADDR:$RAY_PORT nodes=${EXPANDED_NODES[*]:-unknown}"

EXTRA_BRIDGE_PYTHONPATH=""
if [[ -d "$NEMO_RL_ROOT/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src" ]]; then
  EXTRA_BRIDGE_PYTHONPATH="$NEMO_RL_ROOT/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src"
fi
if [[ -d "$NEMO_RL_ROOT/3rdparty/Megatron-LM-workspace/Megatron-LM" ]]; then
  EXTRA_BRIDGE_PYTHONPATH="${EXTRA_BRIDGE_PYTHONPATH:+$EXTRA_BRIDGE_PYTHONPATH:}$NEMO_RL_ROOT/3rdparty/Megatron-LM-workspace/Megatron-LM"
fi
export PYTHONPATH="$HF_MODULES_CACHE:$VERL_ROOT${EXTRA_BRIDGE_PYTHONPATH:+:$EXTRA_BRIDGE_PYTHONPATH}:${PYTHONPATH:-}"
echo "[env] PYTHONPATH=$PYTHONPATH"
cd "$VERL_ROOT"

if [[ "$ENSURE_LIGHT_DEPS" == "1" ]]; then
  missing_packages="$("$PYTHON_BIN" - <<'PY'
import importlib

required = {
    "codetiming": "codetiming",
    "omegaconf": "omegaconf>=2.3.0",
    "hydra": "hydra-core>1.3,<=1.3.2",
    "ray": "ray[default]",
    "tensordict": "tensordict",
}
missing = []
for module_name, package_spec in required.items():
    try:
        importlib.import_module(module_name)
    except Exception:
        missing.append(package_spec)
print(" ".join(missing))
PY
)"
  if [[ -n "$missing_packages" ]]; then
    echo "[deps] installing missing lightweight packages: $missing_packages"
    "$PYTHON_BIN" -m pip install --no-cache-dir $missing_packages
  fi
fi

if [[ "$ENSURE_MODELOPT" == "1" ]]; then
  if ! "$PYTHON_BIN" -c 'import modelopt' >/dev/null 2>&1; then
    echo "[deps] installing nvidia-modelopt"
    "$PYTHON_BIN" -m pip install --no-cache-dir nvidia-modelopt
  fi
fi

"$PYTHON_BIN" - <<'PY'
import importlib
import sys

checks = [
    "torch",
    "ray",
    "transformers",
    "verl",
    "megatron.core",
    "megatron.bridge",
    "transformer_engine",
]
for name in checks:
    importlib.import_module(name)
print(f"[python] executable={sys.executable} version={sys.version.split()[0]}")
print("[python] import checks passed:", ", ".join(checks), flush=True)
PY

finish() {
  status=$?
  if [[ "$NODE_RANK" == "0" ]]; then
    touch "$STOP_FILE" || true
  fi
  ray stop --force >/dev/null 2>&1 || true
  exit "$status"
}
trap finish EXIT

ray stop --force >/dev/null 2>&1 || true

if [[ "$NODE_RANK" == "0" ]]; then
  rm -f "$STOP_FILE"
  cat > "$INPUT_FILE" <<'JSONL'
{"text": "DeepSeek forward-only smoke test. The policy model should score these tokens."}
JSONL
  echo "[ray] starting head on $HEAD_ADDR:$RAY_PORT"
  ray start --head --node-ip-address="$HEAD_ADDR" --port="$RAY_PORT" --num-gpus="$GPUS_PER_NODE" --include-dashboard=false --temp-dir="$RAY_TMPDIR" --block &
else
  echo "[ray] waiting for head $HEAD_ADDR:$RAY_PORT"
  for _ in $(seq 1 120); do
    if "$PYTHON_BIN" - "$HEAD_ADDR" "$RAY_PORT" >/dev/null 2>&1 <<'PY'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
with socket.create_connection((host, port), timeout=2):
    pass
PY
    then
      break
    fi
    sleep 2
  done
  echo "[ray] starting worker for $HEAD_ADDR:$RAY_PORT"
  ray start --address "$HEAD_ADDR:$RAY_PORT" --num-gpus="$GPUS_PER_NODE" --temp-dir="$RAY_TMPDIR" --block &
fi

sleep 15

if [[ "$NODE_RANK" != "0" ]]; then
  echo "[ray] worker waiting for stop marker $STOP_FILE"
  while [[ ! -f "$STOP_FILE" ]]; do
    sleep 5
  done
  echo "[ray] worker stop marker observed"
  exit 0
fi

expected_gpus=$((NNODES * GPUS_PER_NODE))
"$PYTHON_BIN" - "$expected_gpus" <<'PY'
import sys
import time

import ray

expected = int(sys.argv[1])
deadline = time.time() + 600
last = None
while time.time() < deadline:
    try:
        ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
        resources = ray.cluster_resources()
        total = int(resources.get("GPU", 0))
        last = resources
        print(f"[ray] resources={resources}", flush=True)
        if total >= expected:
            sys.exit(0)
    except Exception as exc:
        last = repr(exc)
        print(f"[ray] resource check failed: {exc!r}", flush=True)
    time.sleep(5)
raise SystemExit(f"Timed out waiting for {expected} Ray GPUs. Last={last}")
PY

args=(
  "$VERL_ROOT/scripts/policy_forward_only_logprobs.py"
  --model "$MODEL"
  --tokenizer "$TOKENIZER"
  --backend "$BACKEND"
  --trust-remote-code
  --ray-address auto
  --process-on-nodes "$PROCESS_ON_NODES"
  --input-jsonl "$INPUT_FILE"
  --text-field text
  --output-jsonl "$OUTPUT_FILE"
  --batch-size "$BATCH_SIZE"
  --max-length "$MAX_LENGTH"
  --progress-every "$PROGRESS_EVERY"
  --tensor-model-parallel-size "$TP"
  --pipeline-model-parallel-size "$PP"
  --expert-model-parallel-size "$EP"
  --context-parallel-size "$CP"
  --infer-max-token-len-per-gpu "$INFER_MAX_TOKEN_LEN_PER_GPU"
)

if [[ "$ETP" != "" && "$ETP" != "none" ]]; then
  args+=(--expert-tensor-parallel-size "$ETP")
fi
if [[ -n "$OVERRIDE_TRANSFORMER_CONFIG_JSON" ]]; then
  args+=(--override-transformer-config-json "$OVERRIDE_TRANSFORMER_CONFIG_JSON")
fi
if [[ -n "$OVERRIDE_MCORE_MODEL_CONFIG_JSON" ]]; then
  args+=(--override-mcore-model-config-json "$OVERRIDE_MCORE_MODEL_CONFIG_JSON")
fi
if [[ "$VANILLA_MBRIDGE" == "1" ]]; then
  args+=(--vanilla-mbridge)
else
  args+=(--no-vanilla-mbridge)
fi
if [[ "$USE_REMOVE_PADDING" == "1" ]]; then
  args+=(--use-remove-padding)
else
  args+=(--no-use-remove-padding)
fi
if [[ "$USE_FUSED_KERNELS" == "1" ]]; then
  args+=(--use-fused-kernels)
fi
if [[ "$PARAM_OFFLOAD" == "1" ]]; then
  args+=(--param-offload)
fi

echo "[run] $PYTHON_BIN ${args[*]}"
"$PYTHON_BIN" "${args[@]}"

"$PYTHON_BIN" - "$OUTPUT_FILE" <<'PY'
import json
import sys

path = sys.argv[1]
rows = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))
if len(rows) != 1:
    raise SystemExit(f"expected 1 output row, got {len(rows)}")
row = rows[0]
if not row.get("logprobs"):
    raise SystemExit("output row has no logprobs")
if len(row["logprobs"]) != len(row["scored_token_ids"]):
    raise SystemExit("logprobs length does not match scored_token_ids")
print(f"[validate] rows=1 scored_tokens={len(row['scored_token_ids'])} output={path}", flush=True)
PY

echo "[done] forward-only policy smoke completed"
