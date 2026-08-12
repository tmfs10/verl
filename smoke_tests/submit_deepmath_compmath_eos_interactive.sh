#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="${SCRIPT_ROOT:-/home/siddjain/workspace/scripts/src}"
SKILLS_DIR="${SKILLS_DIR:-/home/siddjain/workspace/skills_latest}"
VERL_LOCAL_ROOT="${VERL_LOCAL_ROOT:-/home/siddjain/workspace/verl/verl_svopsd}"
CONFIG_SRC="${CONFIG_SRC:-/home/siddjain/workspace/scripts/nemo_configs/cluster/codegen/eos.yaml}"
CONFIG_DIR="${CONFIG_DIR:-$HOME/data/verl_runs/deepmath_sdpo_svopsd_eos/config_onelogger}"
REMOTE_HOST="${REMOTE_HOST:-eos}"
CLUSTER="${CLUSTER:-eos}"
PARTITION="${PARTITION:-interactive}"
NODES="${NODES:-1}"
GPUS="${GPUS:-8}"
VARIANT="${VARIANT:-svsdpo_caa}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-deepmath_compmath_${VARIANT}_eos_interactive_${RUN_TAG}}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/output/smoke_tests/deepmath_compmath_eos}"
REMOTE_OUTPUT_ROOT="${REMOTE_OUTPUT_ROOT:-/lustre/fsw/llmservice_nemo_reasoning/siddjain/nemo-run/output}"
ACTOR_MODEL="${ACTOR_MODEL:-/hf_models/Qwen3-30B-A3B}"
TRAIN_DATA="${TRAIN_DATA:-/data/rl/mathgen/deepmath_verl.jsonl}"
VAL_DATA="${VAL_DATA:-/data/rl/mathgen/comp_math_verl.jsonl}"
REWARD_FILE="${REWARD_FILE:-$SCRIPT_ROOT/nemo_verl/reward/verl_code_reward.py}"
N_PROMPTS="${N_PROMPTS:-8}"
N_SAMPLES="${N_SAMPLES:-8}"
N_VAL_SAMPLES="${N_VAL_SAMPLES:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2k}"
MAX_LEN="${MAX_LEN:-10k}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-10240}"
ACTOR_LR="${ACTOR_LR:-5e-6}"
ROLLOUT_TP="${ROLLOUT_TP:-4}"
LAYER_FRACTIONS="${LAYER_FRACTIONS:-0.31-0.37}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-8}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-8}"
DRY_RUN="${DRY_RUN:-0}"
WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-0}"
POLL_SECONDS="${POLL_SECONDS:-60}"
LOCAL_SUBMIT_LOG_ROOT="${LOCAL_SUBMIT_LOG_ROOT:-$HOME/data/smoke_tests/deepmath_compmath_eos/$EXP_NAME}"
EOS_VERL_IMAGE="${EOS_VERL_IMAGE:-/lustre/fsw/llmservice_nemo_reasoning/siddjain/containers/verl_vllm012_flashattn_20260321.sqsh}"
EOS_VERL_ONELOGGER_IMAGE="${EOS_VERL_ONELOGGER_IMAGE:-/lustre/fsw/llmservice_nemo_reasoning/siddjain/containers/verl_vllm012_flashattn_20260321_onelogger.sqsh}"

if [[ "$PARTITION" != "interactive" ]]; then
  echo "[error] EOS smoke tests must run on interactive; got PARTITION=$PARTITION" >&2
  exit 1
fi
if (( NODES > 2 )); then
  echo "[error] interactive smoke tests must use at most 2 nodes; got NODES=$NODES" >&2
  exit 1
fi
if (( GPUS != 8 )); then
  echo "[error] GPU jobs must request 8 GPUs per node; got GPUS=$GPUS" >&2
  exit 1
fi
if (( N_PROMPTS % (NODES * GPUS) != 0 )); then
  echo "[error] N_PROMPTS must be divisible by NODES*GPUS; got $N_PROMPTS % $((NODES * GPUS))" >&2
  exit 1
fi

mkdir -p "$CONFIG_DIR" "$LOCAL_SUBMIT_LOG_ROOT"
cp "$CONFIG_SRC" "$CONFIG_DIR/eos.yaml"
python3 - "$CONFIG_DIR/eos.yaml" "$EOS_VERL_ONELOGGER_IMAGE" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
verl_image = sys.argv[2]
text = path.read_text()
lines = []
replaced_verl = False
for line in text.splitlines():
    stripped = line.strip()
    if stripped.startswith("verl:"):
        indent = line[: len(line) - len(line.lstrip())]
        lines.append(f"{indent}verl: {verl_image}")
        replaced_verl = True
        continue
    line = line.replace(
        "/lustre/fsw/portfolios/llmservice/users/siddjain:/home",
        "/lustre/fsw/llmservice_nemo_reasoning/siddjain:/home",
    )
    line = line.replace(
        "/lustre/fsw/portfolios/llmservice/users/siddjain/.netrc:/root/.netrc",
        "/lustre/fsw/llmservice_nemo_reasoning/siddjain/.netrc:/root/.netrc",
    )
    lines.append(line)
if not replaced_verl:
    raise SystemExit("did not find containers.verl in EOS config")
out = "\n".join(lines) + "\n"
if "\ntimeouts:\n" in out and "\n  interactive:" not in out:
    out = out.replace(
        "timeouts:\n  batch: 04:00:00\n",
        "timeouts:\n  batch: 04:00:00\n  interactive: 02:00:00\n",
        1,
    )
path.write_text(out)
PY

ssh "$REMOTE_HOST" "set -euo pipefail
if ! test -e '$EOS_VERL_IMAGE'; then
  echo '[error] EOS VERL image missing: $EOS_VERL_IMAGE' >&2
  exit 1
fi
ln -sfn '$EOS_VERL_IMAGE' '$EOS_VERL_ONELOGGER_IMAGE'
test -e '$EOS_VERL_ONELOGGER_IMAGE'
test -f /lustre/fsw/llmservice_nemo_reasoning/siddjain/llm/data/rl/mathgen/deepmath_verl.jsonl
test -f /lustre/fsw/llmservice_nemo_reasoning/siddjain/llm/data/rl/mathgen/comp_math_verl.jsonl
test -d /lustre/fsw/llmservice_nemo_reasoning/hf_models/Qwen3-30B-A3B
"

if ! grep -q 'onelogger' "$CONFIG_DIR/eos.yaml"; then
  echo "[error] generated EOS config does not contain onelogger container path" >&2
  exit 1
fi

common_extra_args=$(cat <<EOF
data.filter_overlong_prompts=False
++data.dynamic_masked_solution=False
++data.min_masked_fraction=null
++data.max_masked_fraction=null
++data.mask_seed=null
++actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP
data.train_max_samples=$TRAIN_MAX_SAMPLES
data.val_max_samples=$VAL_MAX_SAMPLES
trainer.logger=['console']
trainer.save_freq=1000
trainer.log_val_generations=0
EOF
)
common_extra_args="$(printf '%s' "$common_extra_args" | tr '\n' ' ')"

opsd_extra_args=$(cat <<EOF
++algorithm.opsd.mode=opsd
++algorithm.opsd.teacher_source=sdpo_success_rollout
++algorithm.opsd.teacher_model=ema
++algorithm.opsd.teacher_ema_rate=0.05
++algorithm.opsd.topk=100
++algorithm.opsd.distill_loss=topk_jsd
++algorithm.opsd.distill_beta=0.5
++algorithm.opsd.sdpo_distill_only_failed=True
++algorithm.opsd.sdpo_exclude_self_success=True
++algorithm.opsd.offpolicy_is_mode=token
++algorithm.opsd.offpolicy_is_clip=2.0
algorithm.norm_adv_by_std_in_grpo=False
actor_rollout_ref.actor.optim.lr_warmup_steps=10
actor_rollout_ref.actor.optim.weight_decay=0.01
actor_rollout_ref.actor.optim.clip_grad=1.0
actor_rollout_ref.actor.optim.lr_scheduler_type=constant
++actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
EOF
)
opsd_extra_args="$(printf '%s' "$opsd_extra_args" | tr '\n' ' ')"

case "$VARIANT" in
  grpo)
    script_module="verl.trainer.main_ppo"
    extra_args="$common_extra_args"
    ;;
  sdpo)
    script_module="recipe.opsd.main_opsd"
    extra_args="$common_extra_args $opsd_extra_args ++algorithm.opsd.sdpo_conditioning_mode=prompt_append"
    ;;
  svsdpo_caa)
    script_module="recipe.opsd.main_opsd"
    extra_args="$common_extra_args $opsd_extra_args ++algorithm.opsd.sdpo_conditioning_mode=steering ++algorithm.opsd.steering.layer_fractions=\\\"$LAYER_FRACTIONS\\\" ++algorithm.opsd.steering.scale=1.0 ++algorithm.opsd.steering.normalize=null ++algorithm.opsd.steering.detach_vectors=True ++algorithm.opsd.steering.apply_positions=all_nonpad"
    ;;
  *)
    echo "[error] unknown VARIANT=$VARIANT; expected grpo, sdpo, or svsdpo_caa" >&2
    exit 1
    ;;
esac

submit_log="$LOCAL_SUBMIT_LOG_ROOT/submit.log"
remote_output="$REMOTE_OUTPUT_ROOT/${OUTPUT_BASE_DIR#/output}/$EXP_NAME"
log_dir="$remote_output/training-logs"
dry_arg=()
if [[ "$DRY_RUN" == "1" ]]; then
  dry_arg=(--dry_run)
fi

echo "[submit] exp=$EXP_NAME variant=$VARIANT cluster=$CLUSTER partition=$PARTITION nodes=$NODES gpus=$GPUS rollout_tp=$ROLLOUT_TP dry_run=$DRY_RUN"
echo "[submit] config=$CONFIG_DIR/eos.yaml"
echo "[submit] output=$remote_output"
echo "[submit] train=$TRAIN_DATA val=$VAL_DATA model=$ACTOR_MODEL"
echo "[submit] verl_image=$EOS_VERL_ONELOGGER_IMAGE"
echo "[submit] wandb=disabled"

cd "$SKILLS_DIR"
export PYTHONPATH="$SKILLS_DIR:${PYTHONPATH:-}"
set +e
conda run -n skills_latest python "$SCRIPT_ROOT/nemo_verl/skills_verl_submit.py" \
  --cluster "$CLUSTER" \
  --config_dir "$CONFIG_DIR" \
  --local_verl_folder "$VERL_LOCAL_ROOT" \
  --script_module "$script_module" \
  --reward_file "$REWARD_FILE" \
  --ground_truth_solution_key ground_truth_answer \
  --output_base_dir "$OUTPUT_BASE_DIR" \
  --expname "$EXP_NAME" \
  --partition "$PARTITION" \
  --nodes "$NODES" \
  --gpus "$GPUS" \
  --actor_model "$ACTOR_MODEL" \
  --prompt_data "$TRAIN_DATA" \
  --eval_data "$VAL_DATA" \
  --n_prompts "$N_PROMPTS" \
  --n_samples "$N_SAMPLES" \
  --n_val_samples "$N_VAL_SAMPLES" \
  --val_batch_size "$VAL_BATCH_SIZE" \
  --max_prompt_len "$MAX_PROMPT_LEN" \
  --max_len "$MAX_LEN" \
  --max_tokens_per_gpu "$MAX_TOKENS_PER_GPU" \
  --num_epochs 1 \
  --num_training_jobs 1 \
  --num_ppo_iter 1 \
  --actor_lr "$ACTOR_LR" \
  --T 0.85 \
  --val_T 0.6 \
  --val_top_p 1.0 \
  --save_freq 1000 \
  --test_freq 1 \
  --ae grpo \
  --kl_coef 0.0 \
  --clip_ae 0.2,0.28 \
  --reward_manager batch \
  --no_sandbox \
  --extra_args "$extra_args" \
  "${dry_arg[@]}" 2>&1 | tee "$submit_log"
rc=${PIPESTATUS[0]}
set -e
if (( rc != 0 )); then
  echo "[error] submit failed; see $submit_log" >&2
  exit "$rc"
fi
if grep -q 'one-logger-utils' "$submit_log"; then
  echo "[error] submit log contains one-logger-utils install path; refusing to continue" >&2
  exit 1
fi
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] completed without one-logger install injection"
  exit 0
fi

job_id="$(grep -Eo 'slurm_tunnel://nemo_run/[0-9]+' "$submit_log" | tail -1 | grep -Eo '[0-9]+' || true)"
if [[ -z "$job_id" ]]; then
  job_id="$(grep -Eo 'Submitted batch job [0-9]+' "$submit_log" | tail -1 | awk '{print $4}' || true)"
fi
if [[ -z "$job_id" ]]; then
  echo "[error] could not parse SLURM job id from $submit_log" >&2
  exit 1
fi
printf '%s\t%s\t%s\t%s\n' "$VARIANT" "$EXP_NAME" "$job_id" "$remote_output" > "$LOCAL_SUBMIT_LOG_ROOT/submitted_jobs.tsv"

echo "[submitted] job_id=$job_id"
echo "[submitted] remote_output=$remote_output"
echo "[submitted] log_dir=$log_dir"

if [[ "$WAIT_FOR_COMPLETION" == "0" ]]; then
  exit 0
fi

while true; do
  live_state="$(ssh "$REMOTE_HOST" "squeue -h -j '$job_id' -o '%T' | head -1" || true)"
  if [[ -n "$live_state" ]]; then
    echo "[wait] job=$job_id state=$live_state"
    sleep "$POLL_SECONDS"
    continue
  fi

  state="$(ssh "$REMOTE_HOST" "sacct -j '$job_id' --format=State -n -P | head -1 | tr -d ' '" || true)"
  echo "[wait] job=$job_id final_state=${state:-unknown}"
  if [[ "$state" != COMPLETED* ]]; then
    echo "[error] job did not complete; log_dir=$log_dir" >&2
    ssh "$REMOTE_HOST" "find '$log_dir' -maxdepth 1 -type f 2>/dev/null | sort | tail -20" || true
    exit 1
  fi
  echo "[result] PASS job=$job_id state=$state remote_output=$remote_output"
  exit 0
done
