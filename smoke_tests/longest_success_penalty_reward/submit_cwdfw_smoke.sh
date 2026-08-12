#!/usr/bin/env bash
set -euo pipefail

# One-update, dry-run-first CW-DFW smoke. Validation is included because this
# change modifies validation reward isolation, and it runs in one full batch.

ACTION="${1:-dry-run}"
SCRIPT_ROOT="${SCRIPT_ROOT:-/home/siddjain/workspace/scripts/src}"
VERL_ROOT="${VERL_ROOT:-/home/siddjain/workspace/verl/verl_main}"
PYTHON="${PYTHON:-/home/siddjain/anaconda3/envs/skills_latest/bin/python}"
LAUNCHER="${LAUNCHER:-$SCRIPT_ROOT/nemo_verl/skills_verl_submit.py}"
REWARD_FILE="${REWARD_FILE:-$SCRIPT_ROOT/nemo_verl/reward/verl_code_reward.py}"
VERIFY_SCRIPT="${VERIFY_SCRIPT:-$VERL_ROOT/smoke_tests/longest_success_penalty_reward/verify_reward_audit.py}"
CONFIG_DIR="${CONFIG_DIR:-/home/siddjain/workspace/scripts/nemo_configs/cluster/codegen}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
MARGIN_PERCENT="${MARGIN_PERCENT:-50.0}"
ACTOR_MODEL="${ACTOR_MODEL:-/hf_models/Qwen3-1.7B}"
TRAIN_DATA="${TRAIN_DATA:-/data/rl/opsd_openthoughts_compmath/openthoughts_math_30k_opsd_full.jsonl}"
VAL_DATA="${VAL_DATA:-/data/rl/mathgen/comp_math_verl.jsonl}"
REMOTE_TRAIN_DATA="${REMOTE_TRAIN_DATA:-/lustre/fsw/portfolios/llmservice/users/siddjain/llm/data/rl/opsd_openthoughts_compmath/openthoughts_math_30k_opsd_full.jsonl}"
REMOTE_VAL_DATA="${REMOTE_VAL_DATA:-/lustre/fsw/portfolios/llmservice/users/siddjain/llm/data/rl/mathgen/comp_math_verl.jsonl}"
EXPECTED_ROWS="${EXPECTED_ROWS:-29427}"
EXPECTED_SHA256="${EXPECTED_SHA256:-f79a42fe155218db2f1927ee903afd101929724f2d0516352bdbb91cdb139178}"
EXPECTED_VAL_ROWS="${EXPECTED_VAL_ROWS:-256}"

N_PROMPTS=16
N_SAMPLES=8
N_VAL_SAMPLES=4
VAL_BATCH_SIZE="$EXPECTED_VAL_ROWS"
EXPECTED_VALIDATION_GENERATIONS=$((EXPECTED_VAL_ROWS * N_VAL_SAMPLES))
MAX_PROMPT_LEN=4096
MAX_RESPONSE_LEN=8192
MODEL_MAX_LEN=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))
EXP_NAME="longest_success_penalty_qwen3_1p7b_smoke_${RUN_TAG}"
OUTPUT_BASE="${OUTPUT_BASE:-/output/smoke_tests/longest_success_penalty_reward_${RUN_TAG}}"
REMOTE_OUTPUT_BASE="${REMOTE_OUTPUT_BASE:-/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/${OUTPUT_BASE#/output/}}"
REMOTE_EXP_DIR="$REMOTE_OUTPUT_BASE/$EXP_NAME"
LOCAL_RUN_DIR="${LOCAL_RUN_DIR:-/home/siddjain/data/smoke_tests/longest_success_penalty_reward/$RUN_TAG}"

die() { echo "[error] $*" >&2; exit 2; }

preflight() {
  [[ "$ACTOR_MODEL" == /hf_models/Qwen3-1.7B ]] || \
    die "audited smoke requires /hf_models/Qwen3-1.7B, got $ACTOR_MODEL"
  [[ -f "$LAUNCHER" && -f "$REWARD_FILE" && -f "$VERIFY_SCRIPT" ]] || \
    die "required launcher, reward function, or verifier is missing"
  "$PYTHON" - <<PY
import math
margin = float("$MARGIN_PERCENT")
if not math.isfinite(margin) or margin < 0:
    raise SystemExit("MARGIN_PERCENT must be finite and non-negative")
PY
  local rows sha val_rows
  rows="$(ssh dfw "wc -l < '$REMOTE_TRAIN_DATA'")"
  sha="$(ssh dfw "sha256sum '$REMOTE_TRAIN_DATA'" | awk '{print $1}')"
  val_rows="$(ssh dfw "wc -l < '$REMOTE_VAL_DATA'")"
  [[ "$rows" == "$EXPECTED_ROWS" ]] || die "training rows $rows != $EXPECTED_ROWS"
  [[ "$sha" == "$EXPECTED_SHA256" ]] || die "training SHA-256 mismatch: $sha"
  [[ "$val_rows" == "$EXPECTED_VAL_ROWS" ]] || die "validation rows $val_rows != $EXPECTED_VAL_ROWS"
  if [[ "$ACTION" == dry-run || "$ACTION" == submit ]] && \
    ssh dfw "test -e '$REMOTE_EXP_DIR/checkpoints' -o -e '$REMOTE_EXP_DIR/generations'"; then
    die "remote experiment directory already contains run outputs: $REMOTE_EXP_DIR"
  fi
  mkdir -p "$LOCAL_RUN_DIR"
  echo "[preflight] cluster=cw-dfw partition=interactive nodes=1 gpus=8 temperature=1.0 prompts=$N_PROMPTS rollouts=$N_SAMPLES margin_percent=$MARGIN_PERCENT"
  echo "[preflight] train_rows=$rows train_sha256=$sha val_rows=$val_rows val_batch_size=$VAL_BATCH_SIZE val_batches=1 output=$REMOTE_EXP_DIR"
}

extra_args() {
  cat <<EOF
data.filter_overlong_prompts=True
data.filter_overlong_prompts_workers=16
data.dataloader_num_workers=0
++data.dynamic_masked_solution=False
++data.min_masked_fraction=null
++data.max_masked_fraction=null
++data.mask_seed=null
actor_rollout_ref.rollout.gpu_memory_utilization=0.4
actor_rollout_ref.rollout.max_model_len=$MODEL_MAX_LEN
actor_rollout_ref.rollout.max_num_batched_tokens=$MODEL_MAX_LEN
actor_rollout_ref.rollout.top_p=1.0
actor_rollout_ref.rollout.top_k=-1
actor_rollout_ref.actor.loss_agg_mode=token-mean
trainer.total_training_steps=1
trainer.total_epochs=1
trainer.logger=['console']
trainer.val_before_train=true
trainer.save_freq=-1
trainer.test_freq=-1
trainer.log_val_generations=0
algorithm.use_kl_in_reward=False
++reward.reward_kwargs.use_longest_success_penalty_reward=true
++reward.reward_kwargs.longest_success_no_penalty_margin_percent=$MARGIN_PERCENT
++reward.reward_kwargs.longest_success_threshold=0.5
EOF
}

submit() {
  local dry_run="$1"
  local flattened action_log
  flattened="$(extra_args | tr '\n' ' ')"
  action_log="$LOCAL_RUN_DIR/submit.log"
  [[ "$dry_run" == 1 ]] && action_log="$LOCAL_RUN_DIR/dry_run.log"
  local -a cmd=(
    "$PYTHON" "$LAUNCHER"
    --cluster cw-dfw
    --config_dir "$CONFIG_DIR"
    --output_base_dir "$OUTPUT_BASE"
    --local_verl_folder "$VERL_ROOT"
    --script_module verl.trainer.main_ppo
    --reward_file "$REWARD_FILE"
    --ground_truth_solution_key solution
    --expname "$EXP_NAME"
    --partition interactive
    --time_limit 04:00:00
    --nodes 1
    --gpus 8
    --actor_model "$ACTOR_MODEL"
    --prompt_data "$TRAIN_DATA"
    --eval_data "$VAL_DATA"
    --n_prompts "$N_PROMPTS"
    --n_samples "$N_SAMPLES"
    --n_val_samples "$N_VAL_SAMPLES"
    --val_batch_size "$VAL_BATCH_SIZE"
    --max_prompt_len "$MAX_PROMPT_LEN"
    --max_len "$MODEL_MAX_LEN"
    --max_tokens_per_gpu "$MODEL_MAX_LEN"
    --num_epochs 1
    --num_training_jobs 1
    --num_ppo_iter 1
    --actor_lr 2e-6
    --clip_ae 0.2,0.28
    --infer_server vllm
    --sequence_parallel_size 1
    --T 1.0
    --val_T 1.0
    --val_top_p 1.0
    --save_freq -1
    --test_freq -1
    --ae grpo
    --seed 1234
    --no_sandbox
    --no_requeue
    --omit_noncore_algorithm_overrides
    --extra_args "$flattened"
  )
  [[ "$dry_run" == 1 ]] && cmd+=(--dry_run)
  {
    printf '#!/usr/bin/env bash\n'
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } > "$LOCAL_RUN_DIR/command.sh"
  chmod +x "$LOCAL_RUN_DIR/command.sh"
  "${cmd[@]}" 2>&1 | tee "$action_log"
  if [[ "$dry_run" == 0 ]]; then
    local job_id verified=0
    job_id="$(grep -Eo 'slurm_tunnel://nemo_run/[0-9]+' "$LOCAL_RUN_DIR/submit.log" | tail -1 | grep -Eo '[0-9]+' || true)"
    [[ -n "$job_id" ]] || job_id="$(grep -Eo 'Submitted batch job [0-9]+' "$LOCAL_RUN_DIR/submit.log" | tail -1 | awk '{print $4}' || true)"
    [[ -n "$job_id" ]] || die "could not parse submitted job ID"
    printf '%s\n' "$job_id" > "$LOCAL_RUN_DIR/job_id"
    for _ in 1 2 3 4 5; do
      if ssh dfw "scontrol update JobId='$job_id' Requeue=0; scontrol show job '$job_id' -o" \
        > "$LOCAL_RUN_DIR/scontrol.txt"; then
        verified=1
        break
      fi
      sleep 2
    done
    [[ "$verified" == 1 ]] || die "could not inspect submitted job $job_id"
    grep -q 'Requeue=0' "$LOCAL_RUN_DIR/scontrol.txt" || die "job is requeueable"
    grep -q 'Partition=interactive' "$LOCAL_RUN_DIR/scontrol.txt" || die "job is not interactive"
    grep -q 'NumNodes=1' "$LOCAL_RUN_DIR/scontrol.txt" || die "job does not request one node"
    grep -Eq 'TresPerNode=gres/gpu:8|Gres=gpu:8' "$LOCAL_RUN_DIR/scontrol.txt" || \
      die "job does not request eight GPUs"
    echo "[submitted] job=$job_id output=$REMOTE_EXP_DIR"
  fi
}

status() {
  local job_id
  job_id="$(cat "$LOCAL_RUN_DIR/job_id")"
  ssh dfw "squeue -j '$job_id' -o '%.18i %.12P %.24j %.10T %.10M %.10l %R'; sacct -X -j '$job_id' -o JobID,State,Elapsed,Start,End,ExitCode -P"
}

verify() {
  local generations_dir="$LOCAL_RUN_DIR/generations"
  local training_logs_dir="$LOCAL_RUN_DIR/training-logs"
  local report_dir="$LOCAL_RUN_DIR/manual_audit"
  mkdir -p "$generations_dir" "$training_logs_dir" "$report_dir"
  rsync -a "dfw:$REMOTE_EXP_DIR/generations/" "$generations_dir/"
  rsync -a "dfw:$REMOTE_EXP_DIR/training-logs/" "$training_logs_dir/"

  mapfile -t rollout_files < <(find "$generations_dir/rollout" -maxdepth 1 -type f -name '*.jsonl' | sort)
  mapfile -t validation_files < <(find "$generations_dir/validation" -maxdepth 1 -type f -name '*.jsonl' | sort)
  mapfile -t trainer_logs < <(find "$training_logs_dir" -maxdepth 1 -type f -name 'main_*_srun.log' | sort)
  [[ "${#rollout_files[@]}" == 1 ]] || die "expected one rollout JSONL, found ${#rollout_files[@]}"
  [[ "${#validation_files[@]}" == 1 ]] || die "expected one validation JSONL, found ${#validation_files[@]}"
  [[ "${#trainer_logs[@]}" == 1 ]] || die "expected one trainer main log, found ${#trainer_logs[@]}"

  "$PYTHON" "$VERIFY_SCRIPT" \
    --rollout-jsonl "${rollout_files[0]}" \
    --validation-jsonl "${validation_files[0]}" \
    --trainer-log "${trainer_logs[0]}" \
    --margin-percent "$MARGIN_PERCENT" \
    --expected-group-size "$N_SAMPLES" \
    --expected-validation-rows "$EXPECTED_VALIDATION_GENERATIONS" \
    --report-dir "$report_dir"
}

preflight
case "$ACTION" in
  dry-run) submit 1 ;;
  submit) submit 0 ;;
  status) status ;;
  verify) verify ;;
  *) die "ACTION must be dry-run, submit, status, or verify" ;;
esac
