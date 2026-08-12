#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="${SCRIPT_ROOT:-/home/siddjain/workspace/scripts/src}"
SKILLS_ENV_PYTHON="${SKILLS_ENV_PYTHON:-/home/siddjain/anaconda3/envs/skills_latest/bin/python}"
LAUNCHER="${LAUNCHER:-$SCRIPT_ROOT/nemo_verl/skills_verl_submit.py}"
VERL_LOCAL_ROOT="${VERL_LOCAL_ROOT:-/home/siddjain/workspace/verl/verl_main}"
CONFIG_DIR="${CONFIG_DIR:-/home/siddjain/workspace/scripts/nemo_configs/cluster/codegen}"
DATA_PREP_SCRIPT="${DATA_PREP_SCRIPT:-$SCRIPT_ROOT/nemo_verl/prepare_openthoughts_math_opsd.py}"

CLUSTER="${CLUSTER:-cw-dfw}"
REMOTE_HOST="${REMOTE_HOST:-dfw}"
ACCOUNT="${ACCOUNT:-nemotron_reason_code}"
PARTITION="${PARTITION:-interactive}"
NODES="${NODES:-1}"
GPUS="${GPUS:-8}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"

ACTOR_MODEL="${ACTOR_MODEL:-/hf_models/Qwen3-1.7B}"
TEACHER_MODEL="${TEACHER_MODEL:-/hf_models/Qwen3-1.7B}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-1536}"
MAX_LEN="$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-4096}"
TEMPERATURE="${TEMPERATURE:-1.0}"

N_PROMPTS="${N_PROMPTS:-8}"
N_SAMPLES="${N_SAMPLES:-2}"
# This VERL branch intentionally exports only best/majority/worst validation
# aggregates. With n=1 there is no aggregate and the trainer rejects an empty
# initial metric dict, so CompMath validation needs at least two samples.
N_VAL_SAMPLES="${N_VAL_SAMPLES:-2}"
TOTAL_STEPS="${TOTAL_STEPS:-3}"
RLVR_WARMUP_STEPS="${RLVR_WARMUP_STEPS:-1}"
SAVE_FREQ="${SAVE_FREQ:--1}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"
AUDIT_FAIL_FAST="${AUDIT_FAIL_FAST:-true}"
AUDIT_PROFILE="${AUDIT_PROFILE:-separate_sft_warmup}"
AUDIT_GLOBAL_STEPS="${AUDIT_GLOBAL_STEPS:-[1,2,3]}"
AUDIT_EXPECT_WARMUP_STEPS="${AUDIT_EXPECT_WARMUP_STEPS-1}"
AUDIT_EXPECT_JOINT_STEPS="${AUDIT_EXPECT_JOINT_STEPS-2 3}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
TEACHER_LR="${TEACHER_LR:-1e-6}"
TEACHER_MODEL_MODE="${TEACHER_MODEL_MODE:-separate}"
TEACHER_RLVR_BACKWARD_SCALE="${TEACHER_RLVR_BACKWARD_SCALE:-0.0}"
TEACHER_SFT_WEIGHT="${TEACHER_SFT_WEIGHT:-1.0}"
TEACHER_SFT_TARGET_SCOPE="${TEACHER_SFT_TARGET_SCOPE:-thinking_and_answer}"
OFFPOLICY_IS_MODE="${OFFPOLICY_IS_MODE:-sequence}"
ADVANTAGE_SHAPING="${ADVANTAGE_SHAPING:-0}"
ADVANTAGE_SHAPING_SCALE="${ADVANTAGE_SHAPING_SCALE:-1.0}"
ADVANTAGE_SHAPING_MAX_RESPONSE_TOKENS="${ADVANTAGE_SHAPING_MAX_RESPONSE_TOKENS:-null}"

TRAIN_DATASET="${TRAIN_DATASET:-siyanzhao/Openthoughts_math_30k_opsd}"
TRAIN_DATASET_REVISION="${TRAIN_DATASET_REVISION:-1f33e9dc2e8a1c639ca74f8024ad4a9f1f5eae62}"
TRAIN_DATASET_SCAN_LIMIT="${TRAIN_DATASET_SCAN_LIMIT:-1024}"
TRAIN_DATA_SELECTION="${TRAIN_DATA_SELECTION:-shortest}"
TRAIN_MAX_PROBLEM_CHARS="${TRAIN_MAX_PROBLEM_CHARS:-1200}"
TRAIN_MAX_SOLUTION_CHARS="${TRAIN_MAX_SOLUTION_CHARS:-8000}"
TRAIN_MAX_ANSWER_CHARS="${TRAIN_MAX_ANSWER_CHARS:-128}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-opsd_openthoughts_compmath_qwen3_1p7b_smoke_${RUN_TAG}}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/output/smoke_tests/opsd_openthoughts_compmath}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-$HOME/data/smoke_tests/opsd_openthoughts_compmath/data}"
LOCAL_FIXTURE="${LOCAL_FIXTURE:-$LOCAL_DATA_DIR/openthoughts_math_30k_opsd_smoke${N_PROMPTS}.jsonl}"
REMOTE_DATA_HOST_DIR="${REMOTE_DATA_HOST_DIR:-/lustre/fsw/portfolios/llmservice/users/siddjain/llm/data/rl/opsd_openthoughts_compmath}"
REMOTE_DATA_HOST_PATH="${REMOTE_DATA_HOST_PATH:-$REMOTE_DATA_HOST_DIR/openthoughts_math_30k_opsd_smoke${N_PROMPTS}.jsonl}"
TRAIN_DATA="${TRAIN_DATA:-/data/rl/opsd_openthoughts_compmath/openthoughts_math_30k_opsd_smoke${N_PROMPTS}.jsonl}"
VAL_DATA="${VAL_DATA:-/data/rl/mathgen/comp_math_verl.jsonl}"
REMOTE_VAL_HOST_PATH="${REMOTE_VAL_HOST_PATH:-/lustre/fsw/portfolios/llmservice/users/siddjain/llm/data/rl/mathgen/comp_math_verl.jsonl}"
REWARD_FILE="${REWARD_FILE:-$SCRIPT_ROOT/nemo_verl/reward/verl_code_reward.py}"

DRY_RUN="${DRY_RUN:-1}"
WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-1}"
POLL_SECONDS="${POLL_SECONDS:-30}"
MAX_POLL_SSH_FAILURES="${MAX_POLL_SSH_FAILURES:-5}"
LOCAL_RUN_DIR="${LOCAL_RUN_DIR:-$HOME/data/smoke_tests/opsd_openthoughts_compmath/$EXP_NAME}"
REMOTE_OUTPUT_ROOT="${REMOTE_OUTPUT_ROOT:-/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output}"
REMOTE_OUTPUT="$REMOTE_OUTPUT_ROOT/${OUTPUT_BASE_DIR#/output}/$EXP_NAME"
LOG_DIR="$REMOTE_OUTPUT/training-logs"

die() {
  echo "[error] $*" >&2
  exit 2
}

[[ "$CLUSTER" == "cw-dfw" ]] || die "this smoke wrapper is pinned to cw-dfw after live ranking; got $CLUSTER"
[[ "$PARTITION" == "interactive" ]] || die "smoke tests must use interactive; got $PARTITION"
[[ "$NODES" -le 2 ]] || die "interactive smoke tests support at most 2 nodes; got $NODES"
[[ "$GPUS" -eq 8 ]] || die "cw-dfw full-node jobs require 8 GPUs; got $GPUS"
[[ "$TEMPERATURE" == "1.0" ]] || die "generation temperature must remain 1.0; got $TEMPERATURE"
[[ "$MAX_POLL_SSH_FAILURES" =~ ^[1-9][0-9]*$ ]] \
  || die "MAX_POLL_SSH_FAILURES must be a positive integer"
[[ "$TEACHER_SFT_TARGET_SCOPE" == "thinking_only" || "$TEACHER_SFT_TARGET_SCOPE" == "thinking_and_answer" ]] \
  || die "TEACHER_SFT_TARGET_SCOPE must be thinking_only or thinking_and_answer"
[[ "$ADVANTAGE_SHAPING" == "0" || "$ADVANTAGE_SHAPING" == "1" ]] \
  || die "ADVANTAGE_SHAPING must be 0 or 1"
[[ "$TEACHER_MODEL_MODE" == "actor" || "$TEACHER_MODEL_MODE" == "separate" ]] \
  || die "TEACHER_MODEL_MODE must be actor or separate"
[[ "$OFFPOLICY_IS_MODE" == "none" || "$OFFPOLICY_IS_MODE" == "sequence" || "$OFFPOLICY_IS_MODE" == "token" ]] \
  || die "OFFPOLICY_IS_MODE must be none, sequence, or token"
case "$AUDIT_PROFILE" in
  shared)
    [[ "$TEACHER_MODEL_MODE" == "actor" && "$TEACHER_SFT_WEIGHT" == "0.0" \
      && "$TEACHER_RLVR_BACKWARD_SCALE" == "0.0" && "$RLVR_WARMUP_STEPS" == "0" ]] \
      || die "shared profile requires actor teacher, no teacher objective, and no warmup"
    ;;
  separate_sft)
    [[ "$TEACHER_MODEL_MODE" == "separate" && "$TEACHER_SFT_WEIGHT" != "0.0" \
      && "$TEACHER_RLVR_BACKWARD_SCALE" == "0.0" && "$RLVR_WARMUP_STEPS" == "0" ]] \
      || die "separate_sft profile requires only separate-teacher SFT and no warmup"
    ;;
  separate_rlvr)
    [[ "$TEACHER_MODEL_MODE" == "separate" && "$TEACHER_SFT_WEIGHT" == "0.0" \
      && "$TEACHER_RLVR_BACKWARD_SCALE" != "0.0" && "$RLVR_WARMUP_STEPS" == "0" \
      && "$OFFPOLICY_IS_MODE" == "token" ]] \
      || die "separate_rlvr profile requires only separate-teacher RLVR with token IS"
    ;;
  separate_sft_warmup)
    [[ "$TEACHER_MODEL_MODE" == "separate" && "$TEACHER_SFT_WEIGHT" != "0.0" \
      && "$TEACHER_RLVR_BACKWARD_SCALE" == "0.0" && "$RLVR_WARMUP_STEPS" -gt 0 ]] \
      || die "separate_sft_warmup profile requires separate-teacher SFT and positive warmup"
    ;;
  *) die "unsupported AUDIT_PROFILE=$AUDIT_PROFILE" ;;
esac
[[ "$MAX_TOKENS_PER_GPU" -ge "$MAX_LEN" ]] || die "MAX_TOKENS_PER_GPU must be >= $MAX_LEN"
[[ "$MAX_ACTOR_CKPT_TO_KEEP" -ge 2 ]] \
  || die "composite OPSD smoke checkpoints require MAX_ACTOR_CKPT_TO_KEEP >= 2"
[[ -f "$DATA_PREP_SCRIPT" ]] || die "missing OpenThoughts adapter: $DATA_PREP_SCRIPT"
[[ -f "$REWARD_FILE" ]] || die "missing math reward function: $REWARD_FILE"

mkdir -p "$LOCAL_RUN_DIR" "$LOCAL_DATA_DIR"
SUBMIT_LOG="$LOCAL_RUN_DIR/submit.log"

echo "[setup] preparing $N_PROMPTS OpenThoughts rows from pinned revision $TRAIN_DATASET_REVISION"
"$SKILLS_ENV_PYTHON" "$DATA_PREP_SCRIPT" \
  --dataset "$TRAIN_DATASET" \
  --revision "$TRAIN_DATASET_REVISION" \
  --output "$LOCAL_FIXTURE" \
  --max-samples "$N_PROMPTS" \
  --scan-limit "$TRAIN_DATASET_SCAN_LIMIT" \
  --selection "$TRAIN_DATA_SELECTION" \
  --max-problem-chars "$TRAIN_MAX_PROBLEM_CHARS" \
  --max-solution-chars "$TRAIN_MAX_SOLUTION_CHARS" \
  --max-answer-chars "$TRAIN_MAX_ANSWER_CHARS" \
  --overwrite

# Exercise the same boxed-answer verifier before using cluster time. Every
# source solution must match its Answer target, while an intentionally wrong
# boxed probe must fail. This checks the exact JSON serialization contract the
# reward manager receives.
PYTHONPATH="$VERL_LOCAL_ROOT" "$SKILLS_ENV_PYTHON" - "$LOCAL_FIXTURE" "$REWARD_FILE" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path

fixture = Path(sys.argv[1])
reward_path = Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("opsd_smoke_math_reward", reward_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rows = [json.loads(line) for line in fixture.read_text().splitlines() if line.strip()]
if any("COT_Reason" in row for row in rows):
    raise SystemExit("COT_Reason leaked into the adapted training data")
if any([message["role"] for message in row["messages"]] != ["system", "user"] for row in rows):
    raise SystemExit("adapted actor prompt contains an unexpected assistant turn")
common = {
    "data_sources": [row["data_source"] for row in rows],
    "ground_truths": [row["reward_model"]["ground_truth"] for row in rows],
    "extra_infos": [row["extra_info"] for row in rows],
}
source_results = module.compute_score(solution_strs=[row["solution"] for row in rows], **common)
wrong_results = module.compute_score(
    solution_strs=[r"The answer is \boxed{THIS_IS_INTENTIONALLY_WRONG}." for _ in rows],
    **common,
)
summary = {
    "rows": len(rows),
    "source_row_indices": [row["source_row_index"] for row in rows],
    "answers": [row["Answer"] for row in rows],
    "source_solution_acc": [result["acc"] for result in source_results],
    "source_solution_pred": [result["pred"] for result in source_results],
    "wrong_probe_acc": [result["acc"] for result in wrong_results],
}
print("[OPSD_DATA_REWARD_PREFLIGHT] " + json.dumps(summary, sort_keys=True), flush=True)
if any(result["acc"] != 1.0 for result in source_results):
    raise SystemExit("a source solution did not verify against its Answer field")
if any(result["acc"] != 0.0 for result in wrong_results):
    raise SystemExit("an intentionally wrong boxed answer passed verification")
PY

echo "[setup] syncing the adapted OpenThoughts smoke data to $REMOTE_HOST:$REMOTE_DATA_HOST_PATH"
ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_DATA_HOST_DIR'"
rsync -a "$LOCAL_FIXTURE" "$REMOTE_HOST:$REMOTE_DATA_HOST_PATH"
local_sha="$(sha256sum "$LOCAL_FIXTURE" | awk '{print $1}')"
remote_sha="$(ssh "$REMOTE_HOST" "sha256sum '$REMOTE_DATA_HOST_PATH' | awk '{print \$1}'")"
[[ "$local_sha" == "$remote_sha" ]] || die "fixture checksum mismatch after sync"
ssh "$REMOTE_HOST" "test -f '$REMOTE_VAL_HOST_PATH'" \
  || die "missing CompMath eval data on $REMOTE_HOST: $REMOTE_VAL_HOST_PATH"

# Use the ordinary boxed-answer RLVR reward. Uniform all-correct/all-wrong
# groups intentionally retain their standard zero GRPO advantage; do not
# replace them with a synthetic response-logprob reward.
if [[ "$ADVANTAGE_SHAPING" == "1" ]]; then
  objective_args=$(cat <<EOF
algorithm.opsd.mix_weight=1.0
algorithm.opsd.distill_backward_scale=0.0
algorithm.opsd.rlvr_backward_scale=$TEACHER_RLVR_BACKWARD_SCALE
algorithm.opsd.advantage_shaping.enable=true
algorithm.opsd.advantage_shaping.score_source=teacher_minus_student_logprob
algorithm.opsd.advantage_shaping.scale=$ADVANTAGE_SHAPING_SCALE
algorithm.opsd.advantage_shaping.normalize=std
algorithm.opsd.advantage_shaping.clip_z=3.0
algorithm.opsd.advantage_shaping.use_distill_mask=true
algorithm.opsd.advantage_shaping.allow_token_sign_flip=false
algorithm.opsd.advantage_shaping.max_response_tokens=$ADVANTAGE_SHAPING_MAX_RESPONSE_TOKENS
algorithm.opsd.advantage_shaping.student_rlvr_backward_scale=1.0
EOF
)
else
  objective_args=$(cat <<EOF
algorithm.opsd.mix_weight=0.5
algorithm.opsd.distill_backward_scale=1.0
algorithm.opsd.rlvr_backward_scale=1.0
algorithm.opsd.advantage_shaping.enable=false
EOF
)
fi
extra_args=$(cat <<EOF
algorithm.opsd.mode=opsd_rlvr
algorithm.opsd.teacher_source=ground_truth
algorithm.opsd.teacher_model=$TEACHER_MODEL_MODE
algorithm.opsd.distill_loss=sampled_reverse_kl
algorithm.opsd.topk=null
algorithm.opsd.distill_beta=null
algorithm.opsd.distill_token_clip=null
algorithm.opsd.distill_token_clip_tail=null
algorithm.opsd.balance_mode=none
$objective_args
algorithm.opsd.rlvr_warmup_steps=$RLVR_WARMUP_STEPS
algorithm.opsd.teacher_sft_weight=$TEACHER_SFT_WEIGHT
algorithm.opsd.teacher_sft_target_scope=$TEACHER_SFT_TARGET_SCOPE
algorithm.opsd.teacher_sft_success_field=acc
algorithm.opsd.teacher_sft_success_threshold=0.5
algorithm.opsd.offpolicy_is_mode=$OFFPOLICY_IS_MODE
algorithm.opsd.behavior_logprob_source=rollout
algorithm.opsd.audit.enabled=true
algorithm.opsd.audit.output_dir=$OUTPUT_BASE_DIR/$EXP_NAME/opsd_audit
algorithm.opsd.audit.global_steps=$AUDIT_GLOBAL_STEPS
algorithm.opsd.audit.fail_fast=$AUDIT_FAIL_FAST
algorithm.opsd.audit.full_batch_ledger=true
algorithm.opsd.audit.reference_forward=true
algorithm.opsd.audit.reference_samples_per_rank=1
algorithm.opsd.max_prompt_length=$MAX_PROMPT_LEN
algorithm.opsd.truncation=error
algorithm.opsd.ground_truth_field=solution
actor_rollout_ref.opsd_teacher.model.path=$TEACHER_MODEL
actor_rollout_ref.opsd_teacher.optim.lr=$TEACHER_LR
actor_rollout_ref.rollout.gpu_memory_utilization=0.4
actor_rollout_ref.rollout.max_model_len=$MAX_LEN
actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_LEN
data.train_max_samples=$N_PROMPTS
data.val_max_samples=8
data.filter_overlong_prompts=True
data.filter_overlong_prompts_workers=1
data.dataloader_num_workers=0
++data.dynamic_masked_solution=False
++data.min_masked_fraction=null
++data.max_masked_fraction=null
++data.mask_seed=null
trainer.total_training_steps=$TOTAL_STEPS
trainer.logger=['console']
trainer.log_val_generations=0
trainer.save_freq=$SAVE_FREQ
trainer.max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP
EOF
)
extra_args="$(printf '%s' "$extra_args" | tr '\n' ' ')"

cmd=(
  "$SKILLS_ENV_PYTHON" "$LAUNCHER"
  --cluster "$CLUSTER"
  --config_dir "$CONFIG_DIR"
  --output_base_dir "$OUTPUT_BASE_DIR"
  --local_verl_folder "$VERL_LOCAL_ROOT"
  --verl_config_file "$VERL_LOCAL_ROOT/recipe/opsd/config/opsd_trainer.yaml"
  --script_module recipe.opsd.main_opsd
  --reward_file "$REWARD_FILE"
  --ground_truth_solution_key solution
  --expname "$EXP_NAME"
  --partition "$PARTITION"
  --time_limit "$TIME_LIMIT"
  --nodes "$NODES"
  --gpus "$GPUS"
  --actor_model "$ACTOR_MODEL"
  --prompt_data "$TRAIN_DATA"
  --eval_data "$VAL_DATA"
  --n_prompts "$N_PROMPTS"
  --n_samples "$N_SAMPLES"
  --n_val_samples "$N_VAL_SAMPLES"
  --val_batch_size "$N_PROMPTS"
  --max_prompt_len "$MAX_PROMPT_LEN"
  --max_len "$MAX_LEN"
  --max_tokens_per_gpu "$MAX_TOKENS_PER_GPU"
  --num_epochs "$TOTAL_STEPS"
  --num_training_jobs 1
  --num_ppo_iter 1
  --actor_lr "$ACTOR_LR"
  --clip_ae 0.2,0.28
  --infer_server vllm
  --sequence_parallel_size 1
  --T "$TEMPERATURE"
  --val_T 1.0
  --val_top_p 1.0
  --save_freq "$SAVE_FREQ"
  --test_freq -1
  --ae grpo
  --no_sandbox
  --omit_noncore_algorithm_overrides
  --extra_args "$extra_args"
)
[[ "$DRY_RUN" != "0" ]] && cmd+=(--dry_run)

echo "[run] cluster=$CLUSTER account=$ACCOUNT partition=$PARTITION time=$TIME_LIMIT nodes=$NODES gpus=$GPUS"
echo "[run] actor=$ACTOR_MODEL teacher=$TEACHER_MODEL prompt=$MAX_PROMPT_LEN response=$MAX_RESPONSE_LEN T=$TEMPERATURE"
echo "[run] steps=$TOTAL_STEPS warmup_steps=$RLVR_WARMUP_STEPS output=$REMOTE_OUTPUT"
echo "[run] save_freq=$SAVE_FREQ max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP"
echo "[run] teacher_mode=$TEACHER_MODEL_MODE teacher_rlvr_scale=$TEACHER_RLVR_BACKWARD_SCALE teacher_sft_weight=$TEACHER_SFT_WEIGHT teacher_sft_target_scope=$TEACHER_SFT_TARGET_SCOPE"
echo "[run] audit_profile=$AUDIT_PROFILE audit_steps=$AUDIT_GLOBAL_STEPS offpolicy_is_mode=$OFFPOLICY_IS_MODE"
echo "[run] advantage_shaping=$ADVANTAGE_SHAPING scale=$ADVANTAGE_SHAPING_SCALE max_response_tokens=$ADVANTAGE_SHAPING_MAX_RESPONSE_TOKENS"
echo "[run] train_dataset=$TRAIN_DATASET@$TRAIN_DATASET_REVISION train=$TRAIN_DATA eval=$VAL_DATA"
printf '[command] '
printf '%q ' "${cmd[@]}"
printf '\n'

set +e
"${cmd[@]}" 2>&1 | tee "$SUBMIT_LOG"
rc=${PIPESTATUS[0]}
set -e
[[ "$rc" -eq 0 ]] || die "launcher failed; see $SUBMIT_LOG"
if [[ "$DRY_RUN" != "0" ]]; then
  echo "[dry-run] submission was not performed"
  exit 0
fi

job_id="$(grep -Eo 'slurm_tunnel://nemo_run/[0-9]+' "$SUBMIT_LOG" | tail -1 | grep -Eo '[0-9]+' || true)"
if [[ -z "$job_id" ]]; then
  job_id="$(grep -Eo 'Submitted batch job [0-9]+' "$SUBMIT_LOG" | tail -1 | awk '{print $4}' || true)"
fi
[[ -n "$job_id" ]] || die "could not parse a Slurm job ID from $SUBMIT_LOG"
echo "$job_id" > "$LOCAL_RUN_DIR/job_id.txt"
echo "[submitted] job_id=$job_id log_dir=$LOG_DIR"

if [[ "$WAIT_FOR_COMPLETION" == "0" ]]; then
  exit 0
fi

poll_ssh_failures=0
while true; do
  set +e
  live_state="$(ssh "$REMOTE_HOST" "squeue -h -j '$job_id' -o '%T' | head -1")"
  live_rc=$?
  set -e
  if [[ "$live_rc" -ne 0 ]]; then
    poll_ssh_failures=$((poll_ssh_failures + 1))
    echo "[wait] job=$job_id SSH poll failed ($poll_ssh_failures/$MAX_POLL_SSH_FAILURES); retrying"
    [[ "$poll_ssh_failures" -lt "$MAX_POLL_SSH_FAILURES" ]] \
      || die "lost SSH polling for $MAX_POLL_SSH_FAILURES consecutive attempts"
    sleep "$POLL_SECONDS"
    continue
  fi
  if [[ -n "$live_state" ]]; then
    poll_ssh_failures=0
    echo "[wait] job=$job_id state=$live_state"
    sleep "$POLL_SECONDS"
    continue
  fi

  set +e
  final_state="$(ssh "$REMOTE_HOST" "sacct -j '$job_id' --format=State -n -P | head -1 | tr -d ' '")"
  final_rc=$?
  set -e
  if [[ "$final_rc" -ne 0 ]]; then
    poll_ssh_failures=$((poll_ssh_failures + 1))
    echo "[wait] job=$job_id final-state SSH poll failed ($poll_ssh_failures/$MAX_POLL_SSH_FAILURES); retrying"
    [[ "$poll_ssh_failures" -lt "$MAX_POLL_SSH_FAILURES" ]] \
      || die "lost SSH polling for $MAX_POLL_SSH_FAILURES consecutive attempts"
    sleep "$POLL_SECONDS"
    continue
  fi
  if [[ -z "$final_state" ]]; then
    poll_ssh_failures=$((poll_ssh_failures + 1))
    echo "[wait] job=$job_id final state is not visible yet ($poll_ssh_failures/$MAX_POLL_SSH_FAILURES); retrying"
    [[ "$poll_ssh_failures" -lt "$MAX_POLL_SSH_FAILURES" ]] \
      || die "Slurm final state remained unavailable for $MAX_POLL_SSH_FAILURES attempts"
    sleep "$POLL_SECONDS"
    continue
  fi
  poll_ssh_failures=0
  echo "[wait] job=$job_id final_state=${final_state:-unknown}"
  [[ "$final_state" == COMPLETED* ]] || die "job did not complete; inspect $LOG_DIR"
  break
done

required_patterns=(
  '[OPSD_AUDIT_LAYOUT]'
  '[OPSD_AUDIT_WORKER_LAYOUT]'
  '[OPSD_AUDIT_VALUES]'
  'actor/opsd_reverse_kl_loss'
  'actor/opsd_reverse_kl_estimate'
  'actor/opsd_reverse_kl_surrogate_loss'
  'actor/teacher_rlvr_loss'
  'actor/teacher_sft_loss'
  'actor/teacher_sft_weight'
  'actor/optimizer_step_skipped'
  'reward/acc/mean'
  'val-agg/'
)
if [[ "$AUDIT_PROFILE" != "shared" ]]; then
  required_patterns+=(
    '[OPSD_TEACHER_INIT]'
    'actor/teacher_grad_norm'
  )
fi
if [[ "$AUDIT_PROFILE" == "separate_sft" || "$AUDIT_PROFILE" == "separate_sft_warmup" ]]; then
  required_patterns+=(
    'actor/teacher_sft_target_tokens'
  )
fi
if [[ "$ADVANTAGE_SHAPING" == "1" ]]; then
  required_patterns+=(
    'actor/advantage_shaping_active_rate'
    'actor/advantage_shaping_token_count'
    'actor/advantage_shaping_prompt_token_count:0.0'
    'actor/advantage_shaping_pad_token_count:0.0'
    'actor/advantage_shaping_pad_delta_max:0.0'
    'actor/advantage_shaping_total_error_max'
    'actor/student_rlvr_loss'
    'actor/student_rlvr_weight'
  )
fi
for pattern in "${required_patterns[@]}"; do
  if ! ssh "$REMOTE_HOST" "grep -RF '$pattern' '$LOG_DIR' >/dev/null 2>&1"; then
    die "completed job is missing expected metric $pattern in $LOG_DIR"
  fi
done

AUDIT_REMOTE_DIR="$REMOTE_OUTPUT/opsd_audit"
AUDIT_LOCAL_DIR="$LOCAL_RUN_DIR/opsd_audit"
mkdir -p "$AUDIT_LOCAL_DIR"
rsync -a "$REMOTE_HOST:$AUDIT_REMOTE_DIR/" "$AUDIT_LOCAL_DIR/"
if [[ "$ADVANTAGE_SHAPING" == "1" ]]; then
  verifier_args=(
    python3 "$VERL_LOCAL_ROOT/smoke_tests/opsd_advantage_shaping/verify_advantage_shaping_audit.py"
    "$AUDIT_LOCAL_DIR"
    --profile "$AUDIT_PROFILE"
    --require-response-axis "$MAX_RESPONSE_LEN"
  )
  for step in $AUDIT_EXPECT_WARMUP_STEPS; do
    verifier_args+=(--expect-warmup-step "$step")
  done
  for step in $AUDIT_EXPECT_JOINT_STEPS; do
    verifier_args+=(--expect-joint-step "$step")
  done
  "${verifier_args[@]}"
else
  python3 "$VERL_LOCAL_ROOT/smoke_tests/opsd_audit/verify_opsd_audit.py" "$AUDIT_LOCAL_DIR"
fi

audit_required_fields=(
  'actor_input_ids'
  'teacher_input_ids'
  'response_attention_mask'
  'distill_global_info'
  'rlvr_global_info'
  'reverse_kl_token_estimate'
  'distill_token_surrogate'
  'teacher_behavior_sequence_log_ratio'
  'teacher_is_mode'
  'teacher_ppo_terms'
  'teacher_sft_mask'
  'teacher_sft_success'
  'teacher_sft_think_end_exclusive'
  'teacher_sft_token_nll'
  'teacher_sft_loss_production'
  'teacher_sft_global_info'
  'gradient_routing'
  'reference_forwards'
  'actor_did_step'
  'teacher_did_step'
  'optimizer_updates_completed_before_step'
  'trainer_updates_completed_before_step'
  'source_dataset'
  'source_solution'
  'source_answer'
  'source_cot_reason_present'
  'reward_ground_truth'
  'ground_truth_field'
)
if [[ "$ADVANTAGE_SHAPING" == "1" ]]; then
  audit_required_fields+=(
    'advantages_shaped'
    'teacher_evidence_scores'
    'advantage_shaping_mask'
    'student_ppo_terms'
    'student_rlvr_loss_production'
    'student_rlvr_weight'
    'student_shaped_rlvr_to_actor'
  )
  if [[ "$TEACHER_MODEL_MODE" == "separate" ]]; then
    audit_required_fields+=(
      'student_shaped_rlvr_to_teacher'
    )
  fi
fi
for field in "${audit_required_fields[@]}"; do
  if ! grep -R -q "\"$field\"" "$AUDIT_LOCAL_DIR"; then
    die "audit artifacts are missing required field $field"
  fi
done

python3 - "$AUDIT_LOCAL_DIR/verification_report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
if report.get("status") != "PASS":
    raise SystemExit("advantage-shaping audit verifier did not pass")
observations = report.get("observations", {})
if observations.get("shaped_tokens", 0) <= 0:
    raise SystemExit("advantage-shaping audit observed no shaped response tokens")
print(
    "[verified] "
    f"profile={report['profile']} response_axes={observations.get('response_axis_lengths')} "
    f"shaped_tokens={observations.get('shaped_tokens')} "
    f"nonzero_delta_tokens={observations.get('nonzero_shaping_delta_tokens')} "
    f"actor_steps={observations.get('actor_optimizer_steps')} "
    f"teacher_steps={observations.get('teacher_optimizer_steps')}"
)
PY

FINAL_STEP_METRICS="$LOCAL_RUN_DIR/final_step_metrics.log"
ssh "$REMOTE_HOST" "grep -R 'training/global_step:$TOTAL_STEPS' '$LOG_DIR' | tail -1" > "$FINAL_STEP_METRICS"
python3 - "$FINAL_STEP_METRICS" "$LOCAL_RUN_DIR/runtime_summary.json" "$TOTAL_STEPS" <<'PY'
import json
import re
import sys
from pathlib import Path

source = Path(sys.argv[1])
output = Path(sys.argv[2])
text = source.read_text()
keys = (
    "timing_s/gen",
    "timing_s/update_actor",
    "timing_s/save_checkpoint",
    "timing_s/step",
    "response_length/max",
    "response_length/clip_ratio",
)
metrics = {}
for key in keys:
    match = re.search(rf"(?:^| - ){re.escape(key)}:([^ ]+)", text)
    metrics[key] = None if match is None else float(match.group(1))
payload = {"global_step": int(sys.argv[3]), "metrics": metrics}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print("[timing] " + json.dumps(payload, sort_keys=True))
PY

if [[ "$SAVE_FREQ" -gt 0 ]]; then
  ssh "$REMOTE_HOST" "test \"\$(cat '$REMOTE_OUTPUT/checkpoints/latest_checkpointed_iteration.txt')\" = '$TOTAL_STEPS'" \
    || die "final checkpoint tracker did not reach step $TOTAL_STEPS"
  if [[ "$TEACHER_MODEL_MODE" == "separate" ]]; then
    ssh "$REMOTE_HOST" "test -d '$REMOTE_OUTPUT/checkpoints/global_step_$TOTAL_STEPS/actor/opsd_teacher'" \
      || die "final separate-teacher composite checkpoint is incomplete"
  fi
fi

echo "[result] PASS job=$job_id state=$final_state"
echo "[result] data_contract=solution->teacher Answer->boxed_verifier COT_Reason->omitted eval=CompMath"
echo "[result] advantage_shaping=$ADVANTAGE_SHAPING audit_profile=$AUDIT_PROFILE"
echo "[result] remote_output=$REMOTE_OUTPUT"
echo "[result] audit_local_dir=$AUDIT_LOCAL_DIR"
ssh "$REMOTE_HOST" \
  "grep -R 'actor/opsd_reverse_kl_loss\|actor/advantage_shaping\|actor/student_rlvr_loss\|actor/teacher_rlvr_loss\|actor/teacher_sft_loss\|actor/teacher_sft_target_tokens\|actor/teacher_grad_norm\|actor/optimizer_step_skipped' '$LOG_DIR' | tail -40" \
  || true
