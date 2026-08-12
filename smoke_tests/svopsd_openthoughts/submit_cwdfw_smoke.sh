#!/usr/bin/env bash
set -euo pipefail

# Dry-run-first live smoke launcher for strict steering-vector OPSD.  The
# audit smoke is small enough for one interactive node and writes exhaustive
# ledgers.  The profile smoke uses the exact two-node production batch and
# sequence lengths for five updates, with bounded token-level KL logging.

ACTION="${1:-dry-run}"
SHAPE="${2:-audit}"
SCRIPT_ROOT="${SCRIPT_ROOT:-/home/siddjain/workspace/scripts/src}"
VERL_ROOT="${VERL_ROOT:-/home/siddjain/workspace/verl/verl_main}"
PYTHON="${PYTHON:-/home/siddjain/anaconda3/envs/skills_latest/bin/python}"
LAUNCHER="${LAUNCHER:-$SCRIPT_ROOT/nemo_verl/skills_verl_submit.py}"
REWARD_FILE="${REWARD_FILE:-$SCRIPT_ROOT/nemo_verl/reward/verl_code_reward.py}"
CONFIG_DIR="${CONFIG_DIR:-/home/siddjain/workspace/scripts/nemo_configs/cluster/codegen}"
VERL_CONFIG="${VERL_CONFIG:-$VERL_ROOT/recipe/opsd/config/opsd_trainer.yaml}"
VERIFY_SCRIPT="${VERIFY_SCRIPT:-$VERL_ROOT/smoke_tests/svopsd_openthoughts/verify_steering_audit.py}"
VERIFY_GAP_SCRIPT="${VERIFY_GAP_SCRIPT:-$VERL_ROOT/smoke_tests/svopsd_gap/verify_gap_ledger.py}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SV_ACTOR_OBJECTIVE="${SV_ACTOR_OBJECTIVE:-direct_reverse_kl}"
SV_STEERING_SCALE="${SV_STEERING_SCALE:-1.0}"
SV_MAX_DELTA_FRACTION="${SV_MAX_DELTA_FRACTION:-1.0}"
SV_CAA_SCOPE="${SV_CAA_SCOPE:-same_prompt}"
SV_STEERING_SOURCE_MODE="${SV_STEERING_SOURCE_MODE:-caa}"
ACTOR_MODEL="${ACTOR_MODEL:-/hf_models/Qwen3-1.7B}"
EXPECTED_ACTOR_MODEL="/hf_models/Qwen3-1.7B"
TRAIN_DATA="${TRAIN_DATA:-/data/rl/opsd_openthoughts_compmath/openthoughts_math_30k_opsd_full.jsonl}"
VAL_DATA="${VAL_DATA:-/data/rl/mathgen/comp_math_verl.jsonl}"
REMOTE_TRAIN_DATA="${REMOTE_TRAIN_DATA:-/lustre/fsw/portfolios/llmservice/users/siddjain/llm/data/rl/opsd_openthoughts_compmath/openthoughts_math_30k_opsd_full.jsonl}"
REMOTE_VAL_DATA="${REMOTE_VAL_DATA:-/lustre/fsw/portfolios/llmservice/users/siddjain/llm/data/rl/mathgen/comp_math_verl.jsonl}"
EXPECTED_ROWS="${EXPECTED_ROWS:-29427}"
EXPECTED_SHA256="${EXPECTED_SHA256:-f79a42fe155218db2f1927ee903afd101929724f2d0516352bdbb91cdb139178}"

case "$SV_ACTOR_OBJECTIVE" in
  vanilla_grpo)
    OPSD_ENABLE=false
    CONFIG_ACTOR_OBJECTIVE=direct_reverse_kl
    OPSD_MODE=opsd
    DISTILL_BACKWARD_SCALE=1.0
    ADVANTAGE_SHAPING_ENABLED=false
    ;;
  direct_reverse_kl)
    OPSD_ENABLE=true
    CONFIG_ACTOR_OBJECTIVE=$SV_ACTOR_OBJECTIVE
    OPSD_MODE=opsd
    DISTILL_BACKWARD_SCALE=1.0
    ADVANTAGE_SHAPING_ENABLED=false
    ;;
  negative_kl_advantage)
    OPSD_ENABLE=true
    CONFIG_ACTOR_OBJECTIVE=$SV_ACTOR_OBJECTIVE
    OPSD_MODE=opsd
    DISTILL_BACKWARD_SCALE=1.0
    ADVANTAGE_SHAPING_ENABLED=false
    ;;
  grpo_advantage_reweighting)
    OPSD_ENABLE=true
    CONFIG_ACTOR_OBJECTIVE=$SV_ACTOR_OBJECTIVE
    OPSD_MODE=opsd_rlvr
    DISTILL_BACKWARD_SCALE=0.0
    ADVANTAGE_SHAPING_ENABLED=true
    ;;
  *)
    echo "[error] SV_ACTOR_OBJECTIVE must be vanilla_grpo, direct_reverse_kl, negative_kl_advantage, or grpo_advantage_reweighting" >&2
    exit 2
    ;;
esac

case "$SHAPE" in
  audit)
    # A batch-global CAA audit must cross a physical node boundary so the
    # verifier exercises and reconstructs the real distributed all-reduce.
    if [[ "$SV_CAA_SCOPE" == global_batch ]]; then NODES=2; else NODES=1; fi
    if [[ "$SV_CAA_SCOPE" == global_batch ]]; then N_PROMPTS=16; else N_PROMPTS=8; fi
    N_SAMPLES=8
    MAX_PROMPT_LEN=4096
    MAX_RESPONSE_LEN=8192
    TOTAL_STEPS=1
    AUDIT_ENABLED=true
    VAL_BEFORE_TRAIN=false
    ;;
  gap)
    NODES=1
    N_PROMPTS=16
    N_SAMPLES=8
    MAX_PROMPT_LEN=4096
    MAX_RESPONSE_LEN=8192
    TOTAL_STEPS=1
    AUDIT_ENABLED=true
    VAL_BEFORE_TRAIN=false
    ;;
  profile)
    NODES=2
    N_PROMPTS=64
    N_SAMPLES=8
    MAX_PROMPT_LEN=4096
    MAX_RESPONSE_LEN=8192
    TOTAL_STEPS=5
    AUDIT_ENABLED=false
    VAL_BEFORE_TRAIN=false
    ;;
  *) echo "[error] SHAPE must be audit, gap, or profile" >&2; exit 2 ;;
esac
if [[ "$SV_ACTOR_OBJECTIVE" == vanilla_grpo ]]; then
  AUDIT_ENABLED=false
fi

EXP_NAME="svopsd_openthoughts_qwen3_1p7b_${SV_STEERING_SOURCE_MODE}_${SV_ACTOR_OBJECTIVE}_${SHAPE}_${RUN_TAG}"
OUTPUT_BASE="${OUTPUT_BASE:-/output/smoke_tests/svopsd_openthoughts_${SV_STEERING_SOURCE_MODE}_${SV_ACTOR_OBJECTIVE}_${RUN_TAG}}"
REMOTE_OUTPUT_BASE="${REMOTE_OUTPUT_BASE:-/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/${OUTPUT_BASE#/output/}}"
REMOTE_EXP_DIR="$REMOTE_OUTPUT_BASE/$EXP_NAME"
LOCAL_RUN_DIR="${LOCAL_RUN_DIR:-/home/siddjain/data/smoke_tests/svopsd_openthoughts/${RUN_TAG}_${SV_STEERING_SOURCE_MODE}_${SV_ACTOR_OBJECTIVE}_${SHAPE}}"
MODEL_MAX_LEN=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))

die() { echo "[error] $*" >&2; exit 2; }

preflight() {
  [[ "$ACTOR_MODEL" == "$EXPECTED_ACTOR_MODEL" ]] || \
    die "this audited smoke requires actor model $EXPECTED_ACTOR_MODEL, got $ACTOR_MODEL"
  [[ "$SV_CAA_SCOPE" == same_prompt || "$SV_CAA_SCOPE" == global_batch ]] || \
    die "SV_CAA_SCOPE must be same_prompt or global_batch"
  [[ "$SV_STEERING_SOURCE_MODE" == caa || "$SV_STEERING_SOURCE_MODE" == policy_gradient ]] || \
    die "SV_STEERING_SOURCE_MODE must be caa or policy_gradient"
  if [[ "$SV_STEERING_SOURCE_MODE" == policy_gradient && "$SV_CAA_SCOPE" != global_batch ]]; then
    die "policy-gradient steering requires SV_CAA_SCOPE=global_batch"
  fi
  [[ -f "$LAUNCHER" && -f "$REWARD_FILE" && -f "$VERL_CONFIG" && -f "$VERIFY_SCRIPT" && -f "$VERIFY_GAP_SCRIPT" ]] \
    || die "required local implementation or launcher is missing"
  local rows sha
  rows="$(ssh dfw "wc -l < '$REMOTE_TRAIN_DATA'")"
  sha="$(ssh dfw "sha256sum '$REMOTE_TRAIN_DATA'" | awk '{print $1}')"
  [[ "$rows" == "$EXPECTED_ROWS" ]] || die "training rows $rows != $EXPECTED_ROWS"
  [[ "$sha" == "$EXPECTED_SHA256" ]] || die "training SHA-256 mismatch: $sha"
  ssh dfw "test -s '$REMOTE_VAL_DATA'"
  mkdir -p "$LOCAL_RUN_DIR"
  echo "[preflight] shape=$SHAPE nodes=$NODES actor_objective=$SV_ACTOR_OBJECTIVE steering_source=$SV_STEERING_SOURCE_MODE caa_scope=$SV_CAA_SCOPE rows=$rows sha256=$sha output=$REMOTE_EXP_DIR"
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
trainer.total_training_steps=$TOTAL_STEPS
trainer.total_epochs=1
trainer.logger=['console']
trainer.val_before_train=$VAL_BEFORE_TRAIN
trainer.save_freq=-1
trainer.test_freq=-1
trainer.log_val_generations=0
algorithm.use_kl_in_reward=False
EOF
  if [[ "$SV_ACTOR_OBJECTIVE" == vanilla_grpo ]]; then
    return
  fi
  cat <<EOF
algorithm.opsd.enable=$OPSD_ENABLE
algorithm.opsd.mode=$OPSD_MODE
algorithm.opsd.actor_objective=$CONFIG_ACTOR_OBJECTIVE
algorithm.opsd.teacher_model=actor
algorithm.opsd.teacher_source=sdpo_success_rollout
algorithm.opsd.sdpo_conditioning_mode=steering
algorithm.opsd.sdpo_distill_only_failed=false
algorithm.opsd.sdpo_exclude_self_success=true
algorithm.opsd.distill_loss=sampled_reverse_kl
algorithm.opsd.topk=null
algorithm.opsd.distill_beta=null
algorithm.opsd.distill_token_clip=null
algorithm.opsd.distill_token_clip_tail=null
algorithm.opsd.distill_max_response_tokens=null
algorithm.opsd.balance_mode=none
algorithm.opsd.mix_weight=1.0
algorithm.opsd.distill_backward_scale=$DISTILL_BACKWARD_SCALE
algorithm.opsd.rlvr_backward_scale=0.0
algorithm.opsd.rlvr_warmup_steps=0
algorithm.opsd.teacher_sft_weight=0.0
algorithm.opsd.advantage_shaping.enable=$ADVANTAGE_SHAPING_ENABLED
algorithm.opsd.advantage_shaping.scale=1.0
algorithm.opsd.advantage_shaping.normalize=null
algorithm.opsd.advantage_shaping.clip_z=null
algorithm.opsd.advantage_shaping.use_distill_mask=true
algorithm.opsd.advantage_shaping.allow_token_sign_flip=false
algorithm.opsd.advantage_shaping.max_delta_fraction=$SV_MAX_DELTA_FRACTION
algorithm.opsd.advantage_shaping.max_response_tokens=null
algorithm.opsd.advantage_shaping.student_rlvr_backward_scale=1.0
algorithm.opsd.steering.strict_contract=true
algorithm.opsd.steering.source_mode=$SV_STEERING_SOURCE_MODE
algorithm.opsd.steering.correct_rollout_aggregation=all
algorithm.opsd.steering.activation_aggregation=per_rollout
algorithm.opsd.steering.gradient_objective=grpo_advantage
algorithm.opsd.steering.gradient_aggregation=per_rollout
algorithm.opsd.steering.caa_scope=$SV_CAA_SCOPE
algorithm.opsd.steering.layer_fractions="0.31-0.37"
algorithm.opsd.steering.expected_model_path="/hf_models/Qwen3-1.7B"
algorithm.opsd.steering.expected_total_layers=28
algorithm.opsd.steering.expected_layer_indices=[9,10]
algorithm.opsd.steering.scale=$SV_STEERING_SCALE
algorithm.opsd.steering.normalize=unit_norm
algorithm.opsd.steering.apply_positions=response_only
algorithm.opsd.steering.detach_vectors=true
algorithm.opsd.steering.gap_diagnostics.enabled=$([[ "$SHAPE" == gap ]] && echo true || echo false)
algorithm.opsd.steering.gap_diagnostics.output_dir=null
algorithm.opsd.steering.gap_diagnostics.interval_steps=1
algorithm.opsd.steering.gap_diagnostics.crossfit_enabled=true
algorithm.opsd.steering.gap_diagnostics.fold_seed=1234
algorithm.opsd.steering.gap_diagnostics.full_sequence_ledger=true
algorithm.opsd.audit.enabled=$AUDIT_ENABLED
algorithm.opsd.audit.global_steps=[1]
algorithm.opsd.audit.full_batch_ledger=true
algorithm.opsd.audit.reference_forward=true
algorithm.opsd.audit.reference_samples_per_rank=1
algorithm.opsd.audit.fail_fast=true
algorithm.opsd.token_kl_logging.enabled=true
algorithm.opsd.token_kl_logging.start_step=1
algorithm.opsd.token_kl_logging.end_step=null
algorithm.opsd.token_kl_logging.interval_steps=1
algorithm.opsd.token_kl_logging.max_samples_per_rank=1
algorithm.opsd.token_kl_logging.max_tokens_per_sample=128
algorithm.opsd.max_prompt_length=$MAX_PROMPT_LEN
algorithm.opsd.truncation=error
algorithm.opsd.ground_truth_field=solution
EOF
}

submit() {
  local dry_run="$1"
  local flattened script_module config_file
  flattened="$(extra_args | tr '\n' ' ')"
  if [[ "$SV_ACTOR_OBJECTIVE" == vanilla_grpo ]]; then
    script_module="verl.trainer.main_ppo"
    config_file=""
  else
    script_module="recipe.opsd.main_opsd"
    config_file="$VERL_CONFIG"
  fi
  local -a cmd=(
    "$PYTHON" "$LAUNCHER"
    --cluster cw-dfw
    --config_dir "$CONFIG_DIR"
    --output_base_dir "$OUTPUT_BASE"
    --local_verl_folder "$VERL_ROOT"
    --script_module "$script_module"
    --reward_file "$REWARD_FILE"
    --ground_truth_solution_key solution
    --expname "$EXP_NAME"
    --partition interactive
    --time_limit 04:00:00
    --nodes "$NODES"
    --gpus 8
    --actor_model "$ACTOR_MODEL"
    --prompt_data "$TRAIN_DATA"
    --eval_data "$VAL_DATA"
    --n_prompts "$N_PROMPTS"
    --n_samples "$N_SAMPLES"
    --n_val_samples 4
    --val_batch_size "$N_PROMPTS"
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
    --disable_val_before_train
    --extra_args "$flattened"
  )
  [[ -n "$config_file" ]] && cmd+=(--verl_config_file "$config_file")
  [[ "$dry_run" == 1 ]] && cmd+=(--dry_run)
  {
    printf '#!/usr/bin/env bash\n'
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } > "$LOCAL_RUN_DIR/command.sh"
  chmod +x "$LOCAL_RUN_DIR/command.sh"
  "${cmd[@]}" 2>&1 | tee "$LOCAL_RUN_DIR/submit.log"
  if [[ "$dry_run" == 0 ]]; then
    local job_id
    job_id="$(grep -Eo 'slurm_tunnel://nemo_run/[0-9]+' "$LOCAL_RUN_DIR/submit.log" | tail -1 | grep -Eo '[0-9]+' || true)"
    [[ -n "$job_id" ]] || job_id="$(grep -Eo 'Submitted batch job [0-9]+' "$LOCAL_RUN_DIR/submit.log" | tail -1 | awk '{print $4}' || true)"
    [[ -n "$job_id" ]] || die "could not parse submitted job ID"
    printf '%s\n' "$job_id" > "$LOCAL_RUN_DIR/job_id"
    # NeMo Run currently renders Requeue=1 even when the submit utility is
    # passed --no_requeue. Correct the live Slurm record before accepting the
    # submission, then verify the effective state below.
    local verified=0
    for _ in 1 2 3 4 5; do
      if ssh dfw "scontrol update JobId='$job_id' Requeue=0; scontrol show job '$job_id' -o" \
        > "$LOCAL_RUN_DIR/scontrol.txt"; then
        verified=1
        break
      fi
      sleep 2
    done
    [[ "$verified" == 1 ]] || die "could not verify live Slurm record for $job_id"
    grep -q 'Requeue=0' "$LOCAL_RUN_DIR/scontrol.txt" || die "job is requeueable"
    grep -q 'Partition=interactive' "$LOCAL_RUN_DIR/scontrol.txt" || die "job is not interactive"
    echo "[submitted] job=$job_id output=$REMOTE_EXP_DIR"
  fi
}

status() {
  local job_id
  job_id="$(cat "$LOCAL_RUN_DIR/job_id")"
  ssh dfw "squeue -j '$job_id' -o '%.18i %.12P %.24j %.10T %.10M %.10l %R'; sacct -X -j '$job_id' -o JobID,State,Elapsed,Start,End,ExitCode -P"
}

verify() {
  [[ "$SHAPE" == audit || "$SHAPE" == gap ]] || \
    die "structured verifier is only enabled for audit or gap smoke"
  local report_dir="$LOCAL_RUN_DIR/manual_audit"
  mkdir -p "$report_dir"
  rsync -a "dfw:$REMOTE_EXP_DIR/checkpoints/opsd_audit/" "$LOCAL_RUN_DIR/opsd_audit/"
  rsync -a "dfw:$REMOTE_EXP_DIR/checkpoints/opsd_token_kl/" "$LOCAL_RUN_DIR/opsd_token_kl/"
  if [[ "$SHAPE" == gap ]]; then
    rsync -a "dfw:$REMOTE_EXP_DIR/checkpoints/opsd_gap/" "$LOCAL_RUN_DIR/opsd_gap/"
  fi
  "$PYTHON" "$VERIFY_SCRIPT" \
    --audit-root "$LOCAL_RUN_DIR/opsd_audit" \
    --token-kl-root "$LOCAL_RUN_DIR/opsd_token_kl" \
    --actor-objective "$SV_ACTOR_OBJECTIVE" \
    --steering-scale "$SV_STEERING_SCALE" \
    --max-delta-fraction "$SV_MAX_DELTA_FRACTION" \
    --caa-scope "$SV_CAA_SCOPE" \
    --steering-source-mode "$SV_STEERING_SOURCE_MODE" \
    --report-dir "$report_dir"
  if [[ "$SHAPE" == gap ]]; then
    "$PYTHON" "$VERIFY_GAP_SCRIPT" \
      --step-dir "$LOCAL_RUN_DIR/opsd_gap/step_000001" \
      --output-dir "$report_dir/gap" \
      --expected-samples "$((N_PROMPTS * N_SAMPLES))" \
      --expected-scale "$SV_STEERING_SCALE"
  fi
}

preflight
case "$ACTION" in
  dry-run) submit 1 ;;
  submit) submit 0 ;;
  status) status ;;
  verify) verify ;;
  *) die "ACTION must be dry-run, submit, status, or verify" ;;
esac
