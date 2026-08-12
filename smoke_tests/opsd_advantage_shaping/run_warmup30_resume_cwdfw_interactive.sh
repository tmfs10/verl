#!/usr/bin/env bash
set -euo pipefail

# Exact advantage-shaping warmup-boundary gate. Segment 0 performs 30
# independent-teacher SFT-only updates and saves a composite actor+teacher
# checkpoint. Segment 1 resumes that checkpoint with the production dataloader
# reset policy and performs shaped joint updates 31-33. W&B is intentionally
# disabled. Checkpoint retention is two so an interrupted composite save cannot
# remove the checkpoint still named by the restart tracker.

ACTION="${1:-dry-run}"
SCRIPT_ROOT="${SCRIPT_ROOT:-/home/siddjain/workspace/scripts/src}"
VERL_ROOT="${VERL_ROOT:-/home/siddjain/workspace/verl/verl_main}"
PYTHON="${PYTHON:-/home/siddjain/anaconda3/envs/skills_latest/bin/python}"
LAUNCHER="${LAUNCHER:-$SCRIPT_ROOT/nemo_verl/skills_verl_submit.py}"
REWARD_FILE="${REWARD_FILE:-$SCRIPT_ROOT/nemo_verl/reward/verl_code_reward.py}"
CONFIG_FILE="$VERL_ROOT/recipe/opsd/config/opsd_trainer.yaml"
CONFIG_DIR="${CONFIG_DIR:-/home/siddjain/workspace/scripts/nemo_configs/cluster/codegen}"

SSH_TARGET="${SSH_TARGET:-dfw}"
RUN_TAG="${RUN_TAG:?set RUN_TAG to a stable smoke identifier}"
EXPNAME="opsd_advantage_shaping_warmup30_${RUN_TAG}"
OUTPUT_BASE="${OUTPUT_BASE:-/output/smoke_tests/opsd_advantage_shaping_warmup30_$RUN_TAG}"
REMOTE_OUTPUT="/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/${OUTPUT_BASE#/output/}/$EXPNAME"
LOCAL_RUN_DIR="${LOCAL_RUN_DIR:-/home/siddjain/data/smoke_tests/opsd_advantage_shaping_warmup30/$RUN_TAG}"
JOBS_TSV="$LOCAL_RUN_DIR/jobs.tsv"

TRAIN_DATA="/data/rl/opsd_openthoughts_compmath/openthoughts_math_30k_opsd_full.jsonl"
VAL_DATA="/data/rl/mathgen/comp_math_verl.jsonl"
REMOTE_TRAIN_DATA="/lustre/fsw/portfolios/llmservice/users/siddjain/llm/data/rl/opsd_openthoughts_compmath/openthoughts_math_30k_opsd_full.jsonl"
REMOTE_VAL_DATA="/lustre/fsw/portfolios/llmservice/users/siddjain/llm/data/rl/mathgen/comp_math_verl.jsonl"
MAX_PROMPT_LEN=4096
MAX_RESPONSE_LEN=1536
MAX_LEN=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))

die() {
  echo "[error] $*" >&2
  exit 2
}

validate() {
  [[ -f "$LAUNCHER" && -f "$REWARD_FILE" && -f "$CONFIG_FILE" ]] || die "missing required code"
  ssh "$SSH_TARGET" "test -s '$REMOTE_TRAIN_DATA' && test -s '$REMOTE_VAL_DATA'" \
    || die "missing OpenThoughts or CompMath data on CW-DFW"
  mkdir -p "$LOCAL_RUN_DIR"
}

extra_args() {
  local endpoint="$1"
  local expected="$2"
  local load_dataloader="$3"
  cat <<EOF
data.filter_overlong_prompts=True
data.filter_overlong_prompts_workers=16
data.dataloader_num_workers=0
++data.dynamic_masked_solution=False
++data.min_masked_fraction=null
++data.max_masked_fraction=null
++data.mask_seed=null
actor_rollout_ref.rollout.gpu_memory_utilization=0.4
actor_rollout_ref.rollout.max_model_len=$MAX_LEN
actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_LEN
actor_rollout_ref.rollout.top_p=1.0
actor_rollout_ref.rollout.top_k=-1
trainer.total_training_steps=$endpoint
trainer.total_epochs=1
trainer.expected_resume_step=$expected
trainer.load_dataloader_state_on_resume=$load_dataloader
trainer.logger=['console']
trainer.log_val_generations=0
trainer.save_freq=30
trainer.test_freq=-1
trainer.max_actor_ckpt_to_keep=2
algorithm.opsd.teacher_source=ground_truth
algorithm.opsd.distill_loss=sampled_reverse_kl
algorithm.opsd.topk=null
algorithm.opsd.distill_beta=null
algorithm.opsd.distill_token_clip=null
algorithm.opsd.distill_token_clip_tail=null
algorithm.opsd.balance_mode=none
algorithm.opsd.mode=opsd_rlvr
algorithm.opsd.teacher_model=separate
algorithm.opsd.rlvr_warmup_steps=30
algorithm.opsd.mix_weight=1.0
algorithm.opsd.distill_backward_scale=0.0
algorithm.opsd.rlvr_backward_scale=0.0
algorithm.opsd.advantage_shaping.enable=true
algorithm.opsd.advantage_shaping.score_source=teacher_minus_student_logprob
algorithm.opsd.advantage_shaping.scale=1.0
algorithm.opsd.advantage_shaping.normalize=std
algorithm.opsd.advantage_shaping.clip_z=3.0
algorithm.opsd.advantage_shaping.use_distill_mask=true
algorithm.opsd.advantage_shaping.allow_token_sign_flip=false
algorithm.opsd.advantage_shaping.max_response_tokens=null
algorithm.opsd.advantage_shaping.student_rlvr_backward_scale=1.0
algorithm.opsd.teacher_sft_weight=1.0
algorithm.opsd.teacher_sft_target_scope=thinking_and_answer
algorithm.opsd.teacher_sft_success_field=acc
algorithm.opsd.teacher_sft_success_threshold=0.5
algorithm.opsd.offpolicy_is_mode=token
algorithm.opsd.offpolicy_is_clip=2.0
algorithm.opsd.behavior_logprob_source=rollout
algorithm.opsd.max_prompt_length=$MAX_PROMPT_LEN
algorithm.opsd.truncation=error
algorithm.opsd.ground_truth_field=solution
algorithm.opsd.audit.enabled=true
algorithm.opsd.audit.output_dir=$OUTPUT_BASE/$EXPNAME/opsd_audit
algorithm.opsd.audit.global_steps=[30,31,32,33]
algorithm.opsd.audit.fail_fast=true
algorithm.opsd.audit.full_batch_ledger=true
algorithm.opsd.audit.reference_forward=true
algorithm.opsd.audit.reference_samples_per_rank=1
actor_rollout_ref.opsd_teacher.model.path=/hf_models/Qwen3-1.7B
actor_rollout_ref.opsd_teacher.optim.lr=2e-6
EOF
}

job_id_for_segment() {
  local segment="$1"
  [[ -f "$JOBS_TSV" ]] || return 0
  awk -F '\t' -v segment="$segment" '$1 == segment {print $2}' "$JOBS_TSV" | tail -1
}

verify_job_shape() {
  local job_id="$1"
  local dependency="$2"
  local record_path="$3"
  ssh "$SSH_TARGET" "scontrol update JobId='$job_id' Requeue=0; scontrol show job '$job_id' -o" > "$record_path"
  grep -q 'Requeue=0' "$record_path" || die "Requeue is not disabled for $job_id"
  grep -q 'Partition=interactive' "$record_path" || die "job $job_id is not interactive"
  grep -q 'OverSubscribe=NO' "$record_path" || die "job $job_id is not exclusive"
  grep -q 'ReqTRES=.*gres/gpu=8' "$record_path" || die "job $job_id does not use all eight GPUs"
  [[ -z "$dependency" ]] || grep -q "Dependency=afterany:$dependency" "$record_path" \
    || die "job $job_id is missing dependency afterany:$dependency"
}

submit_segment() {
  local segment="$1"
  local endpoint="$2"
  local expected="$3"
  local load_dataloader="$4"
  local dependency="$5"
  local dry_run="$6"
  local segment_dir="$LOCAL_RUN_DIR/segment_$segment"
  mkdir -p "$segment_dir"

  if [[ "$dry_run" == "0" ]]; then
    local existing
    existing="$(job_id_for_segment "$segment")"
    if [[ -n "$existing" ]]; then
      verify_job_shape "$existing" "$dependency" "$segment_dir/scontrol.txt"
      echo "[recovered] segment=$segment job=$existing"
      SUBMITTED_JOB_ID="$existing"
      return
    fi
  fi

  local flattened
  flattened="$(extra_args "$endpoint" "$expected" "$load_dataloader" | tr '\n' ' ')"
  local -a cmd=(
    "$PYTHON" "$LAUNCHER"
    --cluster cw-dfw
    --config_dir "$CONFIG_DIR"
    --output_base_dir "$OUTPUT_BASE"
    --local_verl_folder "$VERL_ROOT"
    --script_module recipe.opsd.main_opsd
    --verl_config_file "$CONFIG_FILE"
    --reward_file "$REWARD_FILE"
    --ground_truth_solution_key solution
    --expname "$EXPNAME"
    --partition interactive
    --time_limit 04:00:00
    --nodes 1
    --gpus 8
    --actor_model /hf_models/Qwen3-1.7B
    --prompt_data "$TRAIN_DATA"
    --eval_data "$VAL_DATA"
    --n_prompts 8
    --n_samples 8
    --n_val_samples 2
    --val_batch_size 8
    --max_prompt_len "$MAX_PROMPT_LEN"
    --max_len "$MAX_LEN"
    --max_tokens_per_gpu "$MAX_LEN"
    --num_epochs 1
    --num_training_jobs 1
    --training_job_start_index "$segment"
    --num_ppo_iter 1
    --actor_lr 2e-6
    --clip_ae 0.2,0.28
    --infer_server vllm
    --sequence_parallel_size 1
    --T 1.0
    --val_T 1.0
    --val_top_p 1.0
    --save_freq 30
    --test_freq -1
    --ae grpo
    --seed 1234
    --no_sandbox
    --omit_noncore_algorithm_overrides
    --dependency_type afterany
    --extra_args "$flattened"
  )
  if [[ "$segment" != "0" ]]; then
    cmd+=(--no_copy_verl_folder --disable_val_before_train)
  fi
  [[ -n "$dependency" ]] && cmd+=(--depends_on_slurm_job_id "$dependency")
  [[ "$dry_run" == "1" ]] && cmd+=(--dry_run)

  {
    printf '#!/usr/bin/env bash\n'
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } > "$segment_dir/command.sh"
  chmod +x "$segment_dir/command.sh"
  "${cmd[@]}" 2>&1 | tee "$segment_dir/submit.log"
  if [[ "$dry_run" == "1" ]]; then
    SUBMITTED_JOB_ID=dry-run
    return
  fi

  local job_id
  job_id="$(grep -Eo 'slurm_tunnel://nemo_run/[0-9]+' "$segment_dir/submit.log" | tail -1 | grep -Eo '[0-9]+' || true)"
  [[ -n "$job_id" ]] || job_id="$(grep -Eo 'Submitted batch job [0-9]+' "$segment_dir/submit.log" | tail -1 | awk '{print $4}' || true)"
  [[ -n "$job_id" ]] || die "could not parse segment $segment job id"
  verify_job_shape "$job_id" "$dependency" "$segment_dir/scontrol.txt"
  printf '%s\t%s\t%s\t%s\t%s\n' "$segment" "$job_id" "$expected" "$endpoint" "$load_dataloader" >> "$JOBS_TSV"
  SUBMITTED_JOB_ID="$job_id"
  echo "[submitted] segment=$segment job=$job_id steps=$expected..$endpoint"
}

submit_both() {
  local dry_run="$1"
  if [[ "$dry_run" == "0" && ! -e "$JOBS_TSV" ]]; then
    printf 'segment\tjob_id\texpected\tendpoint\tload_dataloader\n' > "$JOBS_TSV"
  fi
  submit_segment 0 30 0 true "" "$dry_run"
  local first="$SUBMITTED_JOB_ID"
  [[ "$dry_run" == "1" ]] && first=99999999
  submit_segment 1 33 30 false "$first" "$dry_run"
}

monitor() {
  [[ -f "$JOBS_TSV" ]] || die "missing jobs manifest"
  while true; do
    local active=0
    while IFS=$'\t' read -r segment job_id _; do
      [[ "$segment" == "segment" ]] && continue
      local state
      state="$(ssh -n "$SSH_TARGET" "squeue -h -j '$job_id' -o '%T' | head -1")"
      if [[ -n "$state" ]]; then
        active=$((active + 1))
        echo "[monitor] segment=$segment job=$job_id state=$state"
      else
        state="$(ssh -n "$SSH_TARGET" "sacct -X -j '$job_id' -n -P -o State | head -1 | cut -d'|' -f1")"
        echo "[monitor] segment=$segment job=$job_id final=${state:-unknown}"
      fi
    done < "$JOBS_TSV"
    [[ "$active" -eq 0 ]] && break
    sleep 60
  done
}

verify() {
  [[ -f "$JOBS_TSV" ]] || die "missing jobs manifest"
  while IFS=$'\t' read -r segment job_id _; do
    [[ "$segment" == "segment" ]] && continue
    local state
    state="$(ssh -n "$SSH_TARGET" "sacct -X -j '$job_id' -n -P -o State | head -1 | cut -d'|' -f1")"
    [[ "$state" == COMPLETED* ]] || die "segment $segment job $job_id ended in ${state:-unknown}"
  done < "$JOBS_TSV"

  local artifacts="$LOCAL_RUN_DIR/artifacts"
  mkdir -p "$artifacts/opsd_audit"
  rsync -a "$SSH_TARGET:$REMOTE_OUTPUT/training-logs/" "$artifacts/training-logs/"
  rsync -a "$SSH_TARGET:$REMOTE_OUTPUT/opsd_audit/" "$artifacts/opsd_audit/"
  ssh "$SSH_TARGET" "test \"\$(cat '$REMOTE_OUTPUT/checkpoints/latest_checkpointed_iteration.txt')\" = 33"
  ssh "$SSH_TARGET" "test -d '$REMOTE_OUTPUT/checkpoints/global_step_30/actor/opsd_teacher' && test -d '$REMOTE_OUTPUT/checkpoints/global_step_33/actor/opsd_teacher'" \
    || die "retention=2 did not preserve both complete composite checkpoints"
  grep -R "Resume checkpoint guard passed: expected=30 actual=30" "$artifacts/training-logs" >/dev/null \
    || die "resume-step guard confirmation is absent"
  grep -R "Resume dataloader state intentionally reset: global_step=30" "$artifacts/training-logs" >/dev/null \
    || die "dataloader-reset confirmation is absent"

  python3 "$VERL_ROOT/smoke_tests/opsd_advantage_shaping/verify_advantage_shaping_audit.py" \
    "$artifacts/opsd_audit" \
    --profile separate_sft_warmup \
    --expect-warmup-step 30 \
    --expect-joint-step 31 \
    --expect-joint-step 32 \
    --expect-joint-step 33 \
    --require-response-axis "$MAX_RESPONSE_LEN"

  python3 - "$artifacts/opsd_audit" "$artifacts/verification.json" <<'PY'
import json
import sys
from pathlib import Path

audit = Path(sys.argv[1])
output = Path(sys.argv[2])
step30 = json.loads((audit / "step_000030" / "trainer_batch.json").read_text())
step31 = json.loads((audit / "step_000031" / "trainer_batch.json").read_text())
rows30 = sorted({record["source_row_index"] for record in step30["records"]})
rows31 = sorted({record["source_row_index"] for record in step31["records"]})
if rows30 == rows31:
    raise SystemExit("dataloader reset gate did not change from step-30 rows to the reset first batch")
report = json.loads((audit / "verification_report.json").read_text())
payload = {
    "status": "PASS",
    "profile": report["profile"],
    "warmup_rows": rows30,
    "reset_rows": rows31,
    "actor_optimizer_steps": report["observations"]["actor_optimizer_steps"],
    "teacher_optimizer_steps": report["observations"]["teacher_optimizer_steps"],
    "retained_composite_checkpoints": [30, 33],
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
  echo "[result] PASS remote_output=$REMOTE_OUTPUT artifacts=$artifacts"
}

validate
case "$ACTION" in
  dry-run) submit_both 1 ;;
  submit) submit_both 0 ;;
  monitor) monitor ;;
  verify) verify ;;
  all) submit_both 0; monitor; verify ;;
  *) die "ACTION must be dry-run, submit, monitor, verify, or all" ;;
esac
