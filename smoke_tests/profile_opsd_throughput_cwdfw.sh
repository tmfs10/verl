#!/usr/bin/env bash
set -euo pipefail

# Production-shape, dry-run-first throughput smoke for the four directly
# measured OPSD/RLVR variants. Variant 5 is extrapolated from variant 3 by the
# paired summarizer because the requested 30-step teacher warmup is not run.

ACTION="${1:-dry-run}"
SCRIPT_ROOT="${SCRIPT_ROOT:-/home/siddjain/workspace/scripts/src}"
VERL_ROOT="${VERL_ROOT:-/home/siddjain/workspace/verl/verl_main}"
PYTHON="${PYTHON:-/home/siddjain/anaconda3/envs/skills_latest/bin/python}"
LAUNCHER="${LAUNCHER:-$SCRIPT_ROOT/nemo_verl/skills_verl_submit.py}"
DATA_PREP="${DATA_PREP:-$SCRIPT_ROOT/nemo_verl/prepare_openthoughts_math_opsd.py}"
REWARD_FILE="${REWARD_FILE:-$SCRIPT_ROOT/nemo_verl/reward/verl_code_reward.py}"
CONFIG_DIR="${CONFIG_DIR:-/home/siddjain/workspace/scripts/nemo_configs/cluster/codegen}"

CLUSTER="${CLUSTER:-cw-dfw}"
SSH_ALIAS="${SSH_ALIAS:-dfw}"
# Operational SSH/rsync target. This may point at another authenticated login
# node for the same pinned cluster when the canonical alias is unavailable.
SSH_TARGET="${SSH_TARGET:-$SSH_ALIAS}"
ACCOUNT="${ACCOUNT:-nemotron_reason_code}"
PARTITION="${PARTITION:-interactive}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
NODES="${NODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
SEED="${SEED:-1234}"
ACTOR_MODEL="${ACTOR_MODEL:-/hf_models/Qwen3-1.7B}"
TEACHER_MODEL="${TEACHER_MODEL:-/hf_models/Qwen3-1.7B}"
ACTOR_LR="${ACTOR_LR:-2e-6}"
TEACHER_LR="${TEACHER_LR:-2e-6}"
N_PROMPTS="${N_PROMPTS:-64}"
N_SAMPLES="${N_SAMPLES:-8}"
N_VAL_SAMPLES="${N_VAL_SAMPLES:-4}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-4096}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-8192}"
MAX_LEN="$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))"
TOTAL_STEPS="${TOTAL_STEPS:-5}"
TEMPERATURE="${TEMPERATURE:-1.0}"

DATASET="${DATASET:-siyanzhao/Openthoughts_math_30k_opsd}"
DATASET_REVISION="${DATASET_REVISION:-1f33e9dc2e8a1c639ca74f8024ad4a9f1f5eae62}"
EXPECTED_ROWS="${EXPECTED_ROWS:-29427}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/home/siddjain/data/opsd_openthoughts_math_30k_opsd}"
LOCAL_TRAIN_DATA="${LOCAL_TRAIN_DATA:-$LOCAL_DATA_ROOT/openthoughts_math_30k_opsd_full.jsonl}"
REMOTE_DATA_DIR="${REMOTE_DATA_DIR:-/lustre/fsw/portfolios/llmservice/users/siddjain/llm/data/rl/opsd_openthoughts_compmath}"
REMOTE_TRAIN_DATA="${REMOTE_TRAIN_DATA:-$REMOTE_DATA_DIR/openthoughts_math_30k_opsd_full.jsonl}"
TRAIN_DATA="${TRAIN_DATA:-/data/rl/opsd_openthoughts_compmath/openthoughts_math_30k_opsd_full.jsonl}"
REMOTE_VAL_DATA="${REMOTE_VAL_DATA:-/lustre/fsw/portfolios/llmservice/users/siddjain/llm/data/rl/mathgen/comp_math_verl.jsonl}"
VAL_DATA="${VAL_DATA:-/data/rl/mathgen/comp_math_verl.jsonl}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-/output/smoke_tests/opsd_throughput_$RUN_TAG}"
REMOTE_OUTPUT_BASE="${REMOTE_OUTPUT_BASE:-/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/${OUTPUT_BASE#/output/}}"
LOCAL_RUN_DIR="${LOCAL_RUN_DIR:-/home/siddjain/data/smoke_tests/opsd_throughput/$RUN_TAG}"
JOBS_TSV="$LOCAL_RUN_DIR/jobs.tsv"
RESUME_JOBS_TSV="$LOCAL_RUN_DIR/resume_jobs.tsv"
ACCOUNTING_TSV="$LOCAL_RUN_DIR/accounting.tsv"
RESUME_ACCOUNTING_TSV="$LOCAL_RUN_DIR/resume_accounting.tsv"
POLL_SECONDS="${POLL_SECONDS:-60}"

die() {
  echo "[error] $*" >&2
  exit 2
}

validate_invariants() {
  [[ "$CLUSTER" == "cw-dfw" ]] || die "this workflow is pinned to cw-dfw"
  [[ "$SSH_ALIAS" == "dfw" ]] || die "this workflow is pinned to the documented dfw alias"
  [[ "$ACCOUNT" == "nemotron_reason_code" ]] || die "unexpected account: $ACCOUNT"
  [[ "$PARTITION" == "interactive" ]] || die "smoke jobs must use interactive"
  [[ "$TIME_LIMIT" == "04:00:00" ]] || die "throughput smoke must use the full four-hour limit"
  [[ "$NODES" == "2" && "$GPUS_PER_NODE" == "8" ]] || die "expected 2 nodes x 8 GPUs"
  [[ "$N_PROMPTS" == "64" && "$N_SAMPLES" == "8" ]] || die "expected 64 prompts x 8 training rollouts"
  [[ "$N_VAL_SAMPLES" == "4" ]] || die "expected four validation rollouts per CompMath example"
  [[ "$MAX_PROMPT_LEN" == "4096" && "$MAX_RESPONSE_LEN" == "8192" ]] || die "expected 4K prompt + 8K response"
  [[ "$TOTAL_STEPS" == "5" ]] || die "profiling requires exactly five training steps"
  [[ "$TEMPERATURE" == "1.0" ]] || die "generation temperature must be 1.0"
  [[ -f "$LAUNCHER" && -f "$DATA_PREP" && -f "$REWARD_FILE" ]] || die "required local scripts are missing"
  mkdir -p "$LOCAL_RUN_DIR" "$LOCAL_DATA_ROOT"
}

prepare_data() {
  if [[ ! -f "$LOCAL_TRAIN_DATA" ]]; then
    echo "[data] converting complete pinned OpenThoughts split"
    "$PYTHON" "$DATA_PREP" \
      --dataset "$DATASET" \
      --revision "$DATASET_REVISION" \
      --output "$LOCAL_TRAIN_DATA" \
      --max-samples 0 \
      --scan-limit 0 \
      --selection first
  fi

  "$PYTHON" - "$LOCAL_TRAIN_DATA" "$EXPECTED_ROWS" "$DATASET_REVISION" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_rows = int(sys.argv[2])
expected_revision = sys.argv[3]
manifest_path = path.with_suffix(path.suffix + ".manifest.json")
if not manifest_path.is_file():
    raise SystemExit(f"missing manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text())
if manifest.get("revision") != expected_revision:
    raise SystemExit(f"revision mismatch: {manifest.get('revision')} != {expected_revision}")
if manifest.get("output_rows") != expected_rows:
    raise SystemExit(f"manifest row mismatch: {manifest.get('output_rows')} != {expected_rows}")

rows = 0
source_indices = set()
for line_number, line in enumerate(path.open(encoding="utf-8"), start=1):
    row = json.loads(line)
    rows += 1
    roles = [message["role"] for message in row["messages"]]
    if roles != ["system", "user"]:
        raise SystemExit(f"line {line_number}: actor roles are {roles}")
    if "COT_Reason" in row:
        raise SystemExit(f"line {line_number}: COT_Reason leaked into adapted row")
    if not str(row.get("solution", "")).strip():
        raise SystemExit(f"line {line_number}: empty teacher solution")
    if not str(row.get("Answer", "")).strip():
        raise SystemExit(f"line {line_number}: empty Answer")
    if json.loads(row["reward_model"]["ground_truth"]) != row["Answer"]:
        raise SystemExit(f"line {line_number}: verifier ground truth differs from Answer")
    extra = row["extra_info"]
    if extra["teacher_ground_truth_field"] != "solution":
        raise SystemExit(f"line {line_number}: teacher ground-truth field is not solution")
    if extra["source_answer"] != row["Answer"]:
        raise SystemExit(f"line {line_number}: source_answer provenance mismatch")
    if extra["source_cot_reason_present"] is not False:
        raise SystemExit(f"line {line_number}: source COT provenance is not false")
    source_index = int(row["source_row_index"])
    if source_index in source_indices:
        raise SystemExit(f"line {line_number}: duplicate source index {source_index}")
    source_indices.add(source_index)
if rows != expected_rows:
    raise SystemExit(f"data row mismatch: {rows} != {expected_rows}")
print(json.dumps({"rows": rows, "revision": expected_revision, "status": "validated"}, sort_keys=True))
PY

  echo "[data] staging full dataset on CW-DFW"
  ssh "$SSH_TARGET" "mkdir -p '$REMOTE_DATA_DIR'"
  rsync -a "$LOCAL_TRAIN_DATA" "$LOCAL_TRAIN_DATA.manifest.json" "$SSH_TARGET:$REMOTE_DATA_DIR/"
  local_sha="$(sha256sum "$LOCAL_TRAIN_DATA" | awk '{print $1}')"
  remote_sha="$(ssh "$SSH_TARGET" "sha256sum '$REMOTE_TRAIN_DATA' | awk '{print \$1}'")"
  [[ "$local_sha" == "$remote_sha" ]] || die "remote training-data checksum mismatch"
  remote_rows="$(ssh "$SSH_TARGET" "wc -l < '$REMOTE_TRAIN_DATA'")"
  [[ "$remote_rows" == "$EXPECTED_ROWS" ]] || die "remote training-data row mismatch: $remote_rows"
  ssh "$SSH_TARGET" "test -s '$REMOTE_VAL_DATA'"
  echo "[data] ready train=$REMOTE_TRAIN_DATA rows=$remote_rows sha256=$remote_sha eval=$REMOTE_VAL_DATA"
}

common_extra_args() {
  local total_steps="${1:-$TOTAL_STEPS}"
  local save_freq="${2:-5}"
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
trainer.total_training_steps=$total_steps
trainer.logger=['console']
trainer.log_val_generations=0
trainer.save_freq=$save_freq
trainer.test_freq=-1
EOF
}

opsd_common_extra_args() {
  cat <<EOF
algorithm.opsd.teacher_source=ground_truth
algorithm.opsd.distill_loss=sampled_reverse_kl
algorithm.opsd.topk=null
algorithm.opsd.distill_beta=null
algorithm.opsd.distill_token_clip=null
algorithm.opsd.distill_token_clip_tail=null
algorithm.opsd.balance_mode=none
algorithm.opsd.rlvr_warmup_steps=0
algorithm.opsd.teacher_sft_target_scope=thinking_and_answer
algorithm.opsd.teacher_sft_success_field=acc
algorithm.opsd.teacher_sft_success_threshold=0.5
algorithm.opsd.offpolicy_is_mode=token
algorithm.opsd.offpolicy_is_clip=2.0
algorithm.opsd.behavior_logprob_source=rollout
algorithm.opsd.audit.enabled=false
algorithm.opsd.max_prompt_length=$MAX_PROMPT_LEN
algorithm.opsd.truncation=error
algorithm.opsd.ground_truth_field=solution
EOF
}

variant_settings() {
  local variant="$1"
  case "$variant" in
    v1_vanilla_grpo)
      SCRIPT_MODULE="verl.trainer.main_ppo"
      VERL_CONFIG_FILE=""
      VARIANT_EXTRA=""
      ;;
    v2_shared_teacher)
      SCRIPT_MODULE="recipe.opsd.main_opsd"
      VERL_CONFIG_FILE="$VERL_ROOT/recipe/opsd/config/opsd_trainer.yaml"
      VARIANT_EXTRA="$(opsd_common_extra_args)
algorithm.opsd.mode=opsd
algorithm.opsd.teacher_model=actor
algorithm.opsd.distill_backward_scale=1.0
algorithm.opsd.rlvr_backward_scale=0.0
algorithm.opsd.teacher_sft_weight=0.0"
      ;;
    v3_separate_sft)
      SCRIPT_MODULE="recipe.opsd.main_opsd"
      VERL_CONFIG_FILE="$VERL_ROOT/recipe/opsd/config/opsd_trainer.yaml"
      VARIANT_EXTRA="$(opsd_common_extra_args)
algorithm.opsd.mode=opsd_rlvr
algorithm.opsd.teacher_model=separate
algorithm.opsd.mix_weight=0.5
algorithm.opsd.distill_backward_scale=2.0
algorithm.opsd.rlvr_backward_scale=0.0
algorithm.opsd.teacher_sft_weight=1.0
actor_rollout_ref.opsd_teacher.model.path=$TEACHER_MODEL
actor_rollout_ref.opsd_teacher.optim.lr=$TEACHER_LR"
      ;;
    v4_separate_rlvr)
      SCRIPT_MODULE="recipe.opsd.main_opsd"
      VERL_CONFIG_FILE="$VERL_ROOT/recipe/opsd/config/opsd_trainer.yaml"
      VARIANT_EXTRA="$(opsd_common_extra_args)
algorithm.opsd.mode=opsd_rlvr
algorithm.opsd.teacher_model=separate
algorithm.opsd.mix_weight=0.5
algorithm.opsd.distill_backward_scale=2.0
algorithm.opsd.rlvr_backward_scale=2.0
algorithm.opsd.teacher_sft_weight=0.0
actor_rollout_ref.opsd_teacher.model.path=$TEACHER_MODEL
actor_rollout_ref.opsd_teacher.optim.lr=$TEACHER_LR"
      ;;
    *) die "unknown variant: $variant" ;;
  esac
}

submit_variant() {
  local variant="$1"
  local dry_run="$2"
  local run_mode="${3:-profile}"
  local expname="opsd_throughput_${variant}_${RUN_TAG}"
  local local_variant_dir="$LOCAL_RUN_DIR/$variant"
  local remote_output="$REMOTE_OUTPUT_BASE/$expname"
  local manifest="$JOBS_TSV"
  local total_steps="$TOTAL_STEPS"
  local save_freq=5
  local -a resume_args=()
  if [[ "$run_mode" == "resume" ]]; then
    [[ "$variant" == "v3_separate_sft" || "$variant" == "v4_separate_rlvr" ]] \
      || die "resume verification is only defined for separate-teacher variants 3 and 4"
    local_variant_dir="$LOCAL_RUN_DIR/resume/$variant"
    manifest="$RESUME_JOBS_TSV"
    total_steps=6
    save_freq=-1
    resume_args=(--training_job_start_index 1 --disable_val_before_train)
    if [[ "$dry_run" == "0" ]]; then
      tracker="$remote_output/checkpoints/latest_checkpointed_iteration.txt"
      checkpoint="$remote_output/checkpoints/global_step_5"
      tracker_step="$(ssh "$SSH_TARGET" "cat '$tracker'")"
      [[ "$tracker_step" == "5" ]] || die "$variant resume expected checkpoint tracker 5, got $tracker_step"
      ssh "$SSH_TARGET" "test -d '$checkpoint/actor' && test -d '$checkpoint/actor/opsd_teacher'" \
        || die "$variant resume is missing actor or independent-teacher checkpoint state"
    fi
  elif [[ "$run_mode" != "profile" ]]; then
    die "unknown run mode: $run_mode"
  fi
  mkdir -p "$local_variant_dir"
  if [[ "$dry_run" == "0" && -f "$local_variant_dir/job_id.txt" ]]; then
    job_id="$(<"$local_variant_dir/job_id.txt")"
    ssh "$SSH_TARGET" "scontrol update JobId='$job_id' Requeue=0; scontrol show job '$job_id' -o" \
      > "$local_variant_dir/scontrol.txt"
    grep -q 'Requeue=0' "$local_variant_dir/scontrol.txt" || die "job $job_id did not register Requeue=0"
    grep -Eq 'NumNodes=(2|2-2)' "$local_variant_dir/scontrol.txt" || die "job $job_id does not request two nodes"
    grep -q 'OverSubscribe=NO' "$local_variant_dir/scontrol.txt" || die "job $job_id is not exclusive"
    grep -q 'ReqTRES=.*gres/gpu=16' "$local_variant_dir/scontrol.txt" || die "job $job_id does not request all 16 GPUs"
    if ! grep -q "^${variant}"$'\t' "$manifest"; then
      printf '%s\t%s\t%s\t%s\n' "$variant" "$job_id" "$expname" "$remote_output" >> "$manifest"
    fi
    echo "[recovered] variant=$variant job=$job_id output=$remote_output"
    return
  fi
  variant_settings "$variant"
  local extra_args
  extra_args="$(common_extra_args "$total_steps" "$save_freq")
$VARIANT_EXTRA"
  extra_args="$(printf '%s' "$extra_args" | tr '\n' ' ')"

  cmd=(
    "$PYTHON" "$LAUNCHER"
    --cluster "$CLUSTER"
    --config_dir "$CONFIG_DIR"
    --output_base_dir "$OUTPUT_BASE"
    --local_verl_folder "$VERL_ROOT"
    --script_module "$SCRIPT_MODULE"
    --reward_file "$REWARD_FILE"
    --ground_truth_solution_key solution
    --expname "$expname"
    --partition "$PARTITION"
    --time_limit "$TIME_LIMIT"
    --nodes "$NODES"
    --gpus "$GPUS_PER_NODE"
    --actor_model "$ACTOR_MODEL"
    --prompt_data "$TRAIN_DATA"
    --eval_data "$VAL_DATA"
    --n_prompts "$N_PROMPTS"
    --n_samples "$N_SAMPLES"
    --n_val_samples "$N_VAL_SAMPLES"
    --val_batch_size "$N_PROMPTS"
    --max_prompt_len "$MAX_PROMPT_LEN"
    --max_len "$MAX_LEN"
    --max_tokens_per_gpu "$MAX_LEN"
    --num_epochs 1
    --num_training_jobs 1
    --num_ppo_iter 1
    --actor_lr "$ACTOR_LR"
    --clip_ae 0.2,0.28
    --infer_server vllm
    --sequence_parallel_size 1
    --T "$TEMPERATURE"
    --val_T 1.0
    --val_top_p 1.0
    --save_freq "$save_freq"
    --test_freq -1
    --ae grpo
    --seed "$SEED"
    --no_sandbox
    --omit_noncore_algorithm_overrides
    --extra_args "$extra_args"
  )
  cmd+=("${resume_args[@]}")
  if [[ -n "$VERL_CONFIG_FILE" ]]; then
    cmd+=(--verl_config_file "$VERL_CONFIG_FILE")
  fi
  [[ "$dry_run" == "1" ]] && cmd+=(--dry_run)

  echo "[variant] name=$variant exp=$expname script=$SCRIPT_MODULE dry_run=$dry_run"
  printf '[command] '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}" 2>&1 | tee "$local_variant_dir/submit.log"
  if [[ "$dry_run" == "1" ]]; then
    return
  fi

  job_id="$(grep -Eo 'slurm_tunnel://nemo_run/[0-9]+' "$local_variant_dir/submit.log" | tail -1 | grep -Eo '[0-9]+' || true)"
  if [[ -z "$job_id" ]]; then
    job_id="$(grep -Eo 'Submitted batch job [0-9]+' "$local_variant_dir/submit.log" | tail -1 | awk '{print $4}' || true)"
  fi
  [[ -n "$job_id" ]] || die "could not parse job id for $variant"
  echo "$job_id" > "$local_variant_dir/job_id.txt"
  ssh "$SSH_TARGET" "scontrol update JobId='$job_id' Requeue=0; scontrol show job '$job_id' -o" \
    > "$local_variant_dir/scontrol.txt"
  grep -q 'Requeue=0' "$local_variant_dir/scontrol.txt" || die "job $job_id did not register Requeue=0"
  grep -Eq 'NumNodes=(2|2-2)' "$local_variant_dir/scontrol.txt" || die "job $job_id does not request two nodes"
  grep -q 'OverSubscribe=NO' "$local_variant_dir/scontrol.txt" || die "job $job_id is not exclusive"
  grep -q 'ReqTRES=.*gres/gpu=16' "$local_variant_dir/scontrol.txt" || die "job $job_id does not request all 16 GPUs"
  printf '%s\t%s\t%s\t%s\n' "$variant" "$job_id" "$expname" "$remote_output" >> "$manifest"
  echo "[submitted] variant=$variant job=$job_id output=$remote_output"
}

submit_resume_all() {
  local dry_run="${1:-0}"
  if [[ "$dry_run" == "0" ]]; then
    if [[ ! -e "$RESUME_JOBS_TSV" ]]; then
      printf 'variant\tjob_id\texpname\tremote_output\n' > "$RESUME_JOBS_TSV"
    else
      [[ "$(head -n 1 "$RESUME_JOBS_TSV")" == $'variant\tjob_id\texpname\tremote_output' ]] \
        || die "unexpected resume manifest header: $RESUME_JOBS_TSV"
    fi
  fi
  for variant in v3_separate_sft v4_separate_rlvr; do
    submit_variant "$variant" "$dry_run" resume
  done
}

submit_all() {
  local dry_run="$1"
  if [[ "$dry_run" == "0" ]]; then
    if [[ ! -e "$JOBS_TSV" ]]; then
      printf 'variant\tjob_id\texpname\tremote_output\n' > "$JOBS_TSV"
    else
      [[ "$(head -n 1 "$JOBS_TSV")" == $'variant\tjob_id\texpname\tremote_output' ]] \
        || die "unexpected job manifest header: $JOBS_TSV"
    fi
  fi
  for variant in v1_vanilla_grpo v2_shared_teacher v3_separate_sft v4_separate_rlvr; do
    submit_variant "$variant" "$dry_run"
  done
}

monitor_jobs() {
  local jobs_tsv="${1:-$JOBS_TSV}"
  local accounting_tsv="${2:-$ACCOUNTING_TSV}"
  local log_root="${3:-$LOCAL_RUN_DIR/logs}"
  [[ -f "$jobs_tsv" ]] || die "missing job manifest: $jobs_tsv"
  while true; do
    active=0
    while IFS=$'\t' read -r variant job_id expname remote_output; do
      [[ "$variant" == "variant" ]] && continue
      state="$(ssh -n "$SSH_TARGET" "squeue -h -j '$job_id' -o '%T' | head -1")"
      if [[ -n "$state" ]]; then
        active=$((active + 1))
        echo "[monitor] variant=$variant job=$job_id state=$state"
      else
        final_state="$(ssh -n "$SSH_TARGET" "sacct -X -j '$job_id' -n -P -o State | head -1 | cut -d'|' -f1")"
        echo "[monitor] variant=$variant job=$job_id final_state=${final_state:-not-visible}"
      fi
    done < "$jobs_tsv"
    [[ "$active" -eq 0 ]] && break
    sleep "$POLL_SECONDS"
  done

  printf 'job_id\tstate\telapsed_raw\tstart\tend\talloc_tres\tnodelist\n' > "$accounting_tsv"
  while IFS=$'\t' read -r variant job_id expname remote_output; do
    [[ "$variant" == "variant" ]] && continue
    row="$(ssh -n "$SSH_TARGET" "sacct -X -j '$job_id' -n -P -o JobIDRaw,State,ElapsedRaw,Start,End,AllocTRES,NodeList | head -1")"
    IFS='|' read -r acct_job state elapsed start end alloc_tres nodelist <<< "$row"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$acct_job" "$state" "$elapsed" "$start" "$end" "$alloc_tres" "$nodelist" >> "$accounting_tsv"
    mkdir -p "$log_root/$variant"
    rsync -a "$SSH_TARGET:$remote_output/training-logs/" "$log_root/$variant/" < /dev/null
  done < "$jobs_tsv"
}

analyze_jobs() {
  "$PYTHON" "$VERL_ROOT/smoke_tests/summarize_opsd_throughput.py" \
    --jobs "$JOBS_TSV" \
    --accounting "$ACCOUNTING_TSV" \
    --log-root "$LOCAL_RUN_DIR/logs" \
    --output-dir "$LOCAL_RUN_DIR/report"
}

validate_invariants
case "$ACTION" in
  prepare) prepare_data ;;
  dry-run) submit_all 1 ;;
  submit-all) submit_all 0 ;;
  dry-run-resume) submit_resume_all 1 ;;
  submit-resume) submit_resume_all 0 ;;
  monitor) monitor_jobs ;;
  monitor-resume) monitor_jobs "$RESUME_JOBS_TSV" "$RESUME_ACCOUNTING_TSV" "$LOCAL_RUN_DIR/resume_logs" ;;
  analyze) analyze_jobs ;;
  all)
    prepare_data
    submit_all 0
    monitor_jobs
    analyze_jobs
    ;;
  *) die "ACTION must be prepare, dry-run, submit-all, dry-run-resume, submit-resume, monitor, monitor-resume, analyze, or all" ;;
esac
