#!/usr/bin/env bash
set -euo pipefail

required=(MODEL_PATH TRAIN_FILE VAL_FILE)
for name in "${required[@]}"; do
    if [[ -z "${!name:-}" ]]; then
        echo "Set $name before running the intermediate-MC smoke matrix." >&2
        exit 2
    fi
done

SMOKE_ROOT=${SMOKE_ROOT:-/home/siddjain/data/intermediate_mc_value_model/verl/smoke}
GPU_COUNT=${GPU_COUNT:-2}
CELLS=${CELLS:-"scalar_random scalar_ema beta_variance"}
BACKENDS=${BACKENDS:-"fsdp fsdp2"}
DYNAMIC_CELLS=${DYNAMIC_CELLS-"scalar_random"}
DYNAMIC_CRITIQUE_COUNT=${DYNAMIC_CRITIQUE_COUNT:-2}
RUN_DYNAMIC_PARITY=${RUN_DYNAMIC_PARITY:-1}
RUN_RESUME=${RUN_RESUME:-0}
DRY_RUN=${DRY_RUN:-0}
CRITIQUE_COUNTS=${CRITIQUE_COUNTS:-${NUM_CRITIQUES:-"2 0"}}
TARGET_UPDATES=2
if [[ "$RUN_RESUME" == "1" ]]; then
    TARGET_UPDATES=3
fi

export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
mkdir -p "$SMOKE_ROOT"

if [[ "$DRY_RUN" != "1" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required for this GPU smoke test." >&2
    exit 2
fi

cell_config() {
    case "$1" in
        scalar_random) printf '%s %s\n' scalar random ;;
        scalar_ema) printf '%s %s\n' scalar ema ;;
        beta_variance) printf '%s %s\n' beta variance ;;
        *) echo "Unknown smoke cell: $1" >&2; return 2 ;;
    esac
}

run_target() {
    local cell=$1
    local backend=$2
    local num_critiques=$3
    local dynamic_critic=$4
    local run_dir=$5
    local total_steps=$6
    local critic_head mark_selector
    read -r critic_head mark_selector < <(cell_config "$cell")
    local audit_dir="$run_dir/audit"
    local checkpoint_dir="$run_dir/checkpoints"
    local log_file="$run_dir/train-to-${total_steps}.log"
    local -a config_only=()
    local -a dynamic_overrides=()
    if [[ "$DRY_RUN" == "1" ]]; then
        config_only=(--cfg job)
    fi
    if [[ "$dynamic_critic" == "true" ]]; then
        dynamic_overrides=(
            critic.ppo_max_token_len_per_gpu=1024
            critic.forward_max_token_len_per_gpu=1024
        )
    fi

    echo "[$cell/$backend/critiques=$num_critiques/dynamic_critic=$dynamic_critic] native RayPPO training through global step $total_steps"
    python3 -m verl.trainer.main_ppo \
        --config-name=intermediate_mc_ppo_trainer \
        "${config_only[@]}" \
        algorithm.adv_estimator=gae \
        algorithm.gamma=1.0 \
        algorithm.use_kl_in_reward=false \
        algorithm.intermediate_mc_value.critic_head="$critic_head" \
        algorithm.intermediate_mc_value.mark_selector="$mark_selector" \
        algorithm.intermediate_mc_value.num_critiques="$num_critiques" \
        algorithm.intermediate_mc_value.continuations_per_mark=2 \
        algorithm.intermediate_mc_value.max_marks=1 \
        algorithm.intermediate_mc_value.critique_max_response_length=64 \
        algorithm.intermediate_mc_value.mark_start_fraction=0.20 \
        algorithm.intermediate_mc_value.mark_end_fraction=0.80 \
        algorithm.intermediate_mc_value.min_mark_gap=1 \
        algorithm.intermediate_mc_value.ema_baseline_token=1 \
        algorithm.intermediate_mc_value.ema_ratio_up=1.000001 \
        algorithm.intermediate_mc_value.ema_ratio_down=0.999999 \
        algorithm.intermediate_mc_value.variance_scope=rollout \
        algorithm.intermediate_mc_value.variance_random_probability=0.0 \
        algorithm.intermediate_mc_value.audit_output_dir="$audit_dir" \
        data.train_files="$TRAIN_FILE" \
        data.val_files="$VAL_FILE" \
        data.train_batch_size=2 \
        data.train_max_samples=2 \
        data.val_max_samples=2 \
        data.max_prompt_length=512 \
        data.max_response_length=64 \
        data.filter_overlong_prompts=true \
        data.truncation=error \
        actor_rollout_ref.model.path="$MODEL_PATH" \
        actor_rollout_ref.model.use_remove_padding=true \
        actor_rollout_ref.actor.strategy="$backend" \
        actor_rollout_ref.actor.optim.lr=1.0e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=2 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.ppo_epochs=1 \
        actor_rollout_ref.actor.use_dynamic_bsz=false \
        actor_rollout_ref.actor.use_kl_loss=false \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.max_model_len=1024 \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.top_k=-1 \
        actor_rollout_ref.rollout.logprobs_mode=processed_logprobs \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
        actor_rollout_ref.rollout.enforce_eager=true \
        critic.enable=true \
        critic.strategy="$backend" \
        critic.model.path="$MODEL_PATH" \
        critic.model.tokenizer_path="$MODEL_PATH" \
        critic.model.use_remove_padding=true \
        critic.optim.lr=1.0e-5 \
        critic.cliprange_value=0.2 \
        critic.ppo_mini_batch_size=2 \
        critic.ppo_micro_batch_size_per_gpu=1 \
        critic.forward_micro_batch_size_per_gpu=1 \
        critic.use_dynamic_bsz="$dynamic_critic" \
        "${dynamic_overrides[@]}" \
        critic.ppo_epochs=1 \
        reward.reward_model.enable=false \
        trainer.use_legacy_worker_impl=enable \
        trainer.critic_warmup=1 \
        trainer.logger='["console"]' \
        trainer.project_name=intermediate_mc_value_smoke \
        trainer.experiment_name="${cell}_${backend}_critiques${num_critiques}_dynamic${dynamic_critic}" \
        trainer.default_local_dir="$checkpoint_dir" \
        trainer.n_gpus_per_node="$GPU_COUNT" \
        trainer.nnodes=1 \
        trainer.val_before_train=false \
        trainer.save_freq=1 \
        trainer.test_freq=-1 \
        trainer.total_epochs="$total_steps" \
        trainer.total_training_steps="$total_steps" \
        trainer.resume_mode=auto 2>&1 | tee "$log_file"
}

run_case() {
    local cell=$1
    local backend=$2
    local num_critiques=$3
    local dynamic_critic=$4
    local critic_head mark_selector suffix run_dir done_marker
    read -r critic_head mark_selector < <(cell_config "$cell")
    if ! [[ "$num_critiques" =~ ^[0-9]+$ ]]; then
        echo "Invalid non-negative critique count: $num_critiques" >&2
        exit 2
    fi
    if [[ "$dynamic_critic" != "true" && "$dynamic_critic" != "false" ]]; then
        echo "Invalid dynamic critic flag: $dynamic_critic" >&2
        exit 2
    fi
    suffix=""
    if [[ "$dynamic_critic" == "true" ]]; then
        suffix="_dynamic_critic"
    fi
    run_dir="$SMOKE_ROOT/${cell}_${backend}_critiques${num_critiques}${suffix}"
    done_marker="$run_dir/verified-${TARGET_UPDATES}.done"
    mkdir -p "$run_dir"
    if [[ -f "$done_marker" ]]; then
        echo "[$cell/$backend/critiques=$num_critiques/dynamic_critic=$dynamic_critic] already verified; skipping"
        return
    fi

    run_target "$cell" "$backend" "$num_critiques" "$dynamic_critic" "$run_dir" 2
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[$cell/$backend/critiques=$num_critiques/dynamic_critic=$dynamic_critic] configuration dry-run complete"
        return
    fi
    if [[ "$RUN_RESUME" == "1" ]]; then
        run_target "$cell" "$backend" "$num_critiques" "$dynamic_critic" "$run_dir" 3
    fi

    python3 smoke_tests/intermediate_mc_value/verify_audit.py \
        --audit-file "$run_dir/audit/intermediate_mc_value.jsonl" \
        --checkpoint-root "$run_dir/checkpoints" \
        --critic-head "$critic_head" \
        --mark-selector "$mark_selector" \
        --num-critiques "$num_critiques" \
        --critic-warmup 1 \
        --expected-global-step "$TARGET_UPDATES"
    touch "$done_marker"
    echo "[$cell/$backend/critiques=$num_critiques/dynamic_critic=$dynamic_critic] verified"
}

run_dynamic_parity() {
    if [[ "$RUN_DYNAMIC_PARITY" != "1" ]]; then
        return
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "Dynamic critic parity requires GPUs and is skipped during Hydra configuration dry-runs"
        return
    fi
    if ! command -v torchrun >/dev/null 2>&1; then
        echo "torchrun is required for dynamic critic update parity." >&2
        exit 2
    fi
    for backend in $BACKENDS; do
        local parity_dir="$SMOKE_ROOT/dynamic_critic_update_${backend}"
        local result_file="$parity_dir/result.json"
        local done_marker="$parity_dir/verified.done"
        mkdir -p "$parity_dir"
        if [[ -f "$done_marker" ]]; then
            echo "[dynamic critic parity/$backend] already verified; skipping"
            continue
        fi
        torchrun --standalone --nproc-per-node=2 --module \
            smoke_tests.intermediate_mc_value.dynamic_critic_update_smoke \
            --strategy "$backend" \
            --output-json "$result_file" 2>&1 | tee "$parity_dir/run.log"
        touch "$done_marker"
    done
}

run_dynamic_parity

for cell in $CELLS; do
    for backend in $BACKENDS; do
        for num_critiques in $CRITIQUE_COUNTS; do
            run_case "$cell" "$backend" "$num_critiques" false
        done
    done
done

for cell in $DYNAMIC_CELLS; do
    for backend in $BACKENDS; do
        run_case "$cell" "$backend" "$DYNAMIC_CRITIQUE_COUNT" true
    done
done

echo "Intermediate-MC smoke matrix complete: $SMOKE_ROOT (dry_run=$DRY_RUN)"
