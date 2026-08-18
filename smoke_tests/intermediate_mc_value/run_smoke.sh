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
RECIPES=${RECIPES:-"scalar_random beta_variance"}
BACKENDS=${BACKENDS:-"fsdp fsdp2"}
RUN_RESUME=${RUN_RESUME:-1}
DRY_RUN=${DRY_RUN:-0}
NUM_CRITIQUES=${NUM_CRITIQUES:-2}
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

run_target() {
    local recipe=$1
    local backend=$2
    local run_dir=$3
    local total_steps=$4
    local audit_dir="$run_dir/audit"
    local checkpoint_dir="$run_dir/checkpoints"
    local log_file="$run_dir/train-to-${total_steps}.log"
    local -a config_only=()
    if [[ "$DRY_RUN" == "1" ]]; then
        config_only=(--cfg job)
    fi

    echo "[$recipe/$backend] training synchronously through update $total_steps"
    python3 -m verl.trainer.main_ppo "${config_only[@]}" \
        algorithm.adv_estimator=gae \
        algorithm.use_kl_in_reward=false \
        algorithm.intermediate_mc_value.enable=true \
        algorithm.intermediate_mc_value.recipe="$recipe" \
        algorithm.intermediate_mc_value.actor_loss_mode=dppo_tv \
        algorithm.intermediate_mc_value.num_critiques="$NUM_CRITIQUES" \
        algorithm.intermediate_mc_value.continuations_per_mark=2 \
        algorithm.intermediate_mc_value.max_marks=1 \
        algorithm.intermediate_mc_value.critic_warmup_updates=1 \
        algorithm.intermediate_mc_value.critique_max_response_length=64 \
        algorithm.intermediate_mc_value.mark_start_fraction=0.20 \
        algorithm.intermediate_mc_value.mark_end_fraction=0.80 \
        algorithm.intermediate_mc_value.min_mark_gap=1 \
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
        actor_rollout_ref.actor.use_kl_loss=false \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.top_k=-1 \
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
        critic.ppo_mini_batch_size=2 \
        critic.ppo_micro_batch_size_per_gpu=1 \
        critic.forward_micro_batch_size_per_gpu=1 \
        critic.ppo_epochs=1 \
        reward.reward_model.enable=false \
        trainer.use_legacy_worker_impl=enable \
        trainer.critic_warmup=0 \
        trainer.logger='["console"]' \
        trainer.project_name=intermediate_mc_value_smoke \
        trainer.experiment_name="${recipe}_${backend}" \
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

for recipe in $RECIPES; do
    for backend in $BACKENDS; do
        run_dir="$SMOKE_ROOT/${recipe}_${backend}"
        done_marker="$run_dir/verified-${TARGET_UPDATES}.done"
        mkdir -p "$run_dir"
        if [[ -f "$done_marker" ]]; then
            echo "[$recipe/$backend] already verified; skipping"
            continue
        fi

        run_target "$recipe" "$backend" "$run_dir" 2
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "[$recipe/$backend] configuration dry-run complete"
            continue
        fi
        if [[ "$RUN_RESUME" == "1" ]]; then
            run_target "$recipe" "$backend" "$run_dir" 3
        fi

        python3 smoke_tests/intermediate_mc_value/verify_audit.py \
            --audit-file "$run_dir/audit/intermediate_mc_value.jsonl" \
            --checkpoint-root "$run_dir/checkpoints" \
            --recipe "$recipe" \
            --num-critiques "$NUM_CRITIQUES" \
            --expected-critic-updates "$TARGET_UPDATES"
        touch "$done_marker"
        echo "[$recipe/$backend] verified"
    done
done

echo "Intermediate-MC smoke matrix complete: $SMOKE_ROOT (dry_run=$DRY_RUN)"
