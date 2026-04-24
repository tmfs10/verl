## Math Self-Correction Smoke Suite

### Purpose

Smoke test the five math self-correction interaction variants that were added around `MathVerifyInteraction` and the `tool_agent` multi-turn loop.

### Interaction configs

- `/home/siddjain/workspace/verl/verl_main/examples/self_correction_smoke/interaction_configs/verifier_full_history.yaml`
- `/home/siddjain/workspace/verl/verl_main/examples/self_correction_smoke/interaction_configs/verifier_question_with_past_answers.yaml`
- `/home/siddjain/workspace/verl/verl_main/examples/self_correction_smoke/interaction_configs/repeat_until_stable_full_history.yaml`
- `/home/siddjain/workspace/verl/verl_main/examples/self_correction_smoke/interaction_configs/repeat_until_stable_question_with_past_answers.yaml`
- `/home/siddjain/workspace/verl/verl_main/examples/self_correction_smoke/interaction_configs/s2r_full_history.yaml`

### Dataset expectations

- Samples should use `agent_name: "tool_agent"`.
- `data.return_raw_chat=True` must be enabled.
- Each sample should carry `extra_info.interaction_kwargs` with at least:
  - `name: "selfcorr"`
  - `ground_truth`
- The historical smoke workflow used:
  - `/data/rl/mathgen/selfcorr_smoke8.jsonl`

### DFW smoke submission

- Use the submit wrapper, not a direct local trainer invocation.
- Keep W&B disabled for the smoke.
- Keep the multi-turn knobs explicit instead of relying on defaults.

```bash
cd /home/siddjain/workspace/verl/verl_main

COMMON=(
  --cluster cw-dfw
  --config_dir /home/siddjain/workspace/scripts/nemo_configs/cluster/codegen
  --partition interactive
  --output_base_dir /output/rl/selfcorr
  --local_verl_folder /home/siddjain/workspace/verl/verl_main
  --actor_model /my_models/Qwen3-1.7B-Base
  --prompt_data /data/rl/mathgen/selfcorr_smoke8.jsonl
  --eval_data /data/rl/mathgen/selfcorr_smoke8.jsonl
  --nodes 1
  --n_prompts 8
  --n_samples 1
  --n_val_samples 1
  --max_prompt_len 2k
  --max_len 4k
  --num_epochs 1
  --num_ppo_iter 1
  --actor_lr 2e-6
  --save_freq 1000
  --test_freq 1000
  --sequence_parallel_size 1
  --ref_sequence_parallel_size 1
  --script_module verl.trainer.main_ppo
)

submit_one () {
  local expname="$1"
  local cfg="$2"

  conda run -n skills_latest python /home/siddjain/workspace/scripts/src/nemo_verl/skills_verl_submit_addons.py \
    "${COMMON[@]}" \
    --expname "$expname" \
    --extra_args "trainer.val_before_train=False actor_rollout_ref.rollout.multi_turn.enable=True actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent actor_rollout_ref.rollout.multi_turn.interaction_config_path=/opt/verl/examples/self_correction_smoke/interaction_configs/$cfg actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 actor_rollout_ref.rollout.multi_turn.max_user_turns=4"
}

submit_one selfcorr_verifier_full_history_smoke verifier_full_history.yaml
submit_one selfcorr_verifier_qwpa_smoke verifier_question_with_past_answers.yaml
submit_one selfcorr_repeat_full_history_smoke repeat_until_stable_full_history.yaml
submit_one selfcorr_repeat_qwpa_smoke repeat_until_stable_question_with_past_answers.yaml
submit_one selfcorr_s2r_full_history_smoke s2r_full_history.yaml
```

### Expected behavior

- `verifier`:
  - retries only after incorrect verified answers
- `repeat_until_stable`:
  - stops once the extracted completed answer repeats
- `question_with_past_answers`:
  - later turns rebuild the next generation prompt from the original question plus prior extracted answers
- `s2r`:
  - runs answer -> self-verification -> retry / terminate
  - must stay on `turn_context_mode=full_history`

## Aligned Reward-Focus-Tail Conditional Runs

### Purpose

Run the masked-solution conditional-reward workflow where the prompt mask and reward focus are tied to the same low-confidence tail token set.

### Behavior

- `data.masked_solution_selection_mode=reward_focus_tail`
- trainer computes low-confidence tail positions from the live actor
- those exact positions are masked in the prompt with `<|fim_middle|>`
- those same positions are reused by the reward manager

### Smoke guidance

- Smoke tests should disable W&B.
- Keep the smoke as close to production as feasible:
  - same reward mode
  - same `reward_focus_tail` masking mode
  - same `low_confidence_tail_percent`
  - same `truncate_conditioning_response_at_last_think` setting
  - same explicit prompt/response token budgets where possible
- The passing smoke path explicitly needed:
  - `reward.reward_kwargs.low_confidence_min_tokens=1`
  - `data.max_prompt_length=2048`
  - `data.max_response_length=8192`
  - `actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240`
  - `actor_rollout_ref.rollout.max_num_batched_tokens=10240`
  - `critic.ppo_max_token_len_per_gpu=10240`

### Recovery-ratio variant

Use:

- `reward.reward_kwargs.conditioning_reward_mode=low_confidence_recovery_ratio`
- `reward.reward_kwargs.low_confidence_tail_percent=20`
- `reward.reward_kwargs.low_confidence_min_tokens=1`
- `reward.reward_kwargs.use_rlvr_reward_when_group_has_success=False`
- `reward.reward_kwargs.truncate_conditioning_response_at_last_think=False`
- `data.masked_solution_selection_mode=reward_focus_tail`

### Top-k recall variant

Use:

- `reward.reward_kwargs.conditioning_reward_mode=low_confidence_token_topk_recall`
- `reward.reward_kwargs.low_confidence_tail_percent=20`
- `reward.reward_kwargs.low_confidence_min_tokens=1`
- `reward.reward_kwargs.conditioned_token_topk=2`
- `reward.reward_kwargs.use_rlvr_reward_when_group_has_success=False`
- `reward.reward_kwargs.truncate_conditioning_response_at_last_think=False`
- `data.masked_solution_selection_mode=reward_focus_tail`

## Offline Reward Replay On Saved Rollouts

### Purpose

Isolate reward-manager behavior on identical saved responses without touching the live training path.

### When to use

- Investigate train-side metric gaps between `batch` / GRPO and `conditional_logprob` runs.
- Verify whether a gap is coming from:
  - scorer output on identical responses
  - conditional reward replacement on identical responses
  - or generation differences upstream of reward computation

### Procedure

1. Cancel the currently running RL jobs on DFW if they are no longer needed for the investigation.
2. Pick one saved rollout JSONL from each experiment to compare, for example:
   - GRPO rollout: `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/grpo_qwen3_30b_a3b_deepmath_compmath_prompt4k_train8k_val20k_full_v1_tp4/generations/rollout/59.jsonl`
   - RR20 rollout: `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/condlogprob_qwen3_30b_a3b_deepmath_compmath_prompt4k_train8k_val20k_recoveryratio20_allfail_full_v1_tp4/generations/rollout/59.jsonl`
3. Run an offline replay script from `~/data`, not from the repo workspace.
4. Reconstruct a minimal `DataProto` from the saved rows and run:
   - `BatchRewardManager`
   - `ConditionalLogProbRewardManager`
   on the exact same saved responses.
5. Compare:
   - raw `acc`
   - `reward/standard_acc`
   - `rule_reward`
   - final `score`
   - conditional-only fields such as `used_conditional_logprob` and `low_confidence_recovery_ratio`
6. Treat any gap that survives replay on identical responses as a scoring / reward-manager issue. Treat any gap that disappears under replay as upstream generation / batching behavior.

### Local replay script

- Script:
  - `/home/siddjain/data/codex_tmp/stage1_reward_replay/replay_saved_rollout_scores.py`
- Example artifacts used:
  - `/home/siddjain/data/codex_tmp/stage1_reward_replay/grpo_59.jsonl`
  - `/home/siddjain/data/codex_tmp/stage1_reward_replay/rr20_59.jsonl`
- Example summary outputs:
  - `/home/siddjain/data/codex_tmp/stage1_reward_replay/summary_step59.json`
  - `/home/siddjain/data/codex_tmp/stage1_reward_replay/summary_step59_skills_env.json`

Run it with the `skills_latest` env and `PYTHONPATH` pointing at both `skills_latest` and `verl_main`, otherwise the math grader falls back to all-incorrect:

```bash
PYTHONPATH=/home/siddjain/workspace/skills_latest:/home/siddjain/workspace/verl/verl_main \
conda run -n skills_latest python \
  /home/siddjain/data/codex_tmp/stage1_reward_replay/replay_saved_rollout_scores.py \
  --reward-module /home/siddjain/workspace/scripts/src/nemo_verl/reward/verl_code_reward.py \
  --grpo /home/siddjain/data/codex_tmp/stage1_reward_replay/grpo_59.jsonl \
  --rr20 /home/siddjain/data/codex_tmp/stage1_reward_replay/rr20_59.jsonl \
  --out /home/siddjain/data/codex_tmp/stage1_reward_replay/summary_step59_skills_env.json
```

### Notes

- This workflow is intentionally offline-only.
- It should not modify trainer code, rollout code, or cluster job configuration.
- Temporary scripts for this workflow belong under `~/data/codex_tmp`.

## Validation Dashboard Filtering

### Purpose

Reduce validation logging noise so dashboards only show the selection-style aggregates that are useful for model comparison.

### Behavior

- Validation now exports only:
  - `best@N/*`
  - `maj@N/*`
  - `worst@N/*`
- Validation no longer exports:
  - plain `mean@N`
  - plain `std@N`
  - `val-aux/num_turns/*`

### Scope

- This is a logger/export filter in PPO validation.
- It does not change generation, reward computation, or the internal bootstrap aggregation logic.
- Existing W&B runs keep their historical panels; the filter only affects new logs emitted after this code change.

## SLURM Timeout Checkpointing

### Purpose

Make PPO/GRPO/conditional-logprob jobs save a checkpoint shortly before SLURM walltime expiry, then exit cleanly so dependent continuation jobs can resume from a real checkpoint instead of timing out mid-step.

### Behavior

- On SLURM, the trainer auto-detects `SLURM_JOB_END_TIME` and uses it as the checkpoint deadline when no explicit override is set.
- Outside SLURM, or for manual testing, set:
  - `trainer.checkpoint_must_save_by=DD:HH:MM:SS`
- If `trainer.checkpoint_must_save_by` is set, it overrides `SLURM_JOB_END_TIME`. This is the correct way to smoke test the timeout path on SLURM.
- The timeout save trigger uses:
  - longest observed historical step duration
  - `trainer.checkpoint_save_duration`
  - `trainer.esi_redundant_time`
- After a timeout-triggered save, training stops early instead of entering validation or another rollout step.

### Relevant knobs

- `trainer.save_freq`
- `trainer.checkpoint_must_save_by`
- `trainer.checkpoint_save_duration`
- `trainer.esi_redundant_time`

### Example

To keep periodic checkpoints every 10 steps and also force a last save before a 4-hour deadline:

```bash
--save_freq 10 \
--extra_args "trainer.checkpoint_save_duration=120 trainer.esi_redundant_time=120"
```

On SLURM no explicit `trainer.checkpoint_must_save_by` is required for production runs, because the trainer will use `SLURM_JOB_END_TIME` automatically. For non-SLURM local/manual testing:

```bash
--extra_args "trainer.checkpoint_must_save_by=00:03:55:00 trainer.checkpoint_save_duration=120 trainer.esi_redundant_time=120"
```

### Smoke test on DFW interactive

To verify the timeout-save path itself, use a 1-node GRPO smoke with:

- W&B disabled
- `save_freq=-1` so only the timeout hook can create a checkpoint
- a large `trainer.total_training_steps` so the run does not finish naturally
- `trainer.checkpoint_save_duration=120`
- `trainer.esi_redundant_time=120`
- `trainer.checkpoint_must_save_by=00:00:12:00`

Submit with the normal VErl wrapper, keep W&B off, and use the explicit `trainer.checkpoint_must_save_by` deadline for the smoke trigger itself. A shortened SLURM walltime can still be used as a hard cap, but post-launch `scontrol update JobId=... TimeLimit=...` is not sufficient to validate the trainer-side deadline logic by itself.

Launchers:

- `/home/siddjain/data/codex_tmp/run_timeout_ckpt_smoke_qwen3_1p7b_grpo.sh`
- `/home/siddjain/data/codex_tmp/run_timeout_ckpt_smoke_qwen3_1p7b_grpo_v2.sh`
- `/home/siddjain/data/codex_tmp/run_timeout_ckpt_smoke_qwen3_1p7b_grpo_v3.sh`

Expected behavior:

1. Job starts on DFW interactive.
2. Training reaches at least one actor-update step so `max_steps_duration` is populated.
3. Before SLURM timeout, log prints:
   - `Force saving checkpoint: job timeout approaching.`
   - `Timeout-triggered checkpoint saved, stopping training early.`
4. `checkpoints/latest_checkpointed_iteration.txt` exists even though `save_freq=-1`.

## OPSD Qwen3-1.7B smoke tests

### Purpose
Run the two OPSD smokes on DFW through the cluster submit wrapper, not locally:

1. Distillation-only OPSD with `algorithm.opsd.mode=opsd` and `algorithm.opsd.teacher_model=fixed`
2. `opsd_rlvr` with `algorithm.opsd.mode=opsd_rlvr` and `algorithm.opsd.teacher_model=actor`

Both smokes use the same requested distillation settings:

- `algorithm.opsd.teacher_source=ground_truth`
- `algorithm.opsd.distill_loss=topk_jsd`
- `algorithm.opsd.topk=64`
- `algorithm.opsd.distill_beta=0.5`
- `algorithm.opsd.distill_token_clip=0.05`
- `algorithm.opsd.distill_token_clip_tail=false`

### Guardrails

- Use `cw-dfw`
- Smoke tests go on `interactive`
- Request `8` GPUs per node
- Omit `--enable_wandb` for smoke tests
- Use the cluster submit wrapper, not direct local `conda run`
- For `recipe.opsd.main_opsd` via `skills_verl_submit.py`, the wrapper sets `data.prompt_key=messages`, so the dataset must use a `messages` field, not `prompt`
- Do not rely on wrapper defaults or `opsdv2_trainer.yaml` defaults for the smoke knobs below; pass them explicitly
- Use `--max_prompt_len 2k` and `--max_len 6k`
- Keep `--max_len 6k` for the current OPSD smoke workflow unless intentionally changed

### Explicit OPSD smoke baseline

Use these as the baseline smoke settings before layering on the OPSD-specific overrides:

- `--n_prompts 8`
- `--n_samples 2`
- `--n_val_samples 1`
- `--T 1.0`
- `--num_epochs 1`
- `--num_ppo_iter 1`
- `--max_prompt_len 2k`
- `--max_len 6k`
- `--actor_lr 1e-6`
- `--ae grpo`
- `--clip_ae 0.2,0.28`
- `--infer_server vllm`
- `--sequence_parallel_size 1`
- `--max_tokens_per_gpu 6144`
- `--disable_val_before_train`
- `trainer.total_training_steps=1`
- `data.filter_overlong_prompts=False`

### Resolved smoke behavior

With the explicit smoke baseline above, the wrapper resolves the following effective training settings:

- `data.max_response_length = 4k`
- `actor_rollout_ref.actor.ppo_mini_batch_size = 8`
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 1`
- `actor_rollout_ref.actor.use_dynamic_bsz = False`
- `actor_rollout_ref.actor.ulysses_sequence_parallel_size = 1`
- `actor_rollout_ref.actor.ppo_max_token_len_per_gpu = 6144`
- `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu = 1`
- `actor_rollout_ref.rollout.tensor_model_parallel_size = 1`
- `actor_rollout_ref.rollout.max_num_batched_tokens = 6144`

These are part of the effective smoke config and should be treated as such when comparing the smoke against future runs.

Use `recipe/opsd/config/opsdv2_trainer.yaml` only as the recipe entrypoint/base config. For these smokes, the meaningful training/distillation knobs should be set explicitly in the CLI and `--extra_args`, not implicitly inherited from that YAML.

### Smoke 1: OPSD fixed teacher on DFW interactive

Validated on 2026-04-09:

- job: `11007595`
- expname: `opsd_qwen3_1p7b_fixed_teacher_smoke_v3_deepmath6k`
- result: `COMPLETED`
- train: `/data/rl/mathgen/deepmath_verl.jsonl`
- eval: `/data/rl/mathgen/comp_math_verl.jsonl`
- key logged metrics at `step:1`:
  - `actor/opsd_loss=0.04496783041395247`
  - `actor/opsd_jsd_loss=0.04496783041395247`
  - `actor/opsd_topk_divergence_beta=0.5`
  - `actor/teacher_rlvr_loss=0.0`
  - `reward/pass@k=0.1875`

```bash
cd /home/siddjain/workspace/verl/verl_main
conda run -n skills_latest python /home/siddjain/workspace/scripts/src/nemo_verl/skills_verl_submit.py \
  --cluster cw-dfw \
  --config_dir /home/siddjain/workspace/scripts/nemo_configs/cluster/codegen \
  --output_base_dir /output/rl \
  --local_verl_folder /home/siddjain/workspace/verl/verl_main \
  --reward_file /home/siddjain/workspace/scripts/src/nemo_verl/reward/verl_code_reward.py \
  --verl_config_file /home/siddjain/workspace/verl/verl_main/recipe/opsd/config/opsdv2_trainer.yaml \
  --script_module recipe.opsd.main_opsd \
  --actor_model /hf_models/Qwen3-1.7B \
  --prompt_data /data/rl/mathgen/deepmath_verl.jsonl \
  --eval_data /data/rl/mathgen/comp_math_verl.jsonl \
  --nodes 1 \
  --gpus 8 \
  --add_interactive \
  --ae grpo \
  --num_epochs 1 \
  --n_prompts 8 \
  --n_samples 2 \
  --n_val_samples 1 \
  --actor_lr 1e-6 \
  --clip_ae 0.2,0.28 \
  --infer_server vllm \
  --sequence_parallel_size 1 \
  --T 1.0 \
  --max_prompt_len 2k \
  --max_len 6k \
  --max_tokens_per_gpu 6144 \
  --num_ppo_iter 1 \
  --disable_val_before_train \
  --save_freq -1 \
  --test_freq 1 \
  --expname opsd_qwen3_1p7b_fixed_teacher_smoke_v3_deepmath6k \
  --extra_args "data.filter_overlong_prompts=False trainer.project_name=opsd-smoke trainer.total_training_steps=1 ++data.dynamic_masked_solution=False ++data.min_masked_fraction=null ++data.max_masked_fraction=null ++data.mask_seed=null algorithm.opsd.teacher_source=ground_truth algorithm.opsd.distill_loss=topk_jsd algorithm.opsd.topk=64 algorithm.opsd.distill_beta=0.5 algorithm.opsd.distill_token_clip=0.05 algorithm.opsd.distill_token_clip_tail=False algorithm.opsd.mode=opsd algorithm.opsd.teacher_model=fixed"
```

### Smoke 1b: OPSD fixed teacher with shorter train response and longer validation response

Validation-response-length smoke for the new `actor_rollout_ref.rollout.val_kwargs.response_length` override:

- train data: `/data/rl/mathgen/deepmath_verl.jsonl`
- eval data: `/data/rl/mathgen/comp_math_verl.jsonl`
- train prompt/response: `2k / 2k`
- validation prompt/response: `2k / 4k`
- keep validation enabled before training
- because the wrapper still derives training response length from `max_len - max_prompt_len`, use:
  - `--max_prompt_len 2k`
  - `--max_len 4k`
  - `--max_tokens_per_gpu 6144`
  - `actor_rollout_ref.rollout.max_model_len=6144`
  - `actor_rollout_ref.rollout.max_num_batched_tokens=6144`
  - `actor_rollout_ref.rollout.val_kwargs.response_length=4096`

```bash
cd /home/siddjain/workspace/verl/verl_main
conda run -n skills_latest python /home/siddjain/workspace/scripts/src/nemo_verl/skills_verl_submit.py \
  --cluster cw-dfw \
  --config_dir /home/siddjain/workspace/scripts/nemo_configs/cluster/codegen \
  --output_base_dir /output/rl \
  --local_verl_folder /home/siddjain/workspace/verl/verl_main \
  --reward_file /home/siddjain/workspace/scripts/src/nemo_verl/reward/verl_code_reward.py \
  --verl_config_file /home/siddjain/workspace/verl/verl_main/recipe/opsd/config/opsdv2_trainer.yaml \
  --script_module recipe.opsd.main_opsd \
  --actor_model /hf_models/Qwen3-1.7B \
  --prompt_data /data/rl/mathgen/deepmath_verl.jsonl \
  --eval_data /data/rl/mathgen/comp_math_verl.jsonl \
  --nodes 1 \
  --gpus 8 \
  --add_interactive \
  --ae grpo \
  --num_epochs 1 \
  --n_prompts 8 \
  --n_samples 2 \
  --n_val_samples 1 \
  --actor_lr 1e-6 \
  --clip_ae 0.2,0.28 \
  --infer_server vllm \
  --sequence_parallel_size 1 \
  --T 1.0 \
  --max_prompt_len 2k \
  --max_len 4k \
  --max_tokens_per_gpu 6144 \
  --num_ppo_iter 1 \
  --save_freq -1 \
  --test_freq 1 \
  --expname opsd_qwen3_1p7b_fixed_teacher_smoke_v4_train2k_val4k \
  --extra_args "data.filter_overlong_prompts=False trainer.project_name=opsd-smoke trainer.total_training_steps=1 ++data.dynamic_masked_solution=False ++data.min_masked_fraction=null ++data.max_masked_fraction=null ++data.mask_seed=null actor_rollout_ref.rollout.max_model_len=6144 actor_rollout_ref.rollout.max_num_batched_tokens=6144 actor_rollout_ref.rollout.val_kwargs.response_length=4096 algorithm.opsd.teacher_source=ground_truth algorithm.opsd.distill_loss=topk_jsd algorithm.opsd.topk=64 algorithm.opsd.distill_beta=0.5 algorithm.opsd.distill_token_clip=0.05 algorithm.opsd.distill_token_clip_tail=False algorithm.opsd.mode=opsd algorithm.opsd.teacher_model=fixed"
```

### 2026-04-10 OPSD / GRPO full-run configs requested by user

All of these are DFW `batch` runs on `4` nodes with `8` GPUs per node, W&B enabled, `Qwen3-1.7B`, deepmath train, comp-math validation, and the validation-only response-length override enabled.

Shared full baseline unless overridden:

- `--cluster cw-dfw`
- `--partition batch`
- `--nodes 4`
- `--gpus 8`
- `--enable_wandb`
- `--actor_model /hf_models/Qwen3-1.7B`
- `--prompt_data /data/rl/mathgen/deepmath_verl.jsonl`
- `--eval_data /data/rl/mathgen/comp_math_verl.jsonl`
- `--ae grpo`
- `--actor_lr 1e-6`
- `--n_prompts 32`
- `--n_samples 8`
- `--n_val_samples 1`
- `--T 0.85`
- `--num_epochs 5`
- `--num_ppo_iter 2`
- `--clip_ae 0.2,0.28`
- `--infer_server vllm`
- `--sequence_parallel_size 1`
- `--max_prompt_len 4k`
- `trainer.project_name=nemo-skills`
- `--save_freq 5`
- `--test_freq 10`
- `data.filter_overlong_prompts=False`
- `++data.dynamic_masked_solution=False`
- `++data.min_masked_fraction=null`
- `++data.max_masked_fraction=null`
- `++data.mask_seed=null`
- `actor_rollout_ref.rollout.val_kwargs.response_length=16384`
- `actor_rollout_ref.rollout.max_model_len=20480`
- `actor_rollout_ref.rollout.max_num_batched_tokens=20480`

OPSD full runs additionally pin the teacher prompt budget explicitly:

- `algorithm.opsd.max_prompt_length=4096`
- `algorithm.opsd.teacher_source=ground_truth`
- `algorithm.opsd.distill_loss=topk_jsd`
- `algorithm.opsd.topk=64`
- `algorithm.opsd.distill_beta=0.0`
- `algorithm.opsd.distill_token_clip=0.05`
- `algorithm.opsd.distill_token_clip_tail=False`

Requested 1.7B jobs:

1. Fixed teacher, long-train config
   - `--script_module recipe.opsd.main_opsd`
   - `--verl_config_file /home/siddjain/workspace/verl/verl_main/recipe/opsd/config/opsdv2_trainer.yaml`
   - `--max_len 12k`
   - `--max_tokens_per_gpu 12288`
   - training response length `8192`
   - `algorithm.opsd.mode=opsd`
   - `algorithm.opsd.teacher_model=fixed`

2. Teacher RLVR, long-train config
   - `--script_module recipe.opsd.main_opsd`
   - `--verl_config_file /home/siddjain/workspace/verl/verl_main/recipe/opsd/config/opsdv2_trainer.yaml`
   - `--max_len 12k`
   - `--max_tokens_per_gpu 12288`
   - training response length `8192`
   - `algorithm.opsd.mode=opsd_rlvr`
   - `algorithm.opsd.teacher_model=actor`

3. Vanilla GRPO, long-train config
   - `--script_module verl.trainer.main_ppo`
   - `--max_len 12k`
   - `--max_tokens_per_gpu 12288`
   - training response length `8192`
   - no OPSD overrides

4. Fixed teacher, short-train config
   - `--script_module recipe.opsd.main_opsd`
   - `--verl_config_file /home/siddjain/workspace/verl/verl_main/recipe/opsd/config/opsdv2_trainer.yaml`
   - `--max_len 5k`
   - `--max_tokens_per_gpu 5120`
   - training response length `1024`
   - `algorithm.opsd.mode=opsd`
   - `algorithm.opsd.teacher_model=fixed`
   - same validation response length override `16384`

### 2026-04-10 Qwen3-30B-A3B OPSD follow-on full runs

After the four 1.7B runs were live and had reached real training steps, the requested follow-on plan was to launch the first OPSD length config on `Qwen3-30B-A3B`.

Use the previously validated 30B startup pattern only as the TP sizing reference. The user later corrected the production requirement to `4` nodes, so the actual full runs should be launched on `4` nodes:

- `4` nodes
- `8` GPUs per node
- rollout `tensor_model_parallel_size=4`
- DFW `batch`

Keep the same first length config semantics as the 1.7B long runs:

- `--max_prompt_len 4k`
- `--max_len 12k`
- training response length `8192`
- `actor_rollout_ref.rollout.val_kwargs.response_length=16384`
- `actor_rollout_ref.rollout.max_model_len=20480`
- `actor_rollout_ref.rollout.max_num_batched_tokens=20480`
- `--max_tokens_per_gpu 12288`

Shared 30B OPSD full baseline:

- `--cluster cw-dfw`
- `--partition batch`
- `--nodes 4`
- `--gpus 8`
- `--enable_wandb`
- `--actor_model /hf_models/Qwen3-30B-A3B`
- `--prompt_data /data/rl/mathgen/deepmath_verl.jsonl`
- `--eval_data /data/rl/mathgen/comp_math_verl.jsonl`
- `--ae grpo`
- `--actor_lr 1e-6`
- `--n_prompts 32`
- `--n_samples 8`
- `--n_val_samples 1`
- `--T 0.85`
- `--num_epochs 5`
- `--num_ppo_iter 2`
- `--clip_ae 0.2,0.28`
- `--infer_server vllm`
- `--sequence_parallel_size 1`
- `trainer.project_name=nemo-skills`
- `data.filter_overlong_prompts=False`
- `++data.dynamic_masked_solution=False`
- `++data.min_masked_fraction=null`
- `++data.max_masked_fraction=null`
- `++data.mask_seed=null`
- `++actor_rollout_ref.rollout.tensor_model_parallel_size=4`
- `algorithm.opsd.max_prompt_length=4096`
- `algorithm.opsd.teacher_source=ground_truth`
- `algorithm.opsd.distill_loss=topk_jsd`
- `algorithm.opsd.topk=64`
- `algorithm.opsd.distill_beta=0.0`
- `algorithm.opsd.distill_token_clip=0.05`
- `algorithm.opsd.distill_token_clip_tail=False`

Variants:

1. Fixed teacher
   - `algorithm.opsd.mode=opsd`
   - `algorithm.opsd.teacher_model=fixed`

2. Teacher RLVR
   - `algorithm.opsd.mode=opsd_rlvr`
   - `algorithm.opsd.teacher_model=actor`

### Smoke 2: OPSD RLVR teacher on DFW interactive

Pending 2026-04-09 follow-up smoke:

- expname: `opsd_qwen3_1p7b_teacher_rlvr_smoke_v3_deepmath6k`
- same data and smoke knobs as Smoke 1
- only OPSD mode differs:
  - `algorithm.opsd.mode=opsd_rlvr`
  - `algorithm.opsd.teacher_model=actor`

```bash
cd /home/siddjain/workspace/verl/verl_main
conda run -n skills_latest python /home/siddjain/workspace/scripts/src/nemo_verl/skills_verl_submit.py \
  --cluster cw-dfw \
  --config_dir /home/siddjain/workspace/scripts/nemo_configs/cluster/codegen \
  --output_base_dir /output/rl \
  --local_verl_folder /home/siddjain/workspace/verl/verl_main \
  --reward_file /home/siddjain/workspace/scripts/src/nemo_verl/reward/verl_code_reward.py \
  --verl_config_file /home/siddjain/workspace/verl/verl_main/recipe/opsd/config/opsdv2_trainer.yaml \
  --script_module recipe.opsd.main_opsd \
  --actor_model /hf_models/Qwen3-1.7B \
  --prompt_data /data/rl/mathgen/deepmath_verl.jsonl \
  --eval_data /data/rl/mathgen/comp_math_verl.jsonl \
  --nodes 1 \
  --gpus 8 \
  --add_interactive \
  --ae grpo \
  --num_epochs 1 \
  --n_prompts 8 \
  --n_samples 2 \
  --n_val_samples 1 \
  --actor_lr 1e-6 \
  --clip_ae 0.2,0.28 \
  --infer_server vllm \
  --sequence_parallel_size 1 \
  --T 1.0 \
  --max_prompt_len 2k \
  --max_len 6k \
  --max_tokens_per_gpu 6144 \
  --num_ppo_iter 1 \
  --disable_val_before_train \
  --save_freq -1 \
  --test_freq 1 \
  --expname opsd_qwen3_1p7b_teacher_rlvr_smoke_v3_deepmath6k \
  --extra_args "data.filter_overlong_prompts=False trainer.project_name=opsd-smoke trainer.total_training_steps=1 ++data.dynamic_masked_solution=False ++data.min_masked_fraction=null ++data.max_masked_fraction=null ++data.mask_seed=null algorithm.opsd.teacher_source=ground_truth algorithm.opsd.distill_loss=topk_jsd algorithm.opsd.topk=64 algorithm.opsd.distill_beta=0.5 algorithm.opsd.distill_token_clip=0.05 algorithm.opsd.distill_token_clip_tail=False algorithm.opsd.mode=opsd_rlvr algorithm.opsd.teacher_model=actor"
```

### Monitoring

Use the standard DFW log mapping:

- output dir: `/output/rl/<expname>`
- remote dir: `/lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/<expname>`
- main log glob: `training-logs/main_<expname>-ppo-0_*_srun.log`

Useful checks:

```bash
ssh dfw "squeue -u siddjain"
ssh dfw "sacct -j <jobid> --format=JobID,State,Elapsed,NodeList%40 -P"
ssh dfw "tail -n 200 /lustre/fsw/portfolios/llmservice/users/siddjain/nemo-run/output/rl/<expname>/training-logs/main_<expname>-ppo-0_<jobid>_srun.log"
```

## 2026-04-09 DFW RL workflows

### Common submit context

- Cluster: `cw-dfw`
- Config dir: `/home/siddjain/workspace/scripts/nemo_configs/cluster/codegen`
- Submit wrapper: `/home/siddjain/workspace/scripts/src/nemo_verl/skills_verl_submit_addons.py`
- Checkout synced into the cluster job: `/home/siddjain/workspace/verl/verl_main`
- DFW container used by the wrapper: `/lustre/fsw/portfolios/llmservice/users/siddjain/containers/verl_vllm012_flashattn_20260321.sqsh`
- Smoke tests should omit `--enable_wandb`
- For plain deepmath/compmath prompts with no `{solution}` placeholder, explicitly disable dynamic masked-solution rewriting:
  - `++data.dynamic_masked_solution=False`
  - `++data.min_masked_fraction=null`
  - `++data.max_masked_fraction=null`
  - `++data.mask_seed=null`

### Workflow: GRPO uniform-outcome response-logprob reward

Use the standard DFW GRPO submit wrapper and add the uniform-outcome reward override on top of the normal batch reward manager.

Smoke/full command template:

```bash
cd /home/siddjain/workspace/verl/verl_main
conda run -n skills_latest python /home/siddjain/workspace/scripts/src/nemo_verl/skills_verl_submit_addons.py \
  --cluster cw-dfw \
  --config_dir /home/siddjain/workspace/scripts/nemo_configs/cluster/codegen \
  --output_base_dir /output/rl \
  --actor_model /hf_models/Qwen3-1.7B \
  --prompt_data /data/rl/mathgen/deepmath_verl.jsonl \
  --eval_data /data/rl/mathgen/comp_math_verl.jsonl \
  --nodes 1 \
  --add_interactive \
  --ae grpo \
  --num_epochs 1 \
  --n_prompts 32 \
  --n_samples 8 \
  --n_val_samples 8 \
  --actor_lr 1e-6 \
  --clip_ae 0.2,0.28 \
  --T 0.85 \
  --max_prompt_len 2k \
  --max_len 10k \
  --num_ppo_iter 1 \
  --num_training_jobs 1 \
  --save_freq 1 \
  --test_freq 1 \
  --local_verl_folder /home/siddjain/workspace/verl/verl_main \
  --expname grpo_qwen3_1p7b_deepmath_compmath_uniformoutcome_smoke \
  --extra_args "++data.dynamic_masked_solution=False ++data.min_masked_fraction=null ++data.max_masked_fraction=null ++data.mask_seed=null ++reward.reward_manager.name=batch ++reward.reward_kwargs.use_response_logprob_reward_for_uniform_outcome_groups=True"
```

All-failure-only variant:

```bash
--extra_args "++data.dynamic_masked_solution=False ++data.min_masked_fraction=null ++data.max_masked_fraction=null ++data.mask_seed=null ++reward.reward_manager.name=batch ++reward.reward_kwargs.use_response_logprob_reward_for_uniform_outcome_groups=True ++reward.reward_kwargs.uniform_outcome_response_logprob_reward_mode=all_failure"
```

Verification checklist:

- Validation must stay standard:
  - no uniform-outcome fields in `generations/validation/*.jsonl`
  - no validation-only reward inflation from the uniform path
- Rollout JSONL must include:
  - `group_all_success`
  - `group_all_failure`
  - `used_uniform_outcome_response_logprob_reward`
  - `uniform_outcome_response_logprob_reward`
  - `response_mean_logprob`
  - `response_median_logprob`
- Manually verify one row with:
  - `uniform_outcome_response_logprob_reward == exp(response_median_logprob - response_mean_logprob)`

### Workflow: Conditional-logprob recovery-ratio smoke on Qwen3-30B-A3B

This is the validated `reward_focus_tail` smoke that restored the objective-specific `conditional_logprob` path and used the standard validation reward manager.

```bash
cd /home/siddjain/workspace/verl/verl_main
conda run -n skills_latest python /home/siddjain/workspace/scripts/src/nemo_verl/skills_verl_submit_addons.py \
  --cluster cw-dfw \
  --config_dir /home/siddjain/workspace/scripts/nemo_configs/cluster/codegen \
  --output_base_dir /output/rl \
  --actor_model /hf_models/Qwen3-30B-A3B \
  --prompt_data /data/rl/mathgen/deepmath_verl.jsonl \
  --eval_data /data/rl/mathgen/comp_math_verl.jsonl \
  --nodes 1 \
  --add_interactive \
  --ae grpo \
  --num_epochs 1 \
  --n_prompts 8 \
  --n_samples 8 \
  --n_val_samples 8 \
  --actor_lr 1e-6 \
  --clip_ae 0.2,0.28 \
  --T 0.85 \
  --max_prompt_len 2k \
  --max_len 10k \
  --num_ppo_iter 1 \
  --num_training_jobs 1 \
  --save_freq 1 \
  --test_freq 1 \
  --local_verl_folder /home/siddjain/workspace/verl/verl_main \
  --expname condlogprob_qwen3_30b_a3b_deepmath_compmath_resp8k_rewardfocustail_recoveryratio_smoke \
  --extra_args "++data.dynamic_masked_solution=False ++data.min_masked_fraction=null ++data.max_masked_fraction=null ++data.mask_seed=null ++reward.reward_manager.name=conditional_logprob ++data.masked_solution_selection_mode=reward_focus_tail ++reward.reward_kwargs.conditioning_reward_mode=low_confidence_recovery_ratio ++reward.reward_kwargs.low_confidence_tail_percent=20 ++reward.reward_kwargs.low_confidence_min_tokens=1 ++reward.reward_kwargs.use_rlvr_reward_when_group_has_success=False ++reward.reward_kwargs.align_conditioning_focus_with_prompt_mask=True ++actor_rollout_ref.rollout.tensor_model_parallel_size=8 ++actor_rollout_ref.rollout.gpu_memory_utilization=0.5 ++data.gen_batch_size=8 ++actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 ++actor_rollout_ref.actor.use_dynamic_bsz=False"
```

Verification checklist:

- Validation JSONL must stay standard:
  - no `used_conditional_logprob`
  - no `cond_logprob`
  - no `low_confidence_recovery_ratio`
- Rollout JSONL must include:
  - `used_conditional_logprob=true`
  - `focus_logprob_improvement_mean`
  - `overall_logprob_improvement_mean`
  - `low_confidence_recovery_ratio`
- Manually verify:
  - `score = min(exp(focus_logprob_improvement_mean) / max(exp(overall_logprob_improvement_mean), 1e-3), 10.0)`

### Workflow: No-solution-prompt conditional-logprob variant

If the prompt template contains no `{solution}` or `{masked_solution}` placeholder, `conditional_logprob` can still run in prompt-only mode. The key difference is that the low-confidence focus set comes from prompt-only GT logprobs rather than from materialized prompt mask positions.

Required settings:

- `reward.reward_manager.name=conditional_logprob`
- `reward.reward_kwargs.conditioning_reward_mode=low_confidence_recovery_ratio`
- prompt data with no solution placeholder

Dataset safety note:

- For no-placeholder deepmath/compmath prompts, explicitly set:
  - `++data.dynamic_masked_solution=False`
  - `++data.min_masked_fraction=null`
  - `++data.max_masked_fraction=null`
  - `++data.mask_seed=null`

### Workflow: Qwen3-30B-A3B 4-node GRPO vs recovery-ratio full runs on deepmath/comp-math

This is the 4-node DFW `batch` launch pattern for comparing plain GRPO against prompt-only `conditional_logprob` low-confidence recovery ratio on the no-placeholder deepmath/comp-math pipeline.

Common settings:

- model: `/hf_models/Qwen3-30B-A3B`
- train: `/data/rl/mathgen/deepmath_verl.jsonl`
- eval: `/data/rl/mathgen/comp_math_verl.jsonl`
- nodes: `4`
- GPUs per node: `8`
- rollout TP: `4`
- `n_prompts=32`
- `n_samples=8`
- `n_val_samples=8`
- `actor_lr=1e-6`
- `T=0.85`
- `max_prompt_len=2k`
- `max_len=10k`
- `num_epochs=5`
- `num_ppo_iter=2`
- `save_freq=20`
- `test_freq=20`
- W&B enabled
- no-placeholder safety:
  - `++data.dynamic_masked_solution=False`
  - `++data.min_masked_fraction=null`
  - `++data.max_masked_fraction=null`
  - `++data.mask_seed=null`

Launcher:

- script: [`~/data/codex_tmp/run_grpo_and_recoveryratio_q30_4nodes.sh`](/home/siddjain/data/codex_tmp/run_grpo_and_recoveryratio_q30_4nodes.sh)

Submit with:

```bash
/home/siddjain/data/codex_tmp/run_grpo_and_recoveryratio_q30_4nodes.sh
```

GRPO variant:

- expname: `grpo_qwen3_30b_a3b_deepmath_compmath_resp8k_full_v1_tp4`
- active job: `11053708`

Recovery-ratio variant:

- expname: `condlogprob_qwen3_30b_a3b_deepmath_compmath_resp8k_nosol_recoveryratio20_full_v1_tp4`
- active job: `11053714`
- extra args:
  - `++reward.reward_manager.name=conditional_logprob`
  - `++data.masked_solution_selection_mode=reward_focus_tail`
  - `++reward.reward_kwargs.conditioning_reward_mode=low_confidence_recovery_ratio`
  - `++reward.reward_kwargs.low_confidence_tail_percent=20`
  - `++reward.reward_kwargs.low_confidence_min_tokens=1`
  - `++reward.reward_kwargs.use_rlvr_reward_when_group_has_success=False`

All-fail-groups-only recovery-ratio variant:

- expname: `condlogprob_qwen3_30b_a3b_deepmath_compmath_resp8k_nosol_recoveryratio20_allfail_full_v1_tp4`
- same config as the recovery-ratio variant above, except:
  - `++reward.reward_kwargs.use_rlvr_reward_when_group_has_success=True`
- effect:
  - conditional recovery-ratio reward is only used for prompt groups with no successful sample
  - prompt groups with any success stay on the rule reward path
- launcher:
  - [`~/data/codex_tmp/run_recoveryratio_q30_4nodes_allfail_groups.sh`](/home/siddjain/data/codex_tmp/run_recoveryratio_q30_4nodes_allfail_groups.sh)

All-token recovery-ratio variants:

- These use:
  - `++reward.reward_kwargs.low_confidence_tail_percent=1.0`
- In the current code, `1.0` means 100% of ground-truth tokens. For `low_confidence_recovery_ratio`, this now uses the unnormalized numerator `exp(focus_delta_mean)` instead of dividing by the overall-improvement denominator.
- Launchers:
  - all prompts use conditional reward:
    - [`~/data/codex_tmp/run_recoveryratio_q30_4nodes_alltokens_allprompts.sh`](/home/siddjain/data/codex_tmp/run_recoveryratio_q30_4nodes_alltokens_allprompts.sh)
  - only all-fail prompt groups use conditional reward:
    - [`~/data/codex_tmp/run_recoveryratio_q30_4nodes_alltokens_allfail_groups.sh`](/home/siddjain/data/codex_tmp/run_recoveryratio_q30_4nodes_alltokens_allfail_groups.sh)

### Workflow: 30B 4-node prompt4k val20k GRPO vs recovery-ratio

Use this when you want the same deepmath/comp-math 30B setup as the earlier 4-node runs, with:

- prompt length `4k`
- train response length `8k`
- validation response length `20k`
- `save_freq=10`
- `test_freq=20`

Common settings:

- model: `/hf_models/Qwen3-30B-A3B`
- train: `/data/rl/mathgen/deepmath_verl.jsonl`
- eval: `/data/rl/mathgen/comp_math_verl.jsonl`
- partition: `batch`
- nodes: `4`
- GPUs per node: `8`
- rollout TP: `4`
- `n_prompts=32`
- `n_samples=8`
- `n_val_samples=8`
- `actor_lr=1e-6`
- `T=0.85`
- `max_prompt_len=4096`
- train `max_len=12288`
- validation response length override `20480`
- rollout validation capacity:
  - `actor_rollout_ref.rollout.max_model_len=24576`
  - `actor_rollout_ref.rollout.max_num_batched_tokens=24576`
- `num_epochs=5`
- `num_ppo_iter=2`
- `save_freq=10`
- `test_freq=20`
- W&B enabled
- no-placeholder safety:
  - `++data.dynamic_masked_solution=False`
  - `++data.min_masked_fraction=null`
  - `++data.max_masked_fraction=null`
  - `++data.mask_seed=null`

Launcher:

- script: [`~/data/codex_tmp/run_grpo_and_recoveryratio_q30_4nodes_prompt4k_val20k.sh`](/home/siddjain/data/codex_tmp/run_grpo_and_recoveryratio_q30_4nodes_prompt4k_val20k.sh)

Variants:

- GRPO:
  - expname: `grpo_qwen3_30b_a3b_deepmath_compmath_prompt4k_train8k_val20k_full_v1_tp4`
- recovery-ratio tail 20, all-fail groups only:
  - expname: `condlogprob_qwen3_30b_a3b_deepmath_compmath_prompt4k_train8k_val20k_recoveryratio20_allfail_full_v1_tp4`
  - extra args:
    - `++reward.reward_manager.name=conditional_logprob`
    - `++data.masked_solution_selection_mode=reward_focus_tail`
    - `++reward.reward_kwargs.conditioning_reward_mode=low_confidence_recovery_ratio`
    - `++reward.reward_kwargs.low_confidence_tail_percent=20`
    - `++reward.reward_kwargs.low_confidence_min_tokens=1`
    - `++reward.reward_kwargs.use_rlvr_reward_when_group_has_success=True`
- recovery-ratio all tokens, all-fail groups only:
  - expname: `condlogprob_qwen3_30b_a3b_deepmath_compmath_prompt4k_train8k_val20k_recoveryratio_alltokens_allfail_full_v1_tp4`
  - same as above, except:
    - `++reward.reward_kwargs.low_confidence_tail_percent=1.0`
  - current code behavior:
    - this uses the unnormalized numerator `exp(focus_delta_mean)` instead of dividing by the overall-improvement denominator

Live cluster state at submission check:

- `11053708` GRPO: `RUNNING`
- `11053714` recovery-ratio: `PENDING (Resources)`

### Workflow: 30B no-validation smoke for GRPO vs recovery-ratio

Use this when you want to isolate the training-side `reward/standard_acc` behavior from any validation path.

Intent:

- Keep the 30B deepmath train setup faithful to the earlier 4-node TP4 runs.
- Remove validation entirely:
  - `--disable_val_before_train`
  - `--test_freq 1000`
  - `trainer.total_training_steps=6`
- Keep W&B disabled because this is a smoke.

Common settings:

- model: `/hf_models/Qwen3-30B-A3B`
- train: `/data/rl/mathgen/deepmath_verl.jsonl`
- eval: `/data/rl/mathgen/comp_math_verl.jsonl` (unused because validation is disabled)
- partition: `batch`
- nodes: `4`
- GPUs per node: `8`
- rollout TP: `4`
- `n_prompts=32`
- `n_samples=8`
- `n_val_samples=8`
- `actor_lr=1e-6`
- `T=0.85`
- `max_prompt_len=2k`
- `max_len=10k`
- `num_epochs=5`
- `num_ppo_iter=2`
- `save_freq=-1`
- `test_freq=1000`
- `trainer.total_training_steps=6`
- W&B disabled

Launcher:

- script: [`~/data/codex_tmp/run_grpo_and_recoveryratio_q30_4nodes_noval_smoke.sh`](/home/siddjain/data/codex_tmp/run_grpo_and_recoveryratio_q30_4nodes_noval_smoke.sh)
- if one side needs an isolated rerun after a cluster failure:
  - GRPO only: [`~/data/codex_tmp/run_grpo_q30_4nodes_noval_smoke.sh`](/home/siddjain/data/codex_tmp/run_grpo_q30_4nodes_noval_smoke.sh)

Variants:

- GRPO:
  - expname: `grpo_qwen3_30b_a3b_deepmath_compmath_resp8k_noval_smoke_v1_tp4`
  - isolated rerun expname: `grpo_qwen3_30b_a3b_deepmath_compmath_resp8k_noval_smoke_v2_tp4`
- recovery-ratio:
  - expname: `condlogprob_qwen3_30b_a3b_deepmath_compmath_resp8k_nosol_recoveryratio20_noval_smoke_v1_tp4`
  - extra args:
    - `++reward.reward_manager.name=conditional_logprob`
    - `++data.masked_solution_selection_mode=reward_focus_tail`
    - `++reward.reward_kwargs.conditioning_reward_mode=low_confidence_recovery_ratio`
    - `++reward.reward_kwargs.low_confidence_tail_percent=20`
    - `++reward.reward_kwargs.low_confidence_min_tokens=1`
    - `++reward.reward_kwargs.use_rlvr_reward_when_group_has_success=False`

### Workflow: Generation-dump token/logprob verification

The generation JSONL dumper now records:

- `response_tokens`
- `response_token_logprobs`

To verify on cluster:

1. Submit a smoke with rollout and validation dumps enabled.
2. Confirm the main log shows:
   - `Enabled actor_rollout_ref.rollout.calculate_log_probs because generation dump JSONLs are configured.`
3. Check both:
   - `generations/validation/*.jsonl`
   - `generations/rollout/*.jsonl`
4. For a sampled row:
   - `len(response_tokens) == len(response_token_logprobs)`

### Workflow: Qwen3-30B-A3B TP sizing and startup smokes

Validated startup smoke:

- model: `/hf_models/Qwen3-30B-A3B`
- nodes: `2`
- prompts: `128`
- samples: `8`
- rollout TP: `4`
- response length: `8192`
- result: startup succeeded through model load and reached validation generation

Submission pattern:

```bash
cd /home/siddjain/workspace/verl/verl_main
conda run -n skills_latest python /home/siddjain/workspace/scripts/src/nemo_verl/skills_verl_submit_addons.py \
  --cluster cw-dfw \
  --config_dir /home/siddjain/workspace/scripts/nemo_configs/cluster/codegen \
  --output_base_dir /output/rl \
  --actor_model /hf_models/Qwen3-30B-A3B \
  --prompt_data /data/rl/mathgen/deepmath_verl.jsonl \
  --eval_data /data/rl/mathgen/comp_math_verl.jsonl \
  --nodes 2 \
  --add_interactive \
  --ae grpo \
  --num_epochs 1 \
  --n_prompts 128 \
  --n_samples 8 \
  --n_val_samples 8 \
  --actor_lr 1e-6 \
  --clip_ae 0.2,0.28 \
  --T 0.85 \
  --max_prompt_len 2k \
  --max_len 10k \
  --num_ppo_iter 1 \
  --num_training_jobs 1 \
  --save_freq 1 \
  --test_freq 1 \
  --local_verl_folder /home/siddjain/workspace/verl/verl_main \
  --expname grpo_qwen3_30b_a3b_deepmath_compmath_vanilla_tp4_n128_smoke_2nodes \
  --extra_args "++data.dynamic_masked_solution=False ++data.min_masked_fraction=null ++data.max_masked_fraction=null ++data.mask_seed=null ++actor_rollout_ref.rollout.tensor_model_parallel_size=4"
```

### Known bad launch patterns

- `Qwen3-30B-A3B` full runs with rollout `tensor_model_parallel_size=1`:
  - failed during vLLM engine startup with no KV-cache memory available
- `Qwen3.5-35B-A3B` in the current DFW container:
  - failed at model init because the container `transformers` stack did not recognize `qwen3_5_moe`
- Plain deepmath/compmath runs with implicit masked-solution rewriting left enabled:
  - incorrect, because those prompts have no solution placeholder

### Workflow: Continue an existing 30B run from the latest checkpoint

For an already-started experiment directory that has checkpoint saves under:

- `checkpoints/global_step_*`
- `checkpoints/latest_checkpointed_iteration.txt`

the simplest continuation pattern is:

1. Reuse the exact same `expname`.
2. Reuse the exact same submit config.
3. Do not pass `--checkpoint_path`.
4. Set `--num_training_jobs N` to schedule `N` serial dependent continuation jobs.
5. If W&B is enabled and the intent is to continue the same run, explicitly pass the
   original `++wandb_id=...` in `--extra_args`.

Reason:

- `skills_verl_submit.py` sets `trainer.resume_mode=auto` when `--checkpoint_path` is omitted.
- VERL then finds the latest checkpoint inside the existing output directory for that `expname`.
- The dependent jobs therefore resume one after the other from the newest checkpoint written by the previous job in the chain.
- But `skills_verl_submit.py` also auto-mints a fresh `++wandb_id` on every submit, so W&B continuity is not preserved unless you override it explicitly in `extra_args`.

For the `Qwen3-30B-A3B`, `4`-node, `prompt4k / train8k / val20k` runs, the continuation submit keeps:

- `save_freq=10`
- `test_freq=20`
- `max_prompt_len=4096`
- `max_len=12288`
- `actor_rollout_ref.rollout.val_kwargs.response_length=20480`
- `actor_rollout_ref.rollout.max_model_len=24576`
- `actor_rollout_ref.rollout.max_num_batched_tokens=24576`
- `actor_rollout_ref.rollout.tensor_model_parallel_size=4`

and only changes:

- `--num_training_jobs 3`

This schedules `3` dependent continuation jobs behind the already-finished root run for each experiment.
