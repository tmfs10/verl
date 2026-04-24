## 2026-04-15

### Math Self-Correction Interaction Stack

- Added a dedicated multi-turn math interaction in `/home/siddjain/workspace/verl/verl_main/verl/interactions/math_verify_interaction.py`.
- Added three math self-correction modes:
  - `verifier`
  - `repeat_until_stable`
  - `s2r`
- Added two turn-context modes:
  - `full_history`
  - `question_with_past_answers`
- Added per-turn state tracking for:
  - extracted answer history
  - completed-answer history
  - answer correctness history
  - S2R verification verdict and agreement history
- Added turn-budget reminder support so interactions can inject per-turn response-budget hints into prompts.

### Tool-Agent Multi-Turn PPO Alignment

- Updated `/home/siddjain/workspace/verl/verl_main/verl/experimental/agent_loop/tool_agent_loop.py` so the interaction path:
  - passes `stop_reason` into the interaction
  - still records terminal assistant turns on cap-based termination
  - carries interaction metadata into `extra_fields`
  - supports prompt reset via `next_generation_messages`
  - aligns PPO with the actual prompt used for the selected retry turn in `question_with_past_answers` mode

### Reward Handling For Self-Correction

- Updated `/home/siddjain/workspace/verl/verl_main/verl/utils/reward_score/__init__.py` to support `reward_mode=\"last_completed_turn\"`.
- Added optional wrong-answer entropy shaping over `answer_history`.
- Final default for `entropy_bonus_coef` was set back to `0.0`; entropy shaping is opt-in.
- The resulting self-correction path remains sparse final-reward PPO, not dense per-turn PPO shaping.

### Tests, Notes, And Smoke Configs

- Added interaction tests in `/home/siddjain/workspace/verl/verl_main/tests/interactions/test_math_verify_interaction.py`.
- Added reward-entropy coverage in `/home/siddjain/workspace/verl/verl_main/tests/utils/reward_score/test_math_entropy_bonus.py`.
- Added self-correction interaction configs under `/home/siddjain/workspace/verl/verl_main/examples/self_correction_smoke/interaction_configs`.
- Added a repo summary note at `/home/siddjain/workspace/verl/verl_main/self_correction.md`.

## 2026-04-12

### Validation Metric Export Filtering

- Changed PPO validation logging so only `best@N/*`, `maj@N/*`, and `worst@N/*` metric families are exported.
- Validation no longer logs plain `mean@N`, `std@N`, or `val-aux/num_turns/*` panels.
- This was done in the trainer export path, so the internal validation aggregation math is unchanged; only the emitted logger/W&B keys are filtered.
- Added a focused CPU unit test for the validation metric-section filter in `tests/trainer/ppo/test_generation_dump_on_cpu.py`.

## 2026-04-11

### Timeout-Aware Checkpointing

- Added deadline-based checkpoint triggering to PPO training in addition to the existing ESI expiry logic.
- New trainer config knobs:
  - `trainer.checkpoint_must_save_by`
  - `trainer.checkpoint_save_duration`
- If `trainer.checkpoint_must_save_by` is set, PPO uses that explicit `DD:HH:MM:SS` deadline relative to trainer start and it overrides `SLURM_JOB_END_TIME`.
- Otherwise, if `SLURM_JOB_END_TIME` is present, PPO automatically treats the SLURM walltime as a checkpoint deadline.
- Timeout-triggered checkpoints now stop training early after the save so the job does not immediately spend the remaining walltime on validation or another long step.
- Added focused CPU tests in `tests/utils/ckpt/test_timeout_save_ckpt_on_cpu.py`.

## 2026-04-09

### Validation Response-Length Override

- Added `actor_rollout_ref.rollout.val_kwargs.response_length` so validation can use a different response-length cap than training rollout.
- Wired the validation override through the PPO trainer, async rollout sampling params, HF rollout, TRT-LLM async rollout, and ES-FSDP validation.
- Fixed `HFRollout` to pad/truncate against the effective per-request response length instead of always using the training rollout length.
- Fixed the agent-loop rollout path so validation no longer truncates generated responses back to the training `response_length` in single-turn/tool-agent postprocessing and tensor padding.
- Added a focused CPU regression test in `tests/workers/rollout/test_hf_rollout_response_length_on_cpu.py`.

### OPSD / OPSDv2

- Added generalized sparse top-k distillation controls to `OPSDConfig`. Token clipping is based on https://arxiv.org/html/2601.18734v3:
  - `algorithm.opsd.distill_beta`
  - `algorithm.opsd.distill_token_clip`
  - `algorithm.opsd.distill_token_clip_tail`
- Generalized the sparse top-k distillation loss so the same path can represent:
  - forward KL at `beta=0.0`
  - JSD at `beta=0.5`
  - reverse-KL-style interpolation at `beta=1.0`
- Added an option to exclude the collapsed tail bucket from token clipping when using sparse top-k state.
- Updated the OPSD actor loss path and metrics to use the generalized top-k divergence and report:
  - divergence beta
  - whether tail clipping is enabled
  - clipped and unclipped token statistics
  - clip fraction
- Kept the existing legacy JSD metrics populated when `beta == 0.5`.

### Fixed Teacher

- Implemented a real `teacher_model=fixed` path for OPSD distillation-only mode.
- `fixed` now means a separate frozen teacher module initialized from the actor snapshot, not any adapter-state toggle on the live actor.
- `teacher_model=fixed` is rejected for `algorithm.opsd.mode=opsd_rlvr`.
- The separate OPSD teacher module is checkpointed under `opsd_teacher/` and reloaded on resume so the fixed snapshot is preserved.
- Resume now errors if a fixed-teacher run is resumed without the saved fixed-teacher snapshot.

### Configs and Presets

- Updated `recipe/opsd/config/opsd_trainer.yaml` to expose the generalized divergence controls.
- Added `recipe/opsd/config/opsdv2_trainer.yaml` with paper-oriented defaults:
  - ground-truth teacher source
  - top-k distillation
  - `distill_beta: 0.0`
  - `distill_token_clip: 0.05`

### Tests

- Added/updated OPSD tests covering:
  - config validation for `distill_beta`, `distill_token_clip`, `distill_token_clip_tail`
  - config validation for `teacher_model=fixed`
  - sparse generalized top-k divergence behavior
  - clipping with and without tail clipping
  - fixed-teacher snapshot behavior
  - EMA teacher update behavior

### Workflow Documentation

- Added `WORKFLOWS.md` entries for the local Qwen3-1.7B OPSD smoke workflows.
- Documented:
  - tiny smoke dataset creation under `~/data/opsd_smoke`
  - fixed-teacher OPSD smoke command
  - teacher-RLVR smoke command
  - W&B-disabled defaults

### Validation Notes

- `py_compile` passed on the touched Python files.
- Local smoke attempts were blocked by the host NVIDIA/CUDA state:
  - `nvidia-smi` reports a driver/library mismatch
  - the local host currently cannot initialize CUDA/Triton cleanly
- Installed `codetiming` into the local `deepseek` conda env because the recipe import path required it before reaching runtime.

### GRPO Uniform-Outcome Reward

- Added a batch reward-manager path that can replace the per-sample rule reward with
  `exp(response_median_logprob - response_mean_logprob)` when an entire prompt group is uniformly successful or uniformly failing.
- Added `reward.reward_kwargs.use_response_logprob_reward_for_uniform_outcome_groups`.
- Added `reward.reward_kwargs.uniform_outcome_response_logprob_reward_mode` with:
  - `all_success_or_failure`
  - `all_failure`
  - `all_success`
- Added rollout metadata for this path:
  - `group_all_success`
  - `group_all_failure`
  - `used_uniform_outcome_response_logprob_reward`
  - `uniform_outcome_response_logprob_reward`
  - `response_mean_logprob`
  - `response_median_logprob`
  - `response_logprob_source`
- `BatchRewardManager` now disables the async reward loop when this group-level reward is enabled, because the reward depends on prompt-group context and actor-side logprob recomputation.
- Added `BatchRewardManager.run_single(...)` for the standard per-sample async path when the group-level override is not in use.
- Added focused CPU coverage in `tests/workers/reward_manager/test_batch_reward_uniform_outcome_on_cpu.py`.

### Validation Reward Isolation

- Added `_build_validation_reward_config(...)` in `verl/trainer/main_ppo.py` to keep validation on the standard path when training uses experimental reward logic.
- Validation now always forces `use_response_logprob_reward_for_uniform_outcome_groups=False`, even if the training config or explicit `val_reward_kwargs` enables it.
- `conditional_logprob` training now automatically validates with the standard `batch` reward manager.
- Added regression tests in `tests/trainer/test_main_ppo_validation_reward_config_on_cpu.py`.

### Conditional Logprob / Recovery-Ratio Restore

- Restored the objective-specific `conditional_logprob` reward behavior in `verl/workers/reward_manager/conditional.py` instead of collapsing to a single mean-logprob path.
- Supported and revalidated:
  - `mean_logprob`
  - `low_confidence_recovery_ratio`
  - `low_confidence_token_topk_recall`
  - `low_confidence_token_mrr`
- Marked `ConditionalLogProbRewardManager` as `disable_async_reward_loop = True`, because it needs actor-side conditional logprob computation and prompt-group context.
- Preserved the existing group-success RLVR fallback via `use_rlvr_reward_when_group_has_success`.
- Kept `truncate_conditioning_response_at_last_think` and `align_conditioning_focus_with_prompt_mask` as explicit controls.
- Special-cased `low_confidence_recovery_ratio` with `low_confidence_tail_percent=1.0` / `100%` so it uses the unnormalized focus-improvement numerator `exp(focus_delta_mean)` instead of dividing by the overall-improvement denominator. This prevents the all-token case from collapsing to an approximately constant `1.0`.
- Updated reward-loop legacy migration in `verl/experimental/reward_loop/reward_loop.py` so explicit `reward.reward_kwargs` survive legacy config translation instead of being clobbered by `reward_model.reward_kwargs`.
- Added focused CPU coverage in `tests/workers/reward_manager/test_conditional_logprob_reward.py`.

### Dataset / Prompt-Masking Safety

- Fixed `RLHFDataset._build_prompt_ids_override(...)` so plain prompts with no `{solution}` / `{masked_solution}` placeholders no longer crash with `No masked-solution placeholders found in prompt template.`
- `masked_solution_focus_min_tokens: null` now resolves safely to `1`.
- This makes plain deepmath/compmath prompts safe to run with `dynamic_masked_solution=False` and prevents placeholder-specific logic from being forced on datasets that do not support it.

### Generation Dump / Token Logprobs

- Added `response_tokens` and `response_token_logprobs` to rollout and validation JSONL dumps.
- Added `_extract_response_tokens_and_logprobs(...)` in `verl/trainer/ppo/ray_trainer.py`.
- Auto-enables `actor_rollout_ref.rollout.calculate_log_probs` whenever `trainer.rollout_data_dir` or `trainer.validation_data_dir` is configured, so dump JSONLs include per-token logprobs without requiring an extra manual flag.
- Added focused CPU tests in `tests/trainer/ppo/test_generation_dump_on_cpu.py`.

### Cluster Validation Notes

- DFW smoke validation confirmed that the uniform-outcome reward path writes the expected rollout metadata and matches `exp(median - mean)` numerically from real JSONL artifacts.
- DFW smoke validation confirmed that `conditional_logprob` `low_confidence_recovery_ratio` matches the rollout JSONL values numerically and that validation stays on the standard batch reward path.
- DFW smoke validation confirmed that generation dumps now contain decoded response tokens and per-token logprobs.
- DFW 30B startup validation showed:
  - `TP=1` full launches fail during vLLM KV-cache initialization with `No available memory for the cache blocks`
  - `TP=4`, `n_prompts=128`, `2` interactive nodes clears model load, launches vLLM, and reaches validation generation

### Documentation

- Expanded `WORKFLOWS.md` with the DFW RL workflows and failure notes from this session:
  - uniform-outcome GRPO
  - conditional-logprob recovery-ratio
  - generation-dump verification
  - 30B TP sizing / startup smoke
