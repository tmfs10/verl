# OPSD Summary

This document summarizes the OPSD and OPSD+RLVR work added to the VERL codebase and the related trainer plumbing.

## Main Algorithm Surface

- Added a typed OPSD config block in `verl/trainer/config/algorithm.py`.
- Added support for:
  - `mode: opsd | opsd_rlvr`
  - `ground_truth_field`
  - `teacher_prefix`
  - `teacher_suffix`
  - `distill_loss: topk_jsd | sampled_reverse_kl`
  - `topk`
  - `mix_weight`
  - `balance_mode: none | grad_norm`
  - `balance_param_subset`
  - `offpolicy_is_mode`
  - `offpolicy_is_clip`
  - `behavior_logprob_source`
- Added validation for the new OPSD config options.

## New Recipe

- Added a dedicated recipe entrypoint in `recipe/opsd/main_opsd.py`.
- Added recipe-specific trainer, actor, loss, and worker files:
  - `recipe/opsd/opsd_trainer.py`
  - `recipe/opsd/dp_actor.py`
  - `recipe/opsd/opsd_loss.py`
  - `recipe/opsd/fsdp_workers.py`
  - `recipe/opsd/config/opsd_trainer.yaml`

## Teacher Conditioning

- The OPSD trainer rebuilds a teacher-conditioned prompt using the configured ground-truth answer field.
- Teacher prompts are formed with:
  - the original prompt,
  - `teacher_prefix`,
  - the ground-truth answer,
  - `teacher_suffix`.
- This lets the same model be evaluated under both student conditioning and teacher conditioning on the same sampled completion.

## Losses

- `opsd` mode trains only the distillation branch.
- `opsd_rlvr` adds an RLVR PPO-style teacher branch on top of the distillation branch.
- Added two distillation options:
  - `topk_jsd`: sparse top-k-plus-tail JSD between student and teacher token distributions.
  - `sampled_reverse_kl`: sampled-token reverse-KL-style surrogate using student token log-probs and teacher token log-probs.

## Sparse Top-k Refactor

- The original OPSD path retained full response-window logits and then built sparse JSD support in the loss code.
- This was rewritten so `_forward_opsd_branch()` now returns compact sparse state instead of full logits.
- The sparse state contains:
  - `topk_ids`
  - `topk_log_probs`
  - `topk_probs`
  - `tail_prob`
  - `tail_log_prob`
  - rollout-token `log_probs`
- The JSD and alignment helpers now consume that sparse state directly.
- This removed the old full-logit fp32 loss-side copies and significantly reduced OPSD memory use.
- The sparse-union support code also deduplicates the student/teacher top-k union before lookup.

## RLVR And Importance Sampling

- `opsd_rlvr` uses teacher-conditioned PPO on the teacher branch.
- Added off-policy importance-sampling options for correcting teacher PPO against student behavior log-probs.
- Added support for `offpolicy_is_mode=none` for direct teacher-branch PPO without the IS correction.
- Fixed sequence-level IS handling so summed log-ratios are safely clipped before exponentiation.

## Balancing

- Added `balance_mode=none` for the original plain convex combination.
- Added `balance_mode=grad_norm` so `mix_weight` can mean balanced branch contribution rather than raw scalar averaging.
- Added branch-scale combination helpers in `recipe/opsd/opsd_loss.py`.
- Added actor-side measurement of branch-specific grad norms on selected parameters for balancing/debugging.

## Diagnostics And Logging

- Added OPSD actor metrics such as:
  - distillation loss
  - JSD loss
  - reverse-KL loss
  - teacher RLVR loss
  - teacher/student token log-prob summaries
  - teacher-minus-student log-prob deltas
  - top-k alignment metrics for the JSD path
- Added rollout diagnostics to support correctness-vs-teacher-delta checks and similar offline analyses.

## Reward / Trainer Plumbing Related To OPSD

- Fixed reward-manager fallback paths needed by the OPSD submission path.
- Fixed `BatchRewardManager(num_examine=0)` handling.
- Made rollout and validation generation dumping default-on in the PPO trainer config:
  - `rollout_data_dir: ${trainer.default_local_dir}/rollout_data`
  - `validation_data_dir: ${trainer.default_local_dir}/validation_data`
- This made it easier to inspect OPSD rollouts, scores, and diagnostics from JSONL dumps.

## Tests

- Added OPSD loss tests under `tests/recipe/opsd/`.
- Added coverage for sparse-top-k behavior, reverse-KL behavior, stop-grad expectations, and related helpers.

## Practical Result

- The repo now has a dedicated OPSD recipe with both pure distillation and OPSD+RLVR variants.
- The original top-k JSD path exists alongside a lighter reverse-KL option.
- The main memory bottleneck from storing full response logits in the OPSD loss path was removed by switching to compact sparse state.
