# Self-Correction Summary

This document summarizes the self-correction work added around math verification, multi-turn retry behavior, and reward handling.

## Main Behavior Added

- Added a dedicated math self-correction interaction in `verl/interactions/math_verify_interaction.py`.
- Added support for three interaction modes:
  - `verifier`: answer, score it, and retry only when the answer is wrong.
  - `repeat_until_stable`: keep retrying until the extracted final answer repeats.
  - `s2r`: answer, self-verify with an explicit verdict prompt, then retry if needed.
- Added support for two prompt-history modes:
  - `full_history`: normal multi-turn conversation history.
  - `question_with_past_answers`: rebuild later turns from the original question plus prior extracted answers, instead of replaying the full transcript.

## State And Metadata Tracking

- Track extracted answers, completed answers, answer-correctness history, and verification outcomes across turns.
- Record `reward_mode=last_completed_turn` metadata so reward can be based on the last completed non-interrupted answer instead of intermediate retries.
- Added S2R-specific metadata such as verification verdict history, answer/verification agreement, and current phase bookkeeping.

## Reward Changes

- Updated `verl/utils/reward_score/__init__.py` to support `reward_mode="last_completed_turn"`.
- Added optional wrong-answer entropy shaping through `entropy_bonus_coef`.
- Entropy bonus is stored back into `extra_info` for inspection and logging.

## Agent Loop Plumbing

- Updated `verl/experimental/agent_loop/tool_agent_loop.py` so the loop:
  - passes stop reasons into the interaction,
  - still scores the final turn when generation stops due to a cap,
  - supports prompt reset for `question_with_past_answers`,
  - keeps PPO aligned to the actual prompt used on each turn.
- Related loop/interaction metadata tests were added to cover this prompt-reset and metadata behavior.

## Configuration Surface

- The interaction now reads:
  - `interaction_mode`
  - `turn_context_mode`
  - `entropy_bonus_coef`
  - `repeat_until_stable_prompt`
  - `s2r_verify_prompt`
  - `s2r_retry_prompt`
- `s2r` is explicitly restricted to `turn_context_mode="full_history"`.

## Files Added Or Changed

- `verl/interactions/math_verify_interaction.py`
- `verl/utils/reward_score/__init__.py`
- `verl/experimental/agent_loop/tool_agent_loop.py`
- `verl/experimental/agent_loop/single_turn_agent_loop.py`
- `tests/interactions/test_math_verify_interaction.py`
- `tests/experimental/agent_loop/test_tool_agent_loop_interaction_metadata.py`
- `tests/utils/reward_score/test_math_entropy_bonus.py`

## Practical Result

- The repo can now run verifier-driven or self-verification-driven retry loops for math tasks.
- Rewards can be attributed to the final completed answer rather than to intermediate failed attempts.
- Training sees the same prompt structure that the generator actually used on each retry turn.
