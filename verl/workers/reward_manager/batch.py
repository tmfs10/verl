# Copyright 2025 Individual Contributor: Mert Unsal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn


def _extract_scalar_reward_and_acc(score: Any) -> tuple[float, float]:
    if isinstance(score, dict):
        reward = float(score["score"])
        acc = float(score.get("acc", reward))
        return reward, acc

    reward = float(score)
    return reward, reward


def _select_uniform_outcome_group_usage(
    group_keys: list[Any],
    correctness: list[float],
    *,
    success_threshold: float,
    mode: str,
) -> tuple[list[bool], list[bool], list[bool]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, group_key in enumerate(group_keys):
        groups[str(group_key)].append(idx)

    supported_modes = {"all_success_or_failure", "all_failure", "all_success"}
    if mode not in supported_modes:
        raise ValueError(
            "uniform_outcome_response_logprob_reward_mode must be one of "
            f"{sorted(supported_modes)}, got {mode!r}"
        )

    group_all_success = [False] * len(group_keys)
    group_all_failure = [False] * len(group_keys)
    for group_indices in groups.values():
        outcomes = [float(correctness[idx]) > success_threshold for idx in group_indices]
        all_success = all(outcomes)
        all_failure = not any(outcomes)
        for idx in group_indices:
            group_all_success[idx] = all_success
            group_all_failure[idx] = all_failure

    if mode == "all_success_or_failure":
        selected = [group_all_success[idx] or group_all_failure[idx] for idx in range(len(group_keys))]
    elif mode == "all_failure":
        selected = list(group_all_failure)
    else:
        selected = list(group_all_success)

    return (
        selected,
        group_all_success,
        group_all_failure,
    )


def _resolve_uniform_outcome_group_keys(data: DataProto) -> tuple[list[Any], str]:
    prompt_group_ids = data.non_tensor_batch.get("prompt_group_id", None)
    if prompt_group_ids is not None:
        if hasattr(prompt_group_ids, "tolist"):
            prompt_group_ids = list(prompt_group_ids.tolist())
        else:
            prompt_group_ids = list(prompt_group_ids)
        if len(set(prompt_group_ids)) < len(prompt_group_ids):
            return prompt_group_ids, "prompt_group_id"

    uid_values = data.non_tensor_batch.get("uid", None)
    if uid_values is not None:
        if hasattr(uid_values, "tolist"):
            uid_values = list(uid_values.tolist())
        else:
            uid_values = list(uid_values)
        if len(set(uid_values)) < len(uid_values):
            return uid_values, "uid"

    prompt_tensor = data.batch.get("prompts", None)
    if prompt_tensor is not None:
        return [tuple(prompt_row.tolist()) for prompt_row in prompt_tensor.detach().cpu()], "prompts"

    return [f"sample_{idx}" for idx in range(len(data))], "sample_index"


def _compute_response_logprob_reward_stats(
    response_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[list[float], list[float], list[float]]:
    response_mask = response_mask.to(torch.bool)

    rewards: list[float] = []
    mean_logprobs: list[float] = []
    median_logprobs: list[float] = []
    for row_idx in range(response_log_probs.size(0)):
        valid_log_probs = response_log_probs[row_idx][response_mask[row_idx]]
        if valid_log_probs.numel() <= 0:
            mean_logprobs.append(0.0)
            median_logprobs.append(0.0)
            rewards.append(0.0)
            continue

        valid_log_probs = valid_log_probs.to(torch.float32)
        mean_logprob = float(valid_log_probs.mean().item())
        median_logprob = float(torch.quantile(valid_log_probs, 0.5).item())
        mean_logprobs.append(mean_logprob)
        median_logprobs.append(median_logprob)
        rewards.append(float(math.exp(median_logprob - mean_logprob)))

    return rewards, mean_logprobs, median_logprobs


def _compute_shortest_success_group_rewards(
    group_keys: list[Any],
    correctness: list[float],
    response_lengths: list[int],
    *,
    margin_percent: float,
    success_threshold: float,
    expected_group_size: int,
) -> dict[str, list[Any]]:
    """Select successful rollouts within ``margin_percent`` of the shortest success.

    Lengths must already be response-only token counts. This helper deliberately
    knows nothing about prompt tensors or padding so they cannot affect selection.
    """
    if not math.isfinite(margin_percent) or margin_percent < 0:
        raise ValueError(
            "shortest_success_margin_percent must be finite and non-negative, "
            f"got {margin_percent!r}"
        )
    if not math.isfinite(success_threshold):
        raise ValueError(f"shortest_success_threshold must be finite, got {success_threshold!r}")
    if expected_group_size <= 0:
        raise ValueError(
            "shortest_success_expected_rollouts_per_prompt must be positive, "
            f"got {expected_group_size}"
        )
    if not (len(group_keys) == len(correctness) == len(response_lengths)):
        raise ValueError(
            "Shortest-success inputs must have equal lengths: "
            f"groups={len(group_keys)}, correctness={len(correctness)}, responses={len(response_lengths)}"
        )

    groups: dict[str, list[int]] = defaultdict(list)
    normalized_group_keys: list[str] = []
    for idx, group_key in enumerate(group_keys):
        normalized_key = str(group_key)
        normalized_group_keys.append(normalized_key)
        groups[normalized_key].append(idx)

    bad_group_sizes = {
        group_key: len(indices)
        for group_key, indices in groups.items()
        if len(indices) != expected_group_size
    }
    if bad_group_sizes:
        raise ValueError(
            "Shortest-success reward requires complete rollout groups of size "
            f"{expected_group_size}; observed {bad_group_sizes}"
        )

    num_rows = len(group_keys)
    final_rewards = [0.0] * num_rows
    selected = [False] * num_rows
    group_has_success = [False] * num_rows
    group_min_tokens: list[int | None] = [None] * num_rows
    group_threshold_tokens: list[float | None] = [None] * num_rows
    success_length_ratio: list[float | None] = [None] * num_rows
    group_sizes = [0] * num_rows

    for group_key, indices in groups.items():
        for idx in indices:
            group_sizes[idx] = len(indices)
            length = int(response_lengths[idx])
            if length < 0:
                raise ValueError(f"Response token length must be non-negative, got {length} at row {idx}")

        success_indices = [idx for idx in indices if float(correctness[idx]) > success_threshold]
        if not success_indices:
            continue

        min_tokens = min(int(response_lengths[idx]) for idx in success_indices)
        if min_tokens <= 0:
            raise ValueError(
                "A verifier-successful rollout must contain at least one valid response token; "
                f"group={group_key!r}, minimum={min_tokens}"
            )
        threshold_tokens = float(min_tokens) * (1.0 + margin_percent / 100.0)

        for idx in indices:
            group_has_success[idx] = True
            group_min_tokens[idx] = min_tokens
            group_threshold_tokens[idx] = threshold_tokens
            if float(correctness[idx]) <= success_threshold:
                continue
            ratio = float(response_lengths[idx]) / float(min_tokens)
            success_length_ratio[idx] = ratio
            if float(response_lengths[idx]) <= threshold_tokens:
                selected[idx] = True
                final_rewards[idx] = 1.0

    return {
        "reward": final_rewards,
        "selected": selected,
        "response_tokens": [int(length) for length in response_lengths],
        "group_id": normalized_group_keys,
        "group_size": group_sizes,
        "group_has_success": group_has_success,
        "group_min_tokens": group_min_tokens,
        "group_threshold_tokens": group_threshold_tokens,
        "success_length_ratio": success_length_ratio,
        "margin_percent": [float(margin_percent)] * num_rows,
    }


def _compute_longest_success_penalty_group_rewards(
    group_keys: list[Any],
    correctness: list[float],
    response_lengths: list[int],
    *,
    no_penalty_margin_percent: float,
    success_threshold: float,
    expected_group_size: int,
) -> dict[str, list[Any]]:
    """Reward every success except exact longest ties when the span is too wide.

    Lengths must be response-only token counts. For a group with at least one
    verifier success, every success receives reward 1 when
    ``max_success_tokens <= min_success_tokens * (1 + margin / 100)``. When the
    inclusive condition is false, every successful rollout tied at the exact
    maximum length receives reward 0 and every shorter success receives 1.
    Incorrect rollouts always receive 0.
    """
    if not math.isfinite(no_penalty_margin_percent) or no_penalty_margin_percent < 0:
        raise ValueError(
            "longest_success_no_penalty_margin_percent must be finite and non-negative, "
            f"got {no_penalty_margin_percent!r}"
        )
    if not math.isfinite(success_threshold):
        raise ValueError(f"longest_success_threshold must be finite, got {success_threshold!r}")
    if expected_group_size <= 0:
        raise ValueError(
            "longest_success_expected_rollouts_per_prompt must be positive, "
            f"got {expected_group_size}"
        )
    if not (len(group_keys) == len(correctness) == len(response_lengths)):
        raise ValueError(
            "Longest-success-penalty inputs must have equal lengths: "
            f"groups={len(group_keys)}, correctness={len(correctness)}, responses={len(response_lengths)}"
        )

    groups: dict[str, list[int]] = defaultdict(list)
    normalized_group_keys: list[str] = []
    for idx, group_key in enumerate(group_keys):
        normalized_key = str(group_key)
        normalized_group_keys.append(normalized_key)
        groups[normalized_key].append(idx)

    bad_group_sizes = {
        group_key: len(indices)
        for group_key, indices in groups.items()
        if len(indices) != expected_group_size
    }
    if bad_group_sizes:
        raise ValueError(
            "Longest-success-penalty reward requires complete rollout groups of size "
            f"{expected_group_size}; observed {bad_group_sizes}"
        )

    num_rows = len(group_keys)
    final_rewards = [0.0] * num_rows
    penalized = [False] * num_rows
    group_has_success = [False] * num_rows
    group_within_margin = [False] * num_rows
    group_min_tokens: list[int | None] = [None] * num_rows
    group_max_tokens: list[int | None] = [None] * num_rows
    group_threshold_tokens: list[float | None] = [None] * num_rows
    success_length_ratio: list[float | None] = [None] * num_rows
    group_sizes = [0] * num_rows

    for group_key, indices in groups.items():
        for idx in indices:
            group_sizes[idx] = len(indices)
            length = int(response_lengths[idx])
            if length < 0:
                raise ValueError(f"Response token length must be non-negative, got {length} at row {idx}")

        success_indices = [idx for idx in indices if float(correctness[idx]) > success_threshold]
        if not success_indices:
            continue

        min_tokens = min(int(response_lengths[idx]) for idx in success_indices)
        max_tokens = max(int(response_lengths[idx]) for idx in success_indices)
        if min_tokens <= 0:
            raise ValueError(
                "A verifier-successful rollout must contain at least one valid response token; "
                f"group={group_key!r}, minimum={min_tokens}"
            )
        threshold_tokens = float(min_tokens) * (1.0 + no_penalty_margin_percent / 100.0)
        within_margin = float(max_tokens) <= threshold_tokens

        for idx in indices:
            group_has_success[idx] = True
            group_within_margin[idx] = within_margin
            group_min_tokens[idx] = min_tokens
            group_max_tokens[idx] = max_tokens
            group_threshold_tokens[idx] = threshold_tokens
            if float(correctness[idx]) <= success_threshold:
                continue
            response_tokens = int(response_lengths[idx])
            success_length_ratio[idx] = float(response_tokens) / float(min_tokens)
            is_penalized = not within_margin and response_tokens == max_tokens
            penalized[idx] = is_penalized
            final_rewards[idx] = 0.0 if is_penalized else 1.0

    return {
        "reward": final_rewards,
        "penalized": penalized,
        "response_tokens": [int(length) for length in response_lengths],
        "group_id": normalized_group_keys,
        "group_size": group_sizes,
        "group_has_success": group_has_success,
        "group_within_margin": group_within_margin,
        "group_min_tokens": group_min_tokens,
        "group_max_tokens": group_max_tokens,
        "group_threshold_tokens": group_threshold_tokens,
        "success_length_ratio": success_length_ratio,
        "no_penalty_margin_percent": [float(no_penalty_margin_percent)] * num_rows,
    }


@register("batch")
class BatchRewardManager(AbstractRewardManager):
    """
    A batch reward manager that computes rewards for a batch of data.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for decoding the responses.
        num_examine (int): The number of responses to examine.
        compute_score (callable): The function to compute the rewards.
        reward_fn_key (str): The key to use for the reward function.
        reward_kwargs (dict): The keyword arguments to pass to the reward function.
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score: RawRewardFn,
        reward_fn_key="data_source",
        config: Any | None = None,
        use_response_logprob_reward_for_uniform_outcome_groups: bool = False,
        uniform_outcome_group_success_threshold: float = 0.5,
        uniform_outcome_response_logprob_reward_mode: str = "all_success_or_failure",
        use_shortest_success_reward: bool = False,
        shortest_success_margin_percent: float = 10.0,
        shortest_success_threshold: float = 0.5,
        shortest_success_expected_rollouts_per_prompt: int | None = None,
        use_longest_success_penalty_reward: bool = False,
        longest_success_no_penalty_margin_percent: float = 50.0,
        longest_success_threshold: float = 0.5,
        longest_success_expected_rollouts_per_prompt: int | None = None,
        **reward_kwargs,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.use_response_logprob_reward_for_uniform_outcome_groups = bool(
            use_response_logprob_reward_for_uniform_outcome_groups
        )
        self.uniform_outcome_group_success_threshold = float(uniform_outcome_group_success_threshold)
        self.uniform_outcome_response_logprob_reward_mode = str(uniform_outcome_response_logprob_reward_mode)
        self.use_shortest_success_reward = bool(use_shortest_success_reward)
        self.use_longest_success_penalty_reward = bool(use_longest_success_penalty_reward)
        enabled_group_reward_modes = sum(
            (
                self.use_response_logprob_reward_for_uniform_outcome_groups,
                self.use_shortest_success_reward,
                self.use_longest_success_penalty_reward,
            )
        )
        if enabled_group_reward_modes > 1:
            raise ValueError(
                "use_response_logprob_reward_for_uniform_outcome_groups, use_shortest_success_reward, and "
                "use_longest_success_penalty_reward are mutually exclusive"
            )
        self.shortest_success_margin_percent = float(shortest_success_margin_percent)
        self.shortest_success_threshold = float(shortest_success_threshold)
        if shortest_success_expected_rollouts_per_prompt is None and config is not None:
            try:
                shortest_success_expected_rollouts_per_prompt = int(config.actor_rollout_ref.rollout.n)
            except (AttributeError, KeyError, TypeError, ValueError):
                shortest_success_expected_rollouts_per_prompt = None
        self.shortest_success_expected_rollouts_per_prompt = shortest_success_expected_rollouts_per_prompt
        if self.use_shortest_success_reward:
            if self.shortest_success_expected_rollouts_per_prompt is None:
                raise ValueError(
                    "Shortest-success reward requires actor_rollout_ref.rollout.n or "
                    "shortest_success_expected_rollouts_per_prompt"
                )
            if not math.isfinite(self.shortest_success_margin_percent) or self.shortest_success_margin_percent < 0:
                raise ValueError(
                    "shortest_success_margin_percent must be finite and non-negative, "
                    f"got {self.shortest_success_margin_percent!r}"
                )
            if not math.isfinite(self.shortest_success_threshold):
                raise ValueError(
                    f"shortest_success_threshold must be finite, got {self.shortest_success_threshold!r}"
                )
            if int(self.shortest_success_expected_rollouts_per_prompt) <= 0:
                raise ValueError(
                    "shortest_success_expected_rollouts_per_prompt must be positive, "
                    f"got {self.shortest_success_expected_rollouts_per_prompt!r}"
                )
            self.shortest_success_expected_rollouts_per_prompt = int(
                self.shortest_success_expected_rollouts_per_prompt
            )
        self.longest_success_no_penalty_margin_percent = float(longest_success_no_penalty_margin_percent)
        self.longest_success_threshold = float(longest_success_threshold)
        if longest_success_expected_rollouts_per_prompt is None and config is not None:
            try:
                longest_success_expected_rollouts_per_prompt = int(config.actor_rollout_ref.rollout.n)
            except (AttributeError, KeyError, TypeError, ValueError):
                longest_success_expected_rollouts_per_prompt = None
        self.longest_success_expected_rollouts_per_prompt = longest_success_expected_rollouts_per_prompt
        if self.use_longest_success_penalty_reward:
            if self.longest_success_expected_rollouts_per_prompt is None:
                raise ValueError(
                    "Longest-success-penalty reward requires actor_rollout_ref.rollout.n or "
                    "longest_success_expected_rollouts_per_prompt"
                )
            if (
                not math.isfinite(self.longest_success_no_penalty_margin_percent)
                or self.longest_success_no_penalty_margin_percent < 0
            ):
                raise ValueError(
                    "longest_success_no_penalty_margin_percent must be finite and non-negative, "
                    f"got {self.longest_success_no_penalty_margin_percent!r}"
                )
            if not math.isfinite(self.longest_success_threshold):
                raise ValueError(
                    f"longest_success_threshold must be finite, got {self.longest_success_threshold!r}"
                )
            if int(self.longest_success_expected_rollouts_per_prompt) <= 0:
                raise ValueError(
                    "longest_success_expected_rollouts_per_prompt must be positive, "
                    f"got {self.longest_success_expected_rollouts_per_prompt!r}"
                )
            self.longest_success_expected_rollouts_per_prompt = int(
                self.longest_success_expected_rollouts_per_prompt
            )
        # These variants require complete prompt-group context and therefore
        # cannot be computed one rollout at a time during generation.
        self.disable_async_reward_loop = (
            self.use_response_logprob_reward_for_uniform_outcome_groups
            or self.use_shortest_success_reward
            or self.use_longest_success_penalty_reward
        )
        self.reward_kwargs = reward_kwargs

    def verify(self, data):
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_str.append(response_str)

        ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        rollout_reward_scores = data.non_tensor_batch.get("reward_scores", [{} for _ in range(len(data))])
        extras = data.non_tensor_batch.get("extra_info", [{} for _ in range(len(data))])
        tool_extra_fields = data.non_tensor_batch.get("tool_extra_fields", [None for _ in range(len(data))])
        num_turns = data.non_tensor_batch.get("__num_turns__", [None for _ in range(len(data))])

        for i in range(len(data)):
            if tool_extra_fields[i] is not None:
                extras[i].update(tool_extra_fields[i])
            extras[i]["num_turns"] = num_turns[i]
            extras[i]["rollout_reward_scores"] = rollout_reward_scores[i]

        scores = self.compute_score(
            data_sources=data_sources,
            solution_strs=responses_str,
            ground_truths=ground_truths,
            extra_infos=extras,
            **self.reward_kwargs,
        )

        return scores

    def _resolve_response_log_probs(
        self,
        data: DataProto,
        selected_indices: list[int],
        *,
        actor_wg: Any | None,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        selected_data = data[selected_indices]
        response_mask = (
            selected_data.batch["response_mask"]
            if "response_mask" in selected_data.batch.keys()
            else selected_data.batch["attention_mask"][:, selected_data.batch["prompts"].shape[-1] :]
        )

        if "rollout_log_probs" in selected_data.batch.keys():
            return selected_data.batch["rollout_log_probs"], response_mask, "rollout_log_probs"
        if "old_log_probs" in selected_data.batch.keys():
            return selected_data.batch["old_log_probs"], response_mask, "old_log_probs"

        if actor_wg is None:
            raise ValueError(
                "BatchRewardManager requires either rollout_log_probs/old_log_probs in the batch or "
                "'actor_wg' to recompute response logprobs for uniform-outcome-group rewards."
            )

        required_batch_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        missing_batch_keys = [key for key in required_batch_keys if key not in selected_data.batch.keys()]
        if missing_batch_keys:
            raise ValueError(
                "BatchRewardManager cannot recompute response logprobs because the batch is missing "
                f"required keys: {missing_batch_keys}"
            )

        batch_keys = list(required_batch_keys)
        for optional_key in ("prompts", "response_mask"):
            if optional_key in selected_data.batch.keys():
                batch_keys.append(optional_key)

        non_tensor_batch_keys = []
        for optional_key in ("multi_modal_inputs", "uid"):
            if optional_key in selected_data.non_tensor_batch:
                non_tensor_batch_keys.append(optional_key)

        logprob_batch = selected_data.select(
            batch_keys=batch_keys,
            non_tensor_batch_keys=non_tensor_batch_keys,
        )
        logprob_batch_padded, pad_size = pad_dataproto_to_divisor(logprob_batch, actor_wg.world_size)
        logprob_output = actor_wg.compute_log_prob(logprob_batch_padded)
        response_log_probs = unpad_dataproto(logprob_output, pad_size=pad_size).batch["old_log_probs"]
        return response_log_probs, response_mask, "actor_recompute"

    def __call__(
        self,
        data: DataProto,
        return_dict: bool = False,
        actor_wg: Any | None = None,
    ) -> torch.Tensor | dict[str, Any]:
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        reward_from_rm_scores = None
        if not (
            self.use_response_logprob_reward_for_uniform_outcome_groups
            or self.use_shortest_success_reward
            or self.use_longest_success_penalty_reward
        ):
            reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]

        scores = self.verify(data)
        rewards: list[float] = []
        accs: list[float] = []
        already_printed: dict[str, Any] = {}
        selected_for_uniform_reward = [False] * len(data)
        group_all_success = [False] * len(data)
        group_all_failure = [False] * len(data)
        uniform_outcome_response_logprob_reward: list[float] = [0.0] * len(data)
        response_mean_logprob: list[float | None] = [None] * len(data)
        response_median_logprob: list[float | None] = [None] * len(data)
        response_logprob_source: list[str | None] = [None] * len(data)
        shortest_response_mask: torch.Tensor | None = None
        shortest_reward_token_indices: list[int | None] | None = None
        longest_response_mask: torch.Tensor | None = None
        longest_reward_token_indices: list[int | None] | None = None

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            score = scores[i]
            reward, acc = _extract_scalar_reward_and_acc(score)

            if isinstance(score, dict):
                for key, value in score.items():
                    reward_extra_info[key].append(value)

            rewards.append(reward)
            accs.append(acc)

            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine:
                response_str = self.tokenizer.decode(data.batch["responses"][i][:length], skip_special_tokens=True)
                prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", scores[i])
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        final_rewards = list(rewards)
        reward_extra_info["acc"] = list(accs)
        reward_extra_info["rule_reward"] = list(rewards)

        if self.use_shortest_success_reward:
            if "prompt_group_id" not in data.non_tensor_batch:
                raise KeyError("Shortest-success reward requires prompt_group_id in non_tensor_batch")
            if "response_mask" not in data.batch.keys():
                raise KeyError("Shortest-success reward requires response_mask in the response coordinate system")

            shortest_response_mask = data.batch["response_mask"]
            if shortest_response_mask.ndim != 2 or tuple(shortest_response_mask.shape) != tuple(
                data.batch["responses"].shape
            ):
                raise ValueError(
                    "Shortest-success response_mask must exactly match responses: "
                    f"mask={tuple(shortest_response_mask.shape)}, responses={tuple(data.batch['responses'].shape)}"
                )
            if not torch.isfinite(shortest_response_mask).all():
                raise ValueError("Shortest-success response_mask contains non-finite values")
            binary_mask = torch.logical_or(shortest_response_mask == 0, shortest_response_mask == 1)
            if not bool(binary_mask.all().item()):
                raise ValueError("Shortest-success response_mask must be binary")

            shortest_response_mask = shortest_response_mask.to(torch.bool)
            # Generated responses are left-aligned. A 1 after the first 0 would
            # make `sum(mask)` an invalid last-token position and is therefore
            # rejected rather than silently rewarding a PAD or gap position.
            seen_zero = torch.cummax((~shortest_response_mask).to(torch.int8), dim=-1).values.to(torch.bool)
            if bool((seen_zero & shortest_response_mask).any().item()):
                raise ValueError("Shortest-success response_mask must contain contiguous left-aligned valid tokens")

            group_ids = data.non_tensor_batch["prompt_group_id"]
            group_ids = list(group_ids.tolist()) if hasattr(group_ids, "tolist") else list(group_ids)
            if len(group_ids) != len(data):
                raise ValueError(
                    f"prompt_group_id length {len(group_ids)} does not match batch size {len(data)}"
                )
            response_token_lengths = shortest_response_mask.sum(dim=-1).to(torch.int64).cpu().tolist()
            shortest_fields = _compute_shortest_success_group_rewards(
                group_ids,
                accs,
                response_token_lengths,
                margin_percent=self.shortest_success_margin_percent,
                success_threshold=self.shortest_success_threshold,
                expected_group_size=int(self.shortest_success_expected_rollouts_per_prompt),
            )
            final_rewards = list(shortest_fields["reward"])
            shortest_reward_token_indices = [
                int(length) - 1 if int(length) > 0 else None for length in response_token_lengths
            ]
            reward_extra_info["shortest_success_reward"] = list(shortest_fields["reward"])
            reward_extra_info["shortest_success_selected"] = list(shortest_fields["selected"])
            reward_extra_info["shortest_success_response_tokens"] = list(shortest_fields["response_tokens"])
            reward_extra_info["shortest_success_group_id"] = list(shortest_fields["group_id"])
            reward_extra_info["shortest_success_group_size"] = list(shortest_fields["group_size"])
            reward_extra_info["shortest_success_group_has_success"] = list(
                shortest_fields["group_has_success"]
            )
            reward_extra_info["shortest_success_group_min_tokens"] = list(shortest_fields["group_min_tokens"])
            reward_extra_info["shortest_success_group_threshold_tokens"] = list(
                shortest_fields["group_threshold_tokens"]
            )
            reward_extra_info["shortest_success_length_ratio"] = list(shortest_fields["success_length_ratio"])
            reward_extra_info["shortest_success_margin_percent"] = list(shortest_fields["margin_percent"])
            reward_extra_info["shortest_success_reward_token_index"] = list(shortest_reward_token_indices)
            reward_extra_info["shortest_success_group_key_source"] = ["prompt_group_id"] * len(data)

        if self.use_longest_success_penalty_reward:
            if "prompt_group_id" not in data.non_tensor_batch:
                raise KeyError("Longest-success-penalty reward requires prompt_group_id in non_tensor_batch")
            if "response_mask" not in data.batch.keys():
                raise KeyError(
                    "Longest-success-penalty reward requires response_mask in the response coordinate system"
                )

            longest_response_mask = data.batch["response_mask"]
            if longest_response_mask.ndim != 2 or tuple(longest_response_mask.shape) != tuple(
                data.batch["responses"].shape
            ):
                raise ValueError(
                    "Longest-success-penalty response_mask must exactly match responses: "
                    f"mask={tuple(longest_response_mask.shape)}, responses={tuple(data.batch['responses'].shape)}"
                )
            if not torch.isfinite(longest_response_mask).all():
                raise ValueError("Longest-success-penalty response_mask contains non-finite values")
            binary_mask = torch.logical_or(longest_response_mask == 0, longest_response_mask == 1)
            if not bool(binary_mask.all().item()):
                raise ValueError("Longest-success-penalty response_mask must be binary")

            longest_response_mask = longest_response_mask.to(torch.bool)
            seen_zero = torch.cummax((~longest_response_mask).to(torch.int8), dim=-1).values.to(torch.bool)
            if bool((seen_zero & longest_response_mask).any().item()):
                raise ValueError(
                    "Longest-success-penalty response_mask must contain contiguous left-aligned valid tokens"
                )

            attention_response_mask = attention_mask[:, prompt_len:]
            if tuple(attention_response_mask.shape) != tuple(longest_response_mask.shape):
                raise ValueError(
                    "Longest-success-penalty attention-mask response slice must exactly match responses: "
                    f"attention_response={tuple(attention_response_mask.shape)}, "
                    f"responses={tuple(data.batch['responses'].shape)}"
                )
            if not torch.isfinite(attention_response_mask).all():
                raise ValueError("Longest-success-penalty attention-mask response slice contains non-finite values")
            binary_attention_mask = torch.logical_or(attention_response_mask == 0, attention_response_mask == 1)
            if not bool(binary_attention_mask.all().item()):
                raise ValueError("Longest-success-penalty attention-mask response slice must be binary")
            if not torch.equal(attention_response_mask.to(torch.bool), longest_response_mask):
                raise ValueError(
                    "Longest-success-penalty response_mask must equal the response slice of attention_mask"
                )

            group_ids = data.non_tensor_batch["prompt_group_id"]
            group_ids = list(group_ids.tolist()) if hasattr(group_ids, "tolist") else list(group_ids)
            if len(group_ids) != len(data):
                raise ValueError(
                    f"prompt_group_id length {len(group_ids)} does not match batch size {len(data)}"
                )
            response_token_lengths = longest_response_mask.sum(dim=-1).to(torch.int64).cpu().tolist()
            longest_fields = _compute_longest_success_penalty_group_rewards(
                group_ids,
                accs,
                response_token_lengths,
                no_penalty_margin_percent=self.longest_success_no_penalty_margin_percent,
                success_threshold=self.longest_success_threshold,
                expected_group_size=int(self.longest_success_expected_rollouts_per_prompt),
            )
            final_rewards = list(longest_fields["reward"])
            longest_reward_token_indices = [
                int(length) - 1 if int(length) > 0 else None for length in response_token_lengths
            ]
            reward_extra_info["longest_success_penalty_reward"] = list(longest_fields["reward"])
            reward_extra_info["longest_success_penalized"] = list(longest_fields["penalized"])
            reward_extra_info["longest_success_response_tokens"] = list(longest_fields["response_tokens"])
            reward_extra_info["longest_success_group_id"] = list(longest_fields["group_id"])
            reward_extra_info["longest_success_group_size"] = list(longest_fields["group_size"])
            reward_extra_info["longest_success_group_has_success"] = list(
                longest_fields["group_has_success"]
            )
            reward_extra_info["longest_success_group_within_margin"] = list(
                longest_fields["group_within_margin"]
            )
            reward_extra_info["longest_success_group_min_tokens"] = list(longest_fields["group_min_tokens"])
            reward_extra_info["longest_success_group_max_tokens"] = list(longest_fields["group_max_tokens"])
            reward_extra_info["longest_success_group_no_penalty_threshold_tokens"] = list(
                longest_fields["group_threshold_tokens"]
            )
            reward_extra_info["longest_success_length_ratio"] = list(longest_fields["success_length_ratio"])
            reward_extra_info["longest_success_no_penalty_margin_percent"] = list(
                longest_fields["no_penalty_margin_percent"]
            )
            reward_extra_info["longest_success_reward_token_index"] = list(longest_reward_token_indices)
            reward_extra_info["longest_success_group_key_source"] = ["prompt_group_id"] * len(data)

        if self.use_response_logprob_reward_for_uniform_outcome_groups:
            group_keys, group_key_source = _resolve_uniform_outcome_group_keys(data)
            (
                selected_for_uniform_reward,
                group_all_success,
                group_all_failure,
            ) = _select_uniform_outcome_group_usage(
                group_keys,
                accs,
                success_threshold=self.uniform_outcome_group_success_threshold,
                mode=self.uniform_outcome_response_logprob_reward_mode,
            )
            selected_indices = [idx for idx, use_uniform_reward in enumerate(selected_for_uniform_reward) if use_uniform_reward]
            if selected_indices:
                response_log_probs, response_mask, resolved_logprob_source = self._resolve_response_log_probs(
                    data,
                    selected_indices,
                    actor_wg=actor_wg,
                )
                selected_rewards, selected_mean_logprobs, selected_median_logprobs = (
                    _compute_response_logprob_reward_stats(response_log_probs, response_mask)
                )
                for pos, idx in enumerate(selected_indices):
                    final_rewards[idx] = selected_rewards[pos]
                    uniform_outcome_response_logprob_reward[idx] = selected_rewards[pos]
                    response_mean_logprob[idx] = selected_mean_logprobs[pos]
                    response_median_logprob[idx] = selected_median_logprobs[pos]
                    response_logprob_source[idx] = resolved_logprob_source

            reward_extra_info["group_all_success"] = list(group_all_success)
            reward_extra_info["group_all_failure"] = list(group_all_failure)
            reward_extra_info["used_uniform_outcome_response_logprob_reward"] = list(selected_for_uniform_reward)
            reward_extra_info["uniform_outcome_response_logprob_reward"] = list(uniform_outcome_response_logprob_reward)
            reward_extra_info["response_mean_logprob"] = list(response_mean_logprob)
            reward_extra_info["response_median_logprob"] = list(response_median_logprob)
            reward_extra_info["response_logprob_source"] = list(response_logprob_source)

        for i, reward in enumerate(final_rewards):
            if self.use_longest_success_penalty_reward:
                reward_token_index = longest_reward_token_indices[i]
                if reward_token_index is not None:
                    reward_tensor[i, reward_token_index] = reward
            elif self.use_shortest_success_reward:
                reward_token_index = shortest_reward_token_indices[i]
                if reward_token_index is not None:
                    reward_tensor[i, reward_token_index] = reward
            else:
                length = valid_response_lengths[i].item()
                if length > 0:
                    reward_tensor[i, length - 1] = reward

        if self.use_shortest_success_reward:
            nonzero_reward = reward_tensor != 0
            valid_nonzero = nonzero_reward & shortest_response_mask
            pad_nonzero = nonzero_reward & ~shortest_response_mask
            reward_extra_info["shortest_success_reward_tensor_coordinate"] = ["response_only"] * len(data)
            reward_extra_info["shortest_success_reward_tensor_width"] = [int(reward_tensor.shape[1])] * len(data)
            reward_extra_info["shortest_success_nonzero_reward_token_count"] = (
                nonzero_reward.sum(dim=-1).to(torch.int64).cpu().tolist()
            )
            reward_extra_info["shortest_success_valid_response_nonzero_reward_token_count"] = (
                valid_nonzero.sum(dim=-1).to(torch.int64).cpu().tolist()
            )
            reward_extra_info["shortest_success_pad_nonzero_reward_token_count"] = (
                pad_nonzero.sum(dim=-1).to(torch.int64).cpu().tolist()
            )
            reward_extra_info["shortest_success_reward_tensor_row_sum"] = (
                reward_tensor.sum(dim=-1).to(torch.float32).cpu().tolist()
            )

        if self.use_longest_success_penalty_reward:
            nonzero_reward = reward_tensor != 0
            valid_nonzero = nonzero_reward & longest_response_mask
            pad_nonzero = nonzero_reward & ~longest_response_mask
            reward_extra_info["longest_success_reward_tensor_coordinate"] = ["response_only"] * len(data)
            reward_extra_info["longest_success_reward_tensor_width"] = [int(reward_tensor.shape[1])] * len(data)
            reward_extra_info["longest_success_nonzero_reward_token_count"] = (
                nonzero_reward.sum(dim=-1).to(torch.int64).cpu().tolist()
            )
            reward_extra_info["longest_success_valid_response_nonzero_reward_token_count"] = (
                valid_nonzero.sum(dim=-1).to(torch.int64).cpu().tolist()
            )
            reward_extra_info["longest_success_pad_nonzero_reward_token_count"] = (
                pad_nonzero.sum(dim=-1).to(torch.int64).cpu().tolist()
            )
            reward_extra_info["longest_success_reward_tensor_row_sum"] = (
                reward_tensor.sum(dim=-1).to(torch.float32).cpu().tolist()
            )

        data.batch["acc"] = torch.tensor(accs, dtype=torch.float32, device=prompt_ids.device)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor

    async def run_single(self, data: DataProto) -> dict[str, Any]:
        if self.use_shortest_success_reward or self.use_longest_success_penalty_reward:
            raise RuntimeError("Grouped success-length reward requires a complete prompt-group batch")
        assert len(data) == 1, "BatchRewardManager.run_single only supports a single item"
        score = self.verify(data)[0]

        reward_extra_info = {}
        if isinstance(score, dict):
            reward = score["score"]
            reward_extra_info.update(score)
        else:
            reward = score
            reward_extra_info["acc"] = score

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
