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
        use_response_logprob_reward_for_uniform_outcome_groups: bool = False,
        uniform_outcome_group_success_threshold: float = 0.5,
        uniform_outcome_response_logprob_reward_mode: str = "all_success_or_failure",
        **reward_kwargs,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.use_response_logprob_reward_for_uniform_outcome_groups = bool(
            use_response_logprob_reward_for_uniform_outcome_groups
        )
        # This reward variant depends on prompt-group context, so the per-sample async reward
        # loop cannot compute it correctly during rollout streaming.
        self.disable_async_reward_loop = self.use_response_logprob_reward_for_uniform_outcome_groups
        self.uniform_outcome_group_success_threshold = float(uniform_outcome_group_success_threshold)
        self.uniform_outcome_response_logprob_reward_mode = str(uniform_outcome_response_logprob_reward_mode)
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
        if not self.use_response_logprob_reward_for_uniform_outcome_groups:
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
            length = valid_response_lengths[i].item()
            if length > 0:
                reward_tensor[i, length - 1] = reward

        data.batch["acc"] = torch.tensor(accs, dtype=torch.float32, device=prompt_ids.device)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor

    async def run_single(self, data: DataProto) -> dict[str, Any]:
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
