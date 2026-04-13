"""
Conditional log-probability reward manager.

This reward manager supports several related modes:

1. Plain conditional log-prob reward:
   reward = log P(gt_response | prompt, generated_response_prefix)
2. Low-confidence recovery reward:
   compare prompt-only and prompt+generated conditioning on the ground-truth
   continuation, emphasizing the continuation tokens that were least likely
   under the prompt-only baseline.
3. Focus-token top-k recall reward:
   select the prompt-only lowest-confidence continuation tokens, then reward
   conditioned rollouts by the fraction of those focus tokens whose
   ground-truth token lands inside the conditioned model's top-k candidate
   list.
4. Focus-token MRR reward:
   select the prompt-only lowest-confidence continuation tokens, then reward
   conditioned rollouts by the mean reciprocal rank of the ground-truth token.

By default, the manager behaves as a mixed RLVR/logprob reward:
- In top-k recall / MRR mode with ``low_confidence_tail_percent < 100``, use
  the conditioned reward for every rollout.
- In top-k recall / MRR mode with ``low_confidence_tail_percent = 100``, keep
  the normal rule reward on successful rollouts and use the conditioned reward
  only on unsuccessful ones.
- For the other reward modes, keep the prompt-group RLVR fallback: if a prompt
  group has any successful rollout, use the normal rule reward for every
  rollout from that prompt; otherwise use the conditional reward for the whole
  group.

The generated response prefix used for conditioning is truncated through the
last ``</think>`` tag by default. If no closing think tag is present, that
rollout falls back to the prompt-group floor reward, or to ``0`` if the entire
prompt group is invalid for conditional evaluation.
"""

from __future__ import annotations

import functools
import inspect
import math
from collections import defaultdict
from typing import Any, Optional

import torch

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.reward_score import default_compute_score
from verl.utils.torch_functional import get_response_mask
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


def _extract_conditioning_response_prefix(
    response_text: str,
    *,
    truncate_at_last_think: bool,
    think_end_tag: str,
) -> tuple[str, bool]:
    if not truncate_at_last_think or not response_text:
        return response_text, True

    last_tag_idx = response_text.rfind(think_end_tag)
    if last_tag_idx < 0:
        return "", False
    return response_text[: last_tag_idx + len(think_end_tag)], True


def _nested_get(config: Any, path: str, default: Any = None) -> Any:
    current = config
    for part in path.split("."):
        if current is None:
            return default
        if hasattr(current, "get"):
            next_value = current.get(part, None)
        elif isinstance(current, dict):
            next_value = current.get(part, None)
        else:
            next_value = getattr(current, part, None)
        current = next_value
    return default if current is None else current


def _infer_conditioning_total_length_budget(config: Any, explicit_budget: Optional[int]) -> Optional[int]:
    if explicit_budget is not None:
        return int(explicit_budget)

    prompt_budget = _nested_get(config, "data.max_prompt_length")
    response_budget = _nested_get(config, "data.max_response_length")
    combined_budget: Optional[int] = None
    if prompt_budget is not None and response_budget is not None:
        combined_budget = int(prompt_budget) + int(response_budget)

    logprob_budget = _nested_get(config, "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu")
    if logprob_budget is not None:
        logprob_budget = int(logprob_budget)

    if combined_budget is not None and logprob_budget is not None:
        return min(combined_budget, logprob_budget)
    if combined_budget is not None:
        return combined_budget
    if logprob_budget is not None:
        return logprob_budget
    return None


def _extract_scalar_reward_and_acc(score: Any) -> tuple[float, float]:
    if isinstance(score, dict):
        reward = float(score.get("score", 0.0))
        if "acc" in score:
            acc = float(score["acc"])
        else:
            acc = 1.0 if reward > 0.5 else 0.0
        return reward, acc

    reward = float(score)
    return reward, reward


def _compute_score_supports_batched_api(compute_score: Any) -> bool:
    current = compute_score
    seen: set[int] = set()

    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        try:
            params = inspect.signature(current).parameters
        except (TypeError, ValueError):
            params = {}

        if all(name in params for name in ("data_sources", "solution_strs", "ground_truths")):
            return True
        if all(name in params for name in ("data_source", "solution_str", "ground_truth")):
            return False

        wrapped = getattr(current, "__wrapped__", None)
        if callable(wrapped):
            current = wrapped
            continue

        if isinstance(current, functools.partial):
            if current.args and callable(current.args[0]):
                current = current.args[0]
                continue
            current = current.func
            continue

        break

    return False


def _normalize_batched_rule_results(results: Any, expected_len: int) -> list[Any]:
    if isinstance(results, (list, tuple)):
        if len(results) != expected_len:
            raise ValueError(
                "Batched compute_score returned the wrong number of per-sample results: "
                f"expected {expected_len}, got {len(results)}"
            )
        return list(results)

    if isinstance(results, dict):
        normalized: list[dict[str, Any]] = []
        for idx in range(expected_len):
            sample_result: dict[str, Any] = {}
            for key, value in results.items():
                if isinstance(value, (list, tuple)):
                    if len(value) != expected_len:
                        raise ValueError(
                            "Batched compute_score returned a dict with mismatched list lengths: "
                            f"key {key!r} has len {len(value)}, expected {expected_len}"
                        )
                    sample_result[key] = value[idx]
                else:
                    sample_result[key] = value
            normalized.append(sample_result)
        return normalized

    raise TypeError(
        "Batched compute_score must return either a sequence of per-sample results or "
        f"a dict of per-sample lists, got {type(results).__name__}"
    )


def _resolve_group_keys(data: DataProto) -> list[Any]:
    for key in ("prompt_group_id", "uid"):
        values = data.non_tensor_batch.get(key, None)
        if values is None:
            continue
        if hasattr(values, "tolist"):
            return list(values.tolist())
        return list(values)
    return [f"sample_{idx}" for idx in range(len(data))]


def _select_conditional_reward_usage(
    uids: list[Any],
    correctness: list[float],
    *,
    use_rule_reward_when_group_has_success: bool,
    success_threshold: float,
    conditioning_reward_mode: str,
    low_confidence_tail_percent: float,
) -> tuple[list[bool], list[bool]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, uid in enumerate(uids):
        groups[str(uid)].append(idx)

    group_has_success = [False] * len(uids)
    for group_indices in groups.values():
        has_success = any(float(correctness[idx]) > success_threshold for idx in group_indices)
        for idx in group_indices:
            group_has_success[idx] = has_success

    if conditioning_reward_mode in {"low_confidence_token_topk_recall", "low_confidence_token_mrr"}:
        tail_fraction = _resolve_tail_fraction(low_confidence_tail_percent)
        if tail_fraction < 1.0:
            return [True] * len(uids), group_has_success
        return [float(correctness[idx]) <= success_threshold for idx in range(len(uids))], group_has_success

    if not use_rule_reward_when_group_has_success:
        return [True] * len(uids), group_has_success

    return [not has_success for has_success in group_has_success], group_has_success


def _select_group_logprob_usage(
    uids: list[Any],
    correctness: list[float],
    *,
    use_rule_reward_when_group_has_success: bool,
    success_threshold: float,
) -> tuple[list[bool], list[bool]]:
    return _select_conditional_reward_usage(
        uids,
        correctness,
        use_rule_reward_when_group_has_success=use_rule_reward_when_group_has_success,
        success_threshold=success_threshold,
        conditioning_reward_mode="mean_logprob",
        low_confidence_tail_percent=100.0,
    )


def _resolve_tail_fraction(tail_percent: float) -> float:
    if tail_percent <= 0:
        raise ValueError(f"low_confidence_tail_percent must be positive, got {tail_percent}")
    if tail_percent <= 1.0:
        return float(tail_percent)
    return float(tail_percent) / 100.0


def _select_low_confidence_token_indices(
    prompt_only_log_probs: torch.Tensor,
    valid_token_count: int,
    *,
    tail_percent: float,
    min_tokens: int,
) -> list[int]:
    if valid_token_count <= 0:
        return []

    tail_fraction = _resolve_tail_fraction(tail_percent)
    tail_count = max(min_tokens, int(math.ceil(valid_token_count * tail_fraction)))
    tail_count = min(valid_token_count, tail_count)

    valid_log_probs = prompt_only_log_probs[:valid_token_count]
    lowest_positions = torch.argsort(valid_log_probs, dim=-1)[:tail_count]
    return lowest_positions.detach().cpu().tolist()


def _left_pad_prompt_tensors(
    prompt_ids: torch.Tensor,
    prompt_attn: torch.Tensor,
    prompt_pos: torch.Tensor,
    *,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    batch_size = prompt_ids.size(0)
    prompt_valid_lengths = [int(prompt_attn[row_idx].sum().item()) for row_idx in range(batch_size)]
    max_prompt_valid_len = max(prompt_valid_lengths, default=0)
    if max_prompt_valid_len <= 0:
        raise ValueError("Conditional logprob reward requires non-empty prompts.")

    trimmed_prompt_ids = torch.full(
        (batch_size, max_prompt_valid_len),
        fill_value=pad_token_id,
        dtype=prompt_ids.dtype,
        device=prompt_ids.device,
    )
    trimmed_prompt_attn = torch.zeros(
        (batch_size, max_prompt_valid_len),
        dtype=prompt_attn.dtype,
        device=prompt_attn.device,
    )

    if prompt_pos.dim() == 3:
        trimmed_prompt_pos = torch.zeros(
            (batch_size, prompt_pos.size(1), max_prompt_valid_len),
            dtype=prompt_pos.dtype,
            device=prompt_pos.device,
        )
    else:
        trimmed_prompt_pos = torch.zeros(
            (batch_size, max_prompt_valid_len),
            dtype=prompt_pos.dtype,
            device=prompt_pos.device,
        )

    for row_idx, valid_len in enumerate(prompt_valid_lengths):
        trimmed_prompt_ids[row_idx, -valid_len:] = prompt_ids[row_idx, -valid_len:]
        trimmed_prompt_attn[row_idx, -valid_len:] = 1
        if prompt_pos.dim() == 3:
            trimmed_prompt_pos[row_idx, :, -valid_len:] = prompt_pos[row_idx, :, -valid_len:]
        else:
            trimmed_prompt_pos[row_idx, -valid_len:] = prompt_pos[row_idx, -valid_len:]

    return trimmed_prompt_ids, trimmed_prompt_attn, trimmed_prompt_pos, prompt_valid_lengths


def _pad_right_token_lists(
    token_lists: list[torch.Tensor],
    *,
    pad_token_id: int,
    device: torch.device,
    token_dtype: torch.dtype,
    mask_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max((tokens.numel() for tokens in token_lists), default=0)
    if max_len <= 0:
        raise ValueError("Expected at least one non-empty token list.")

    token_batch = torch.full((len(token_lists), max_len), fill_value=pad_token_id, dtype=token_dtype, device=device)
    token_mask = torch.zeros((len(token_lists), max_len), dtype=mask_dtype, device=device)

    for row_idx, tokens in enumerate(token_lists):
        token_count = tokens.numel()
        if token_count <= 0:
            continue
        token_batch[row_idx, :token_count] = tokens.to(device=device, dtype=token_dtype)
        token_mask[row_idx, :token_count] = 1

    return token_batch, token_mask


def _resolve_pad_token_id(meta_info: Any, tokenizer: Any) -> int:
    pad_token_id: Optional[int] = None

    if isinstance(meta_info, dict):
        meta_pad_token_id = meta_info.get("pad_token_id", None)
        if meta_pad_token_id is not None:
            pad_token_id = int(meta_pad_token_id)

    if pad_token_id is None:
        tokenizer_pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if tokenizer_pad_token_id is not None:
            pad_token_id = int(tokenizer_pad_token_id)

    if pad_token_id is None:
        tokenizer_eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if tokenizer_eos_token_id is not None:
            pad_token_id = int(tokenizer_eos_token_id)

    if pad_token_id is None:
        pad_token_id = 0

    return pad_token_id


@register("conditional_logprob")
class ConditionalLogProbRewardManager(AbstractRewardManager):
    """
    Mixed RLVR / conditional log-prob reward manager.
    """

    disable_async_reward_loop = True

    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        compute_score=None,
        reward_fn_key: str = "data_source",
        reduction: str = "mean",
        gt_field_name: str = "ground_truth_response",
        max_gt_len: Optional[int] = None,
        truncate_conditioning_response_at_last_think: bool = True,
        think_end_tag: str = "</think>",
        use_rlvr_reward_when_group_has_success: bool = True,
        rule_reward_success_threshold: float = 0.5,
        conditioning_reward_mode: str = "mean_logprob",
        low_confidence_tail_percent: float = 20.0,
        low_confidence_min_tokens: int = 1,
        conditioned_token_topk: int = 20,
        min_overall_improvement_ratio: float = 1e-3,
        max_logprob_reward: float = 10.0,
        max_conditioning_total_length: Optional[int] = None,
        align_conditioning_focus_with_prompt_mask: bool = False,
        config: Any = None,
        **reward_kwargs: Any,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        assert reduction in ("mean", "sum"), f"Unsupported reduction: {reduction}"
        self.reduction = reduction
        self.gt_field_name = gt_field_name
        self.max_gt_len = max_gt_len
        self.truncate_conditioning_response_at_last_think = bool(truncate_conditioning_response_at_last_think)
        self.think_end_tag = think_end_tag
        self.use_rlvr_reward_when_group_has_success = bool(use_rlvr_reward_when_group_has_success)
        self.rule_reward_success_threshold = float(rule_reward_success_threshold)
        self.conditioning_reward_mode = str(conditioning_reward_mode)
        valid_modes = {
            "mean_logprob",
            "low_confidence_recovery_ratio",
            "low_confidence_token_topk_recall",
            "low_confidence_token_mrr",
        }
        if self.conditioning_reward_mode not in valid_modes:
            raise ValueError(
                f"Unsupported conditioning_reward_mode {self.conditioning_reward_mode!r}; "
                f"expected one of {sorted(valid_modes)}"
            )
        self.low_confidence_tail_percent = float(low_confidence_tail_percent)
        self.low_confidence_min_tokens = max(1, int(low_confidence_min_tokens))
        self.conditioned_token_topk = max(1, int(conditioned_token_topk))
        self.min_overall_improvement_ratio = float(min_overall_improvement_ratio)
        self.max_logprob_reward = float(max_logprob_reward)
        self.max_conditioning_total_length = _infer_conditioning_total_length_budget(config, max_conditioning_total_length)
        self.align_conditioning_focus_with_prompt_mask = bool(align_conditioning_focus_with_prompt_mask)
        self.reward_kwargs = reward_kwargs

    def _get_ground_truth_text(self, data_item) -> Optional[str]:
        gt = data_item.non_tensor_batch.get(self.gt_field_name, None)
        if gt is not None:
            return gt

        reward_model = data_item.non_tensor_batch.get("reward_model", None)
        if isinstance(reward_model, dict):
            return reward_model.get("ground_truth", None)
        return None

    def _compute_rule_results(self, data: DataProto, response_texts: list[str]) -> list[Any]:
        ground_truths = [self._get_ground_truth_text(item) for item in data]
        data_sources = list(data.non_tensor_batch[self.reward_fn_key])
        rollout_reward_scores = data.non_tensor_batch.get("reward_scores", [{} for _ in range(len(data))])
        extras = data.non_tensor_batch.get("extra_info", [{} for _ in range(len(data))])
        tool_extra_fields = data.non_tensor_batch.get("tool_extra_fields", [None for _ in range(len(data))])
        num_turns = data.non_tensor_batch.get("__num_turns__", [None for _ in range(len(data))])

        prepared_extras: list[dict[str, Any]] = []
        for idx in range(len(data)):
            extra = dict(extras[idx]) if extras[idx] is not None else {}
            if tool_extra_fields[idx] is not None:
                extra.update(tool_extra_fields[idx])
            extra["num_turns"] = num_turns[idx]
            extra["rollout_reward_scores"] = rollout_reward_scores[idx]
            prepared_extras.append(extra)

        if _compute_score_supports_batched_api(self.compute_score):
            results = self.compute_score(
                data_sources=data_sources,
                solution_strs=response_texts,
                ground_truths=ground_truths,
                extra_infos=prepared_extras,
                **self.reward_kwargs,
            )
            return _normalize_batched_rule_results(results, len(data))

        return [
            self.compute_score(
                data_source=data_sources[idx],
                solution_str=response_texts[idx],
                ground_truth=ground_truths[idx],
                extra_info=prepared_extras[idx],
                **self.reward_kwargs,
            )
            for idx in range(len(data))
        ]

    def _build_conditioning_prefix(
        self,
        prompt_ids: torch.Tensor,
        prompt_attn: torch.Tensor,
        prompt_pos: torch.Tensor,
        conditioned_token_lists: list[torch.Tensor],
        *,
        pad_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_prefix_len = max((tokens.numel() for tokens in conditioned_token_lists), default=0)
        if max_prefix_len <= 0:
            return prompt_ids, prompt_attn, prompt_pos

        prefix_ids, prefix_attn = _pad_right_token_lists(
            conditioned_token_lists,
            pad_token_id=pad_token_id,
            device=prompt_ids.device,
            token_dtype=prompt_ids.dtype,
            mask_dtype=prompt_attn.dtype,
        )

        delta_pos = torch.arange(1, max_prefix_len + 1, device=prompt_ids.device, dtype=prompt_pos.dtype)
        delta_pos = delta_pos.unsqueeze(0).expand(prompt_ids.size(0), -1)
        if prompt_pos.dim() == 3:
            delta_pos = delta_pos.view(prompt_ids.size(0), 1, -1).expand(prompt_ids.size(0), prompt_pos.size(1), -1)
        prefix_pos = prompt_pos[..., -1:] + delta_pos

        return (
            torch.cat([prompt_ids, prefix_ids], dim=-1),
            torch.cat([prompt_attn, prefix_attn], dim=-1),
            torch.cat([prompt_pos, prefix_pos], dim=-1),
        )

    def _build_logprob_batch(
        self,
        prompt_ids: torch.Tensor,
        prompt_attn: torch.Tensor,
        prompt_pos: torch.Tensor,
        gt_token_lists: list[torch.Tensor],
        *,
        pad_token_id: int,
        conditioned_token_lists: list[torch.Tensor] | None = None,
        topk_k: int = 0,
        calculate_token_mrr: bool = False,
        uid_values: list[Any] | None = None,
    ) -> tuple[DataProto, torch.Tensor]:
        prompt_rep_ids = prompt_ids
        prompt_rep_attn = prompt_attn
        prompt_rep_pos = prompt_pos
        if conditioned_token_lists is not None:
            prompt_rep_ids, prompt_rep_attn, prompt_rep_pos = self._build_conditioning_prefix(
                prompt_rep_ids,
                prompt_rep_attn,
                prompt_rep_pos,
                conditioned_token_lists,
                pad_token_id=pad_token_id,
            )

        gt_ids, gt_mask = _pad_right_token_lists(
            gt_token_lists,
            pad_token_id=pad_token_id,
            device=prompt_rep_ids.device,
            token_dtype=prompt_rep_ids.dtype,
            mask_dtype=prompt_rep_attn.dtype,
        )

        seq_concat = torch.cat([prompt_rep_ids, gt_ids], dim=-1)
        gt_len = gt_ids.size(1)
        delta_pos = torch.arange(1, gt_len + 1, device=prompt_rep_ids.device, dtype=prompt_rep_pos.dtype)
        delta_pos = delta_pos.unsqueeze(0).expand(prompt_rep_ids.size(0), -1)
        if prompt_rep_pos.dim() == 3:
            delta_pos = delta_pos.view(prompt_rep_ids.size(0), 1, -1).expand(prompt_rep_ids.size(0), prompt_rep_pos.size(1), -1)
        gt_position_ids = prompt_rep_pos[..., -1:] + delta_pos
        position_ids = torch.cat([prompt_rep_pos, gt_position_ids], dim=-1)
        attention_mask = torch.cat([prompt_rep_attn, gt_mask], dim=-1)

        non_tensors = {}
        if uid_values is not None:
            non_tensors["uid"] = uid_values

        batch = DataProto.from_dict(
            tensors={
                "prompts": prompt_rep_ids,
                "responses": gt_ids,
                "input_ids": seq_concat,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            non_tensors=non_tensors or None,
        )
        if topk_k > 0:
            batch.meta_info["topk_token_ids_k"] = int(topk_k)
        if calculate_token_mrr:
            batch.meta_info["calculate_token_mrr"] = True
        return batch, gt_mask

    def _compute_log_prob(self, actor_wg: Any, batch: DataProto) -> DataProto:
        world_size = int(getattr(actor_wg, "world_size", 1) or 1)
        if world_size > 1:
            padded_batch, pad_size = pad_dataproto_to_divisor(batch, world_size)
            output = actor_wg.compute_log_prob(padded_batch)
            return unpad_dataproto(output, pad_size=pad_size)
        return actor_wg.compute_log_prob(batch)

    def __call__(self, data: DataProto, return_dict: bool = False, **kwargs: Any):
        actor_wg = kwargs.get("actor_wg", None)
        if actor_wg is None:
            raise ValueError("ConditionalLogProbRewardManager requires 'actor_wg' kwarg (actor worker group)")

        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        full_attention_mask = data.batch["attention_mask"]
        prompt_attn = full_attention_mask[:, :prompt_len]
        prompt_pos = data.batch["position_ids"][..., :prompt_len]
        responses = data.batch["responses"]
        response_mask = (
            data.batch["response_mask"]
            if "response_mask" in data.batch.keys()
            else full_attention_mask[:, prompt_len:]
        )
        valid_response_lengths = response_mask.sum(dim=-1)
        pad_token_id = _resolve_pad_token_id(data.meta_info, self.tokenizer)

        valid_response_ids: list[torch.Tensor] = []
        response_texts: list[str] = []
        for row_idx in range(len(data)):
            valid_len = int(valid_response_lengths[row_idx].item())
            response_ids = responses[row_idx, :valid_len].detach().cpu()
            valid_response_ids.append(response_ids)
            response_texts.append(self.tokenizer.decode(response_ids, skip_special_tokens=True))

        rule_results = self._compute_rule_results(data, response_texts)
        rule_rewards: list[float] = []
        accs: list[float] = []
        reward_extra_info = defaultdict(list)
        for result in rule_results:
            rule_reward, acc = _extract_scalar_reward_and_acc(result)
            rule_rewards.append(rule_reward)
            accs.append(acc)
            if isinstance(result, dict):
                for key, value in result.items():
                    if key in {"score", "acc"}:
                        continue
                    reward_extra_info[key].append(value)

        group_keys = _resolve_group_keys(data)
        use_conditional_reward, group_has_success = _select_conditional_reward_usage(
            group_keys,
            accs,
            use_rule_reward_when_group_has_success=self.use_rlvr_reward_when_group_has_success,
            success_threshold=self.rule_reward_success_threshold,
            conditioning_reward_mode=self.conditioning_reward_mode,
            low_confidence_tail_percent=self.low_confidence_tail_percent,
        )

        final_rewards = list(rule_rewards)
        reward_by_index: dict[int, float] = {}
        extra_by_index: dict[int, dict[str, Any]] = {}
        conditional_group_members: dict[str, list[int]] = defaultdict(list)

        selected_indices = [idx for idx, use_conditional in enumerate(use_conditional_reward) if use_conditional]
        if selected_indices:
            gt_texts = [self._get_ground_truth_text(data[idx]) for idx in selected_indices]
            if any(gt_text is None or len(gt_text) == 0 for gt_text in gt_texts):
                raise ValueError("conditional_logprob reward requires non-empty ground truth for every selected sample.")

            selected_row_tensor = torch.tensor(selected_indices, device=prompt_ids.device, dtype=torch.long)
            selected_prompt_ids = prompt_ids.index_select(0, selected_row_tensor)
            selected_prompt_attn = prompt_attn.index_select(0, selected_row_tensor)
            selected_prompt_pos = prompt_pos.index_select(0, selected_row_tensor)
            selected_prompt_ids, selected_prompt_attn, selected_prompt_pos, prompt_valid_lengths = _left_pad_prompt_tensors(
                selected_prompt_ids,
                selected_prompt_attn,
                selected_prompt_pos,
                pad_token_id=pad_token_id,
            )

            gt_token_lists: list[torch.Tensor] = []
            for gt_text in gt_texts:
                gt_tokens = self.tokenizer(str(gt_text), return_tensors="pt", add_special_tokens=False)["input_ids"].view(-1)
                if self.max_gt_len is not None:
                    gt_tokens = gt_tokens[: self.max_gt_len]
                gt_token_lists.append(gt_tokens.to(prompt_ids.device))

            valid_row_indices: list[int] = []
            valid_sample_indices: list[int] = []
            valid_group_keys: list[Any] = []
            conditioned_token_lists: list[torch.Tensor] = []
            effective_gt_token_lists: list[torch.Tensor] = []

            for row_idx, sample_idx in enumerate(selected_indices):
                group_key = str(group_keys[sample_idx])
                conditional_group_members[group_key].append(sample_idx)

                sample_info: dict[str, Any] = {
                    "conditioning_invalid_reason": None,
                    "conditioned_prefix_truncated": False,
                    "conditioned_prefix_token_count": 0,
                    "effective_gt_token_count": 0,
                    "conditioning_total_length_budget": self.max_conditioning_total_length,
                    "used_prompt_group_floor_reward": None,
                    "all_group_conditioning_invalid": None,
                }

                conditioned_text, has_think_end = _extract_conditioning_response_prefix(
                    response_texts[sample_idx],
                    truncate_at_last_think=self.truncate_conditioning_response_at_last_think,
                    think_end_tag=self.think_end_tag,
                )
                if not has_think_end:
                    sample_info["conditioning_invalid_reason"] = "missing_think_end"
                    extra_by_index[sample_idx] = sample_info
                    continue

                if conditioned_text == response_texts[sample_idx]:
                    conditioned_tokens_full = valid_response_ids[sample_idx].to(prompt_ids.device)
                else:
                    conditioned_tokens_full = self.tokenizer(
                        conditioned_text,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )["input_ids"].view(-1).to(prompt_ids.device)

                gt_tokens = gt_token_lists[row_idx]
                conditioned_tokens = conditioned_tokens_full

                if self.max_conditioning_total_length is not None:
                    gt_budget = int(self.max_conditioning_total_length) - int(prompt_valid_lengths[row_idx])
                    if gt_budget <= 0:
                        sample_info["conditioning_invalid_reason"] = "no_room_for_ground_truth"
                        extra_by_index[sample_idx] = sample_info
                        continue

                    if gt_tokens.numel() > gt_budget:
                        gt_tokens = gt_tokens[:gt_budget]
                    if gt_tokens.numel() <= 0:
                        sample_info["conditioning_invalid_reason"] = "empty_ground_truth_after_budget"
                        extra_by_index[sample_idx] = sample_info
                        continue

                    prefix_budget = max(
                        0,
                        int(self.max_conditioning_total_length) - int(prompt_valid_lengths[row_idx]) - int(gt_tokens.numel()),
                    )
                    if conditioned_tokens.numel() > prefix_budget:
                        conditioned_tokens = conditioned_tokens[:prefix_budget]
                        sample_info["conditioned_prefix_truncated"] = True

                sample_info["conditioned_prefix_token_count"] = int(conditioned_tokens.numel())
                sample_info["effective_gt_token_count"] = int(gt_tokens.numel())
                extra_by_index[sample_idx] = sample_info

                valid_row_indices.append(row_idx)
                valid_sample_indices.append(sample_idx)
                valid_group_keys.append(group_key)
                conditioned_token_lists.append(conditioned_tokens)
                effective_gt_token_lists.append(gt_tokens)

            if valid_sample_indices:
                valid_row_tensor = torch.tensor(valid_row_indices, device=prompt_ids.device, dtype=torch.long)
                valid_prompt_ids = selected_prompt_ids.index_select(0, valid_row_tensor)
                valid_prompt_attn = selected_prompt_attn.index_select(0, valid_row_tensor)
                valid_prompt_pos = selected_prompt_pos.index_select(0, valid_row_tensor)
                valid_uid_values = [group_keys[idx] for idx in valid_sample_indices]

                need_prompt_only = self.conditioning_reward_mode != "mean_logprob"
                prompt_only_log_probs = None
                if need_prompt_only:
                    prompt_only_batch, prompt_only_gt_mask = self._build_logprob_batch(
                        valid_prompt_ids,
                        valid_prompt_attn,
                        valid_prompt_pos,
                        effective_gt_token_lists,
                        pad_token_id=pad_token_id,
                        uid_values=valid_uid_values,
                    )
                    prompt_only_output = self._compute_log_prob(actor_wg, prompt_only_batch)
                    prompt_only_log_probs = prompt_only_output.batch["old_log_probs"]

                conditioned_topk_k = self.conditioned_token_topk if self.conditioning_reward_mode == "low_confidence_token_topk_recall" else 0
                calculate_token_mrr = self.conditioning_reward_mode == "low_confidence_token_mrr"
                conditioned_batch, conditioned_gt_mask = self._build_logprob_batch(
                    valid_prompt_ids,
                    valid_prompt_attn,
                    valid_prompt_pos,
                    effective_gt_token_lists,
                    conditioned_token_lists=conditioned_token_lists,
                    pad_token_id=pad_token_id,
                    topk_k=conditioned_topk_k,
                    calculate_token_mrr=calculate_token_mrr,
                    uid_values=valid_uid_values,
                )
                conditioned_output = self._compute_log_prob(actor_wg, conditioned_batch)
                conditioned_log_probs = conditioned_output.batch["old_log_probs"]
                conditioned_topk_ids = conditioned_output.batch.get("topk_token_ids", None)
                conditioned_token_mrr = conditioned_output.batch.get("token_reciprocal_ranks", None)

                masked_focus_values = data.non_tensor_batch.get("masked_solution_focus_token_indices", None)

                for local_idx, sample_idx in enumerate(valid_sample_indices):
                    valid_gt_count = int(conditioned_gt_mask[local_idx].sum().item())
                    conditioned_valid = conditioned_log_probs[local_idx, :valid_gt_count].to(torch.float32)

                    cond_logprob = (
                        float(conditioned_valid.sum().item())
                        if self.reduction == "sum"
                        else float(conditioned_valid.mean().item())
                    )
                    sample_info = extra_by_index[sample_idx]
                    sample_info["cond_logprob"] = cond_logprob

                    focus_indices: list[int] = []
                    if prompt_only_log_probs is not None:
                        if self.align_conditioning_focus_with_prompt_mask and masked_focus_values is not None:
                            provided_positions = masked_focus_values[sample_idx]
                            if provided_positions is not None:
                                focus_indices = [
                                    int(pos)
                                    for pos in provided_positions
                                    if 0 <= int(pos) < valid_gt_count
                                ]
                        if not focus_indices:
                            focus_indices = _select_low_confidence_token_indices(
                                prompt_only_log_probs[local_idx],
                                valid_gt_count,
                                tail_percent=self.low_confidence_tail_percent,
                                min_tokens=self.low_confidence_min_tokens,
                            )
                        sample_info["low_confidence_token_indices"] = list(focus_indices)
                        sample_info["focus_token_indices"] = list(focus_indices)
                    else:
                        sample_info["low_confidence_token_indices"] = None
                        sample_info["focus_token_indices"] = None

                    if self.conditioning_reward_mode == "mean_logprob":
                        conditional_reward = cond_logprob
                    elif self.conditioning_reward_mode == "low_confidence_recovery_ratio":
                        prompt_only_valid = prompt_only_log_probs[local_idx, :valid_gt_count].to(torch.float32)
                        logprob_deltas = conditioned_valid - prompt_only_valid
                        focus_delta_mean = float(logprob_deltas[focus_indices].mean().item())
                        overall_delta_mean = float(logprob_deltas.mean().item())
                        tail_fraction = _resolve_tail_fraction(self.low_confidence_tail_percent)
                        if tail_fraction >= 1.0:
                            conditional_reward = min(
                                math.exp(focus_delta_mean),
                                self.max_logprob_reward,
                            )
                        else:
                            denominator = max(math.exp(overall_delta_mean), self.min_overall_improvement_ratio)
                            conditional_reward = min(
                                math.exp(focus_delta_mean) / denominator,
                                self.max_logprob_reward,
                            )
                        sample_info["prompt_only_cond_logprob"] = (
                            float(prompt_only_valid.sum().item())
                            if self.reduction == "sum"
                            else float(prompt_only_valid.mean().item())
                        )
                        sample_info["focus_logprob_improvement_mean"] = focus_delta_mean
                        sample_info["overall_logprob_improvement_mean"] = overall_delta_mean
                        sample_info["low_confidence_recovery_ratio"] = conditional_reward
                    elif self.conditioning_reward_mode == "low_confidence_token_topk_recall":
                        if conditioned_topk_ids is None:
                            raise ValueError("Expected conditioned top-k token ids for low_confidence_token_topk_recall.")
                        gt_ids = conditioned_batch.batch["responses"][local_idx, :valid_gt_count]
                        topk_ids = conditioned_topk_ids[local_idx, :valid_gt_count]
                        hits = [
                            bool((topk_ids[int(pos)] == gt_ids[int(pos)]).any().item())
                            for pos in focus_indices
                        ]
                        conditional_reward = float(sum(hits) / max(len(hits), 1))
                        sample_info["focus_token_topk_hits"] = list(hits)
                        sample_info["focus_token_topk_hit_fraction"] = conditional_reward
                        sample_info["low_confidence_token_topk_recall"] = conditional_reward
                    elif self.conditioning_reward_mode == "low_confidence_token_mrr":
                        if conditioned_token_mrr is None:
                            raise ValueError("Expected conditioned reciprocal ranks for low_confidence_token_mrr.")
                        focus_rrs = [
                            float(conditioned_token_mrr[local_idx, int(pos)].item())
                            for pos in focus_indices
                        ]
                        conditional_reward = float(sum(focus_rrs) / max(len(focus_rrs), 1))
                        sample_info["focus_token_reciprocal_ranks"] = list(focus_rrs)
                        sample_info["focus_token_mean_reciprocal_rank"] = conditional_reward
                        sample_info["low_confidence_token_mrr"] = conditional_reward
                    else:
                        raise AssertionError(f"Unhandled conditioning_reward_mode {self.conditioning_reward_mode!r}")

                    reward_by_index[sample_idx] = conditional_reward

        for group_key, member_indices in conditional_group_members.items():
            valid_member_rewards = [reward_by_index[idx] for idx in member_indices if idx in reward_by_index]
            if valid_member_rewards:
                floor_reward = min(valid_member_rewards)
                for sample_idx in member_indices:
                    sample_info = extra_by_index.setdefault(
                        sample_idx,
                        {
                            "conditioning_invalid_reason": None,
                            "used_prompt_group_floor_reward": None,
                            "all_group_conditioning_invalid": None,
                        },
                    )
                    if sample_idx in reward_by_index:
                        sample_info["used_prompt_group_floor_reward"] = False
                        sample_info["all_group_conditioning_invalid"] = False
                    else:
                        reward_by_index[sample_idx] = floor_reward
                        sample_info["used_prompt_group_floor_reward"] = True
                        sample_info["all_group_conditioning_invalid"] = False
            else:
                for sample_idx in member_indices:
                    sample_info = extra_by_index.setdefault(
                        sample_idx,
                        {
                            "conditioning_invalid_reason": None,
                            "used_prompt_group_floor_reward": None,
                            "all_group_conditioning_invalid": None,
                        },
                    )
                    reward_by_index[sample_idx] = 0.0
                    sample_info["used_prompt_group_floor_reward"] = False
                    sample_info["all_group_conditioning_invalid"] = True

        for sample_idx, conditional_reward in reward_by_index.items():
            final_rewards[sample_idx] = conditional_reward

        reward_tensor = torch.zeros_like(responses, dtype=torch.float32)
        for row_idx, reward in enumerate(final_rewards):
            response_len = int(valid_response_lengths[row_idx].item())
            if response_len > 0:
                reward_tensor[row_idx, response_len - 1] = float(reward)

        data.batch["acc"] = torch.tensor(accs, dtype=torch.float32, device=prompt_ids.device)

        reward_extra_info["score"] = list(final_rewards)
        reward_extra_info["acc"] = list(accs)
        reward_extra_info["rule_reward"] = list(rule_rewards)
        reward_extra_info["used_conditional_logprob"] = list(use_conditional_reward)
        reward_extra_info["group_has_success"] = list(group_has_success)

        for key in (
            "cond_logprob",
            "prompt_only_cond_logprob",
            "low_confidence_token_indices",
            "focus_token_indices",
            "focus_logprob_improvement_mean",
            "overall_logprob_improvement_mean",
            "low_confidence_recovery_ratio",
            "focus_token_topk_hits",
            "focus_token_topk_hit_fraction",
            "low_confidence_token_topk_recall",
            "focus_token_reciprocal_ranks",
            "focus_token_mean_reciprocal_rank",
            "low_confidence_token_mrr",
            "conditioning_invalid_reason",
            "conditioned_prefix_truncated",
            "conditioned_prefix_token_count",
            "effective_gt_token_count",
            "conditioning_total_length_budget",
            "used_prompt_group_floor_reward",
            "all_group_conditioning_invalid",
        ):
            reward_extra_info[key] = [extra_by_index.get(idx, {}).get(key, None) for idx in range(len(data))]

        if self.num_examine > 0:
            already_printed: dict[str, int] = {}
            data_sources = list(data.non_tensor_batch[self.reward_fn_key])
            for idx in range(len(data)):
                data_source = str(data_sources[idx])
                if already_printed.get(data_source, 0) >= self.num_examine:
                    continue
                print("[prompt]", self.tokenizer.decode(prompt_ids[idx], skip_special_tokens=True))
                print("[response]", response_texts[idx])
                print("[ground_truth_response]", self._get_ground_truth_text(data[idx]))
                print("[rule_reward]", rule_rewards[idx])
                print("[final_reward]", final_rewards[idx])
                print("[used_conditional_logprob]", use_conditional_reward[idx])
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
