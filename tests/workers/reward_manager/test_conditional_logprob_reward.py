from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.workers.reward_manager.conditional import (
    ConditionalLogProbRewardManager,
    _extract_conditioning_response_prefix,
    _select_group_logprob_usage,
)


class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self._vocab = {"<pad>": 0, "<eos>": 1}
        self._inv_vocab = {0: "<pad>", 1: "<eos>"}

    def _encode(self, text: str) -> list[int]:
        if not text:
            return []

        token_ids: list[int] = []
        for token in text.split():
            if token not in self._vocab:
                token_id = len(self._vocab)
                self._vocab[token] = token_id
                self._inv_vocab[token_id] = token
            token_ids.append(self._vocab[token])
        return token_ids

    def __call__(self, text: str, return_tensors: str = "pt", add_special_tokens: bool = False):
        del return_tensors, add_special_tokens
        return {"input_ids": torch.tensor([self._encode(text)], dtype=torch.long)}

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        tokens: list[str] = []
        for token_id in token_ids.tolist():
            if skip_special_tokens and token_id in {self.pad_token_id, self.eos_token_id}:
                continue
            tokens.append(self._inv_vocab[token_id])
        return " ".join(tokens)


class DummyActorWG:
    def __init__(self, *, prompt_only_prefix_len: int):
        self.prompt_only_prefix_len = prompt_only_prefix_len
        self.world_size = 1

    def compute_log_prob(self, batch: DataProto) -> DataProto:
        prefix_len = batch.batch["prompts"].shape[1]
        attn = batch.batch["attention_mask"]
        prompt_tokens = batch.batch["prompts"][:, -1]
        gt_len = batch.batch["responses"].shape[1]
        out = torch.zeros((len(batch), gt_len), dtype=torch.float32)
        topk_k = int(batch.meta_info.get("topk_token_ids_k", 0) or 0)
        calculate_token_mrr = bool(batch.meta_info.get("calculate_token_mrr", False))
        topk_token_ids = None
        token_reciprocal_ranks = None
        if topk_k > 0:
            topk_token_ids = torch.zeros((len(batch), gt_len, topk_k), dtype=torch.long)
        if calculate_token_mrr:
            token_reciprocal_ranks = torch.zeros((len(batch), gt_len), dtype=torch.float32)

        for row_idx in range(len(batch)):
            valid_prefix_len = int(attn[row_idx, :prefix_len].sum().item())
            prompt_token = int(prompt_tokens[row_idx].item())
            gt_tokens = batch.batch["responses"][row_idx]

            if prompt_token == 2:
                out[row_idx] = torch.tensor([-2.0, -2.0, -2.0, -2.0], dtype=torch.float32)
            elif valid_prefix_len == self.prompt_only_prefix_len:
                out[row_idx] = torch.tensor([-4.0, -1.0, -1.0, -1.0], dtype=torch.float32)
            else:
                out[row_idx] = torch.tensor([-3.8, -0.5, -0.5, -0.5], dtype=torch.float32)

            if topk_token_ids is not None:
                if valid_prefix_len == self.prompt_only_prefix_len:
                    topk_token_ids[row_idx].fill_(0)
                elif prompt_token == 2:
                    topk_token_ids[row_idx].fill_(999)
                    topk_token_ids[row_idx, :, 0] = gt_tokens
                else:
                    topk_token_ids[row_idx].fill_(999)

            if token_reciprocal_ranks is not None:
                if valid_prefix_len == self.prompt_only_prefix_len:
                    token_reciprocal_ranks[row_idx].fill_(0.25)
                elif prompt_token == 2:
                    token_reciprocal_ranks[row_idx].fill_(1.0)
                else:
                    token_reciprocal_ranks[row_idx].fill_(0.5)

        tensors = {"old_log_probs": out}
        if topk_token_ids is not None:
            tensors["topk_token_ids"] = topk_token_ids
        if token_reciprocal_ranks is not None:
            tensors["token_reciprocal_ranks"] = token_reciprocal_ranks
        return DataProto.from_dict(tensors=tensors)


def _build_test_batch(
    tokenizer: DummyTokenizer,
    *,
    response_texts: list[str] | None = None,
    uids: list[str] | None = None,
) -> DataProto:
    prompt_a = tokenizer._encode("prompt_a")
    prompt_b = tokenizer._encode("prompt_b")

    if response_texts is None:
        response_texts = [
            "good </think> correct",
            "bad </think> wrong",
            "alpha </think> beta",
            "gamma </think> delta",
        ]
    if uids is None:
        uids = ["u1", "u1", "u2", "u2"]

    response_ids = [tokenizer._encode(text) for text in response_texts]
    max_response_len = max(len(ids) for ids in response_ids)
    total_len = 1 + max_response_len

    prompts = torch.tensor([prompt_a, prompt_a, prompt_b, prompt_b], dtype=torch.long)
    responses = torch.full((4, max_response_len), tokenizer.pad_token_id, dtype=torch.long)
    input_ids = torch.full((4, total_len), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((4, total_len), dtype=torch.long)
    position_ids = torch.arange(total_len, dtype=torch.long).unsqueeze(0).expand(4, -1).clone()

    for row_idx, ids in enumerate(response_ids):
        responses[row_idx, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        input_ids[row_idx, 0] = prompts[row_idx, 0]
        input_ids[row_idx, 1 : 1 + len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row_idx, 0] = 1
        attention_mask[row_idx, 1 : 1 + len(ids)] = 1

    reward_model = np.array(
        [
            {"ground_truth": "gt1 gt2 gt3 gt4"},
            {"ground_truth": "gt1 gt2 gt3 gt4"},
            {"ground_truth": "gt1 gt2 gt3 gt4"},
            {"ground_truth": "gt1 gt2 gt3 gt4"},
        ],
        dtype=object,
    )
    non_tensors = {
        "reward_model": reward_model,
        "data_source": np.array(["math", "math", "math", "math"], dtype=object),
        "uid": np.array(uids, dtype=object),
        "extra_info": np.array([{}, {}, {}, {}], dtype=object),
    }
    tensors = {
        "prompts": prompts,
        "responses": responses,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def _fake_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    del data_source, ground_truth, extra_info
    score = 1.0 if "correct" in solution_str else 0.0
    return {"score": score, "acc": score}


def test_extract_conditioning_response_prefix_uses_last_think_tag():
    truncated, has_think_end = _extract_conditioning_response_prefix(
        "a </think> b </think> c",
        truncate_at_last_think=True,
        think_end_tag="</think>",
    )
    assert truncated == "a </think> b </think>"
    assert has_think_end is True


def test_select_group_logprob_usage_prefers_rule_reward_after_any_success():
    use_logprob, has_success = _select_group_logprob_usage(
        ["u1", "u1", "u2", "u2"],
        [1.0, 0.0, 0.0, 0.0],
        use_rule_reward_when_group_has_success=True,
        success_threshold=0.5,
    )
    assert use_logprob == [False, False, True, True]
    assert has_success == [True, True, False, False]


def test_recovery_ratio_mixes_rule_reward_and_conditional_fallback():
    tokenizer = DummyTokenizer()
    batch = _build_test_batch(tokenizer)
    actor_wg = DummyActorWG(prompt_only_prefix_len=1)

    manager = ConditionalLogProbRewardManager(
        tokenizer=tokenizer,
        compute_score=_fake_compute_score,
        conditioning_reward_mode="low_confidence_recovery_ratio",
        low_confidence_tail_percent=25.0,
    )

    result = manager(batch, return_dict=True, actor_wg=actor_wg)
    reward_tensor = result["reward_tensor"]
    reward_extra = result["reward_extra_info"]

    expected_conditional_reward = math.exp(0.2) / math.exp(0.425)

    assert reward_tensor[0, 2].item() == 1.0
    assert reward_tensor[1, 2].item() == 0.0
    assert reward_tensor[2, 2].item() == pytest.approx(expected_conditional_reward, rel=1e-6)
    assert reward_tensor[3, 2].item() == pytest.approx(expected_conditional_reward, rel=1e-6)
    assert reward_extra["used_conditional_logprob"] == [False, False, True, True]
    assert reward_extra["group_has_success"] == [True, True, False, False]
    assert reward_extra["low_confidence_token_indices"] == [None, None, [0], [0]]
    assert reward_extra["score"][0] == 1.0
    assert reward_extra["score"][2] == pytest.approx(expected_conditional_reward, rel=1e-6)


def test_recovery_ratio_with_all_tokens_uses_unnormalized_focus_improvement():
    tokenizer = DummyTokenizer()
    batch = _build_test_batch(tokenizer)
    actor_wg = DummyActorWG(prompt_only_prefix_len=1)

    manager = ConditionalLogProbRewardManager(
        tokenizer=tokenizer,
        compute_score=_fake_compute_score,
        conditioning_reward_mode="low_confidence_recovery_ratio",
        low_confidence_tail_percent=1.0,
        use_rlvr_reward_when_group_has_success=False,
    )

    result = manager(batch, return_dict=True, actor_wg=actor_wg)
    reward_tensor = result["reward_tensor"]
    reward_extra = result["reward_extra_info"]

    expected_prompt_a_reward = 1.0
    expected_prompt_b_reward = math.exp(0.425)

    assert reward_tensor[:, 2].tolist() == pytest.approx(
        [expected_prompt_a_reward, expected_prompt_a_reward, expected_prompt_b_reward, expected_prompt_b_reward],
        rel=1e-6,
    )
    assert reward_extra["used_conditional_logprob"] == [True, True, True, True]
    assert reward_extra["focus_token_indices"] == [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
    assert reward_extra["score"][2] == pytest.approx(expected_prompt_b_reward, rel=1e-6)


def test_low_confidence_token_topk_recall_reward_uses_focus_token_hits():
    tokenizer = DummyTokenizer()
    batch = _build_test_batch(tokenizer)
    actor_wg = DummyActorWG(prompt_only_prefix_len=1)

    manager = ConditionalLogProbRewardManager(
        tokenizer=tokenizer,
        compute_score=_fake_compute_score,
        conditioning_reward_mode="low_confidence_token_topk_recall",
        low_confidence_tail_percent=25.0,
        conditioned_token_topk=3,
    )

    result = manager(batch, return_dict=True, actor_wg=actor_wg)
    reward_tensor = result["reward_tensor"]
    reward_extra = result["reward_extra_info"]

    assert reward_tensor[:, 2].tolist() == pytest.approx([1.0, 1.0, 0.0, 0.0], rel=1e-6)
    assert reward_extra["used_conditional_logprob"] == [True, True, True, True]
    assert reward_extra["focus_token_indices"] == [[0], [0], [0], [0]]
    assert reward_extra["focus_token_topk_hit_fraction"] == pytest.approx([1.0, 1.0, 0.0, 0.0], rel=1e-6)


def test_low_confidence_token_mrr_reward_uses_focus_token_reciprocal_ranks():
    tokenizer = DummyTokenizer()
    batch = _build_test_batch(tokenizer)
    actor_wg = DummyActorWG(prompt_only_prefix_len=1)

    manager = ConditionalLogProbRewardManager(
        tokenizer=tokenizer,
        compute_score=_fake_compute_score,
        conditioning_reward_mode="low_confidence_token_mrr",
        low_confidence_tail_percent=25.0,
    )

    result = manager(batch, return_dict=True, actor_wg=actor_wg)
    reward_tensor = result["reward_tensor"]
    reward_extra = result["reward_extra_info"]

    assert reward_tensor[:, 2].tolist() == pytest.approx([1.0, 1.0, 0.5, 0.5], rel=1e-6)
    assert reward_extra["focus_token_indices"] == [[0], [0], [0], [0]]
    assert reward_extra["focus_token_mean_reciprocal_rank"] == pytest.approx([1.0, 1.0, 0.5, 0.5], rel=1e-6)
    assert reward_extra["focus_token_reciprocal_ranks"] == [[1.0], [1.0], [0.5], [0.5]]


def test_missing_think_uses_prompt_group_floor_reward():
    tokenizer = DummyTokenizer()
    batch = _build_test_batch(
        tokenizer,
        response_texts=[
            "good </think> correct",
            "bad </think> wrong",
            "alpha </think> beta",
            "gamma delta",
        ],
    )
    actor_wg = DummyActorWG(prompt_only_prefix_len=1)

    manager = ConditionalLogProbRewardManager(
        tokenizer=tokenizer,
        compute_score=_fake_compute_score,
        conditioning_reward_mode="mean_logprob",
    )

    result = manager(batch, return_dict=True, actor_wg=actor_wg)
    reward_tensor = result["reward_tensor"]
    reward_extra = result["reward_extra_info"]

    sample_2_reward = reward_tensor[2, 2].item()
    sample_3_reward = reward_tensor[3, 1].item()

    assert sample_3_reward == pytest.approx(sample_2_reward, rel=1e-6)
    assert reward_extra["conditioning_invalid_reason"] == [None, None, None, "missing_think_end"]
    assert reward_extra["used_prompt_group_floor_reward"] == [None, None, False, True]
    assert reward_extra["all_group_conditioning_invalid"] == [None, None, False, False]


def test_missing_think_all_invalid_group_gets_zero_reward():
    tokenizer = DummyTokenizer()
    batch = _build_test_batch(
        tokenizer,
        response_texts=[
            "good </think> correct",
            "bad </think> wrong",
            "alpha beta",
            "gamma delta",
        ],
    )
    actor_wg = DummyActorWG(prompt_only_prefix_len=1)

    manager = ConditionalLogProbRewardManager(
        tokenizer=tokenizer,
        compute_score=_fake_compute_score,
        conditioning_reward_mode="mean_logprob",
    )

    result = manager(batch, return_dict=True, actor_wg=actor_wg)
    reward_tensor = result["reward_tensor"]
    reward_extra = result["reward_extra_info"]

    assert reward_tensor[2, 1].item() == 0.0
    assert reward_tensor[3, 1].item() == 0.0
    assert reward_extra["conditioning_invalid_reason"] == [None, None, "missing_think_end", "missing_think_end"]
    assert reward_extra["used_prompt_group_floor_reward"] == [None, None, False, False]
    assert reward_extra["all_group_conditioning_invalid"] == [None, None, True, True]


def test_conditioning_budget_truncates_prefix_to_fit_total_length():
    tokenizer = DummyTokenizer()
    batch = _build_test_batch(tokenizer)
    actor_wg = DummyActorWG(prompt_only_prefix_len=1)

    manager = ConditionalLogProbRewardManager(
        tokenizer=tokenizer,
        compute_score=_fake_compute_score,
        conditioning_reward_mode="mean_logprob",
        use_rlvr_reward_when_group_has_success=False,
        max_conditioning_total_length=6,
    )

    result = manager(batch, return_dict=True, actor_wg=actor_wg)
    reward_extra = result["reward_extra_info"]

    assert reward_extra["conditioned_prefix_truncated"] == [True, True, True, True]
    assert reward_extra["conditioned_prefix_token_count"] == [1, 1, 1, 1]
    assert reward_extra["effective_gt_token_count"] == [4, 4, 4, 4]


def test_conditional_reward_manager_disables_async_reward_loop():
    manager = ConditionalLogProbRewardManager(
        tokenizer=DummyTokenizer(),
        compute_score=_fake_compute_score,
    )
    assert manager.disable_async_reward_loop is True
