from __future__ import annotations

import math

import numpy as np
import torch

from verl import DataProto
from verl.workers.reward_manager.batch import BatchRewardManager


def _assert_close(actual: float, expected: float, rel: float = 1e-6) -> None:
    assert math.isclose(actual, expected, rel_tol=rel, abs_tol=rel), (actual, expected)


def _assert_list_close(actual: list[float], expected: list[float], rel: float = 1e-6) -> None:
    assert len(actual) == len(expected), (actual, expected)
    for actual_item, expected_item in zip(actual, expected, strict=True):
        _assert_close(actual_item, expected_item, rel=rel)


class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self._vocab = {"<pad>": 0, "<eos>": 1}
        self._inv_vocab = {0: "<pad>", 1: "<eos>"}

    def _encode(self, text: str) -> list[int]:
        if not text:
            return []

        token_ids = []
        for token in text.split():
            if token not in self._vocab:
                token_id = len(self._vocab)
                self._vocab[token] = token_id
                self._inv_vocab[token_id] = token
            token_ids.append(self._vocab[token])
        return token_ids

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        tokens = []
        for token_id in token_ids.tolist():
            if skip_special_tokens and token_id in {self.pad_token_id, self.eos_token_id}:
                continue
            tokens.append(self._inv_vocab[token_id])
        return " ".join(tokens)


class DummyActorWG:
    def __init__(self, log_probs_by_first_token: dict[int, list[float]], max_response_len: int):
        self.log_probs_by_first_token = log_probs_by_first_token
        self.max_response_len = max_response_len
        self.world_size = 1

    def compute_log_prob(self, batch: DataProto) -> DataProto:
        old_log_probs = torch.zeros((len(batch), self.max_response_len), dtype=torch.float32)
        for row_idx in range(len(batch)):
            first_token = int(batch.batch["responses"][row_idx, 0].item())
            old_log_probs[row_idx] = torch.tensor(
                self.log_probs_by_first_token[first_token],
                dtype=torch.float32,
            )
        return DataProto.from_dict(tensors={"old_log_probs": old_log_probs})


def _build_test_batch(
    tokenizer: DummyTokenizer,
    *,
    response_texts: list[str],
    uids: list[str],
    prompt_group_ids: list[str] | None = None,
) -> DataProto:
    prompt_ids = [tokenizer._encode("prompt a"), tokenizer._encode("prompt a"), tokenizer._encode("prompt b"), tokenizer._encode("prompt b")]
    response_ids = [tokenizer._encode(text) for text in response_texts]
    max_prompt_len = max(len(ids) for ids in prompt_ids)
    max_response_len = max(len(ids) for ids in response_ids)
    total_len = max_prompt_len + max_response_len

    prompts = torch.full((len(response_ids), max_prompt_len), tokenizer.pad_token_id, dtype=torch.long)
    responses = torch.full((len(response_ids), max_response_len), tokenizer.pad_token_id, dtype=torch.long)
    input_ids = torch.full((len(response_ids), total_len), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(response_ids), total_len), dtype=torch.long)
    position_ids = torch.zeros((len(response_ids), total_len), dtype=torch.long)

    for row_idx, (prompt_row, response_row) in enumerate(zip(prompt_ids, response_ids, strict=True)):
        prompt_len = len(prompt_row)
        response_len = len(response_row)
        prompts[row_idx, -prompt_len:] = torch.tensor(prompt_row, dtype=torch.long)
        responses[row_idx, :response_len] = torch.tensor(response_row, dtype=torch.long)
        input_ids[row_idx, :prompt_len] = torch.tensor(prompt_row, dtype=torch.long)
        input_ids[row_idx, prompt_len : prompt_len + response_len] = torch.tensor(response_row, dtype=torch.long)
        attention_mask[row_idx, : prompt_len + response_len] = 1
        position_ids[row_idx] = torch.arange(total_len, dtype=torch.long)

    reward_model = np.array([{"ground_truth": "unused"} for _ in response_ids], dtype=object)
    extra_info = np.array([{} for _ in response_ids], dtype=object)
    non_tensors = {
        "reward_model": reward_model,
        "data_source": np.array(["math"] * len(response_ids), dtype=object),
        "uid": np.array(uids, dtype=object),
        "extra_info": extra_info,
    }
    if prompt_group_ids is not None:
        non_tensors["prompt_group_id"] = np.array(prompt_group_ids, dtype=object)
    tensors = {
        "prompts": prompts,
        "responses": responses,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def _build_reward_manager(
    tokenizer: DummyTokenizer,
    *,
    acc_by_prefix: dict[str, float],
    uniform_mode: str = "all_success_or_failure",
) -> BatchRewardManager:
    def compute_score(data_sources, solution_strs, ground_truths, extra_infos):
        del data_sources, ground_truths, extra_infos
        return [
            {"score": acc_by_prefix[solution_str.split()[0]], "acc": acc_by_prefix[solution_str.split()[0]]}
            for solution_str in solution_strs
        ]

    return BatchRewardManager(
        tokenizer=tokenizer,
        num_examine=0,
        compute_score=compute_score,
        use_response_logprob_reward_for_uniform_outcome_groups=True,
        uniform_outcome_response_logprob_reward_mode=uniform_mode,
    )


def test_uniform_outcome_group_reward_overrides_all_success_and_all_failure_groups():
    tokenizer = DummyTokenizer()
    response_texts = ["r0 a b", "r1 a b", "r2 a b", "r3 a b c"]
    batch = _build_test_batch(
        tokenizer,
        response_texts=response_texts,
        uids=["u1", "u1", "u2", "u2"],
    )
    encoded_rows = {text.split()[0]: tokenizer._encode(text)[0] for text in response_texts}
    actor_wg = DummyActorWG(
        {
            encoded_rows["r0"]: [-2.0, -4.0, -4.0, 0.0],
            encoded_rows["r1"]: [-3.0, -3.0, -3.0, 0.0],
            encoded_rows["r2"]: [-1.0, -5.0, -1.0, 0.0],
            encoded_rows["r3"]: [-2.0, -2.0, -6.0, -2.0],
        },
        max_response_len=4,
    )
    manager = _build_reward_manager(
        tokenizer,
        acc_by_prefix={"r0": 1.0, "r1": 1.0, "r2": 0.0, "r3": 0.0},
    )

    result = manager(batch, return_dict=True, actor_wg=actor_wg)
    reward_tensor = result["reward_tensor"]
    reward_extra = result["reward_extra_info"]

    expected = [
        math.exp(-4.0 - (-10.0 / 3.0)),
        1.0,
        math.exp(-1.0 - (-7.0 / 3.0)),
        math.exp(-2.0 - (-3.0)),
    ]

    _assert_close(reward_tensor[0, 2].item(), expected[0])
    _assert_close(reward_tensor[1, 2].item(), expected[1])
    _assert_close(reward_tensor[2, 2].item(), expected[2])
    _assert_close(reward_tensor[3, 3].item(), expected[3])
    assert reward_extra["acc"] == [1.0, 1.0, 0.0, 0.0]
    assert reward_extra["rule_reward"] == [1.0, 1.0, 0.0, 0.0]
    assert reward_extra["group_all_success"] == [True, True, False, False]
    assert reward_extra["group_all_failure"] == [False, False, True, True]
    assert reward_extra["used_uniform_outcome_response_logprob_reward"] == [True, True, True, True]
    _assert_list_close(reward_extra["uniform_outcome_response_logprob_reward"], expected)
    _assert_list_close(reward_extra["response_mean_logprob"], [-10.0 / 3.0, -3.0, -7.0 / 3.0, -3.0])
    _assert_list_close(reward_extra["response_median_logprob"], [-4.0, -3.0, -1.0, -2.0])
    assert reward_extra["response_logprob_source"] == ["actor_recompute"] * 4
    _assert_list_close(batch.batch["acc"].tolist(), [1.0, 1.0, 0.0, 0.0])


def test_uniform_outcome_group_reward_only_applies_to_homogeneous_groups():
    tokenizer = DummyTokenizer()
    response_texts = ["r0 a b", "r1 a b", "r2 a b", "r3 a b c"]
    batch = _build_test_batch(
        tokenizer,
        response_texts=response_texts,
        uids=["u1", "u1", "u2", "u2"],
    )
    encoded_rows = {text.split()[0]: tokenizer._encode(text)[0] for text in response_texts}
    actor_wg = DummyActorWG(
        {
            encoded_rows["r2"]: [-1.0, -5.0, -1.0, 0.0],
            encoded_rows["r3"]: [-2.0, -2.0, -6.0, -2.0],
        },
        max_response_len=4,
    )
    manager = _build_reward_manager(
        tokenizer,
        acc_by_prefix={"r0": 1.0, "r1": 0.0, "r2": 0.0, "r3": 0.0},
    )

    result = manager(batch, return_dict=True, actor_wg=actor_wg)
    reward_tensor = result["reward_tensor"]
    reward_extra = result["reward_extra_info"]

    expected_row2 = math.exp(-1.0 - (-7.0 / 3.0))
    expected_row3 = math.exp(-2.0 - (-3.0))

    _assert_close(reward_tensor[0, 2].item(), 1.0)
    _assert_close(reward_tensor[1, 2].item(), 0.0)
    _assert_close(reward_tensor[2, 2].item(), expected_row2)
    _assert_close(reward_tensor[3, 3].item(), expected_row3)
    assert reward_extra["group_all_success"] == [False, False, False, False]
    assert reward_extra["group_all_failure"] == [False, False, True, True]
    assert reward_extra["used_uniform_outcome_response_logprob_reward"] == [False, False, True, True]
    assert reward_extra["uniform_outcome_response_logprob_reward"][:2] == [0.0, 0.0]
    _assert_list_close(reward_extra["uniform_outcome_response_logprob_reward"][2:], [expected_row2, expected_row3])
    assert reward_extra["response_mean_logprob"][:2] == [None, None]
    assert reward_extra["response_median_logprob"][:2] == [None, None]
    assert reward_extra["response_logprob_source"][:2] == [None, None]
    assert reward_extra["response_logprob_source"][2:] == ["actor_recompute", "actor_recompute"]


def test_uniform_outcome_group_reward_can_target_only_all_failure_groups():
    tokenizer = DummyTokenizer()
    response_texts = ["r0 a b", "r1 a b", "r2 a b", "r3 a b c"]
    batch = _build_test_batch(
        tokenizer,
        response_texts=response_texts,
        uids=["u1", "u1", "u2", "u2"],
    )
    encoded_rows = {text.split()[0]: tokenizer._encode(text)[0] for text in response_texts}
    actor_wg = DummyActorWG(
        {
            encoded_rows["r2"]: [-1.0, -5.0, -1.0, 0.0],
            encoded_rows["r3"]: [-2.0, -2.0, -6.0, -2.0],
        },
        max_response_len=4,
    )
    manager = _build_reward_manager(
        tokenizer,
        acc_by_prefix={"r0": 1.0, "r1": 1.0, "r2": 0.0, "r3": 0.0},
        uniform_mode="all_failure",
    )

    result = manager(batch, return_dict=True, actor_wg=actor_wg)
    reward_tensor = result["reward_tensor"]
    reward_extra = result["reward_extra_info"]

    expected_row2 = math.exp(-1.0 - (-7.0 / 3.0))
    expected_row3 = math.exp(-2.0 - (-3.0))

    _assert_close(reward_tensor[0, 2].item(), 1.0)
    _assert_close(reward_tensor[1, 2].item(), 1.0)
    _assert_close(reward_tensor[2, 2].item(), expected_row2)
    _assert_close(reward_tensor[3, 3].item(), expected_row3)
    assert reward_extra["group_all_success"] == [True, True, False, False]
    assert reward_extra["group_all_failure"] == [False, False, True, True]
    assert reward_extra["used_uniform_outcome_response_logprob_reward"] == [False, False, True, True]
    assert reward_extra["uniform_outcome_response_logprob_reward"][:2] == [0.0, 0.0]
    _assert_list_close(reward_extra["uniform_outcome_response_logprob_reward"][2:], [expected_row2, expected_row3])
    assert reward_extra["response_mean_logprob"][:2] == [None, None]
    assert reward_extra["response_median_logprob"][:2] == [None, None]
    assert reward_extra["response_logprob_source"][:2] == [None, None]
    assert reward_extra["response_logprob_source"][2:] == ["actor_recompute", "actor_recompute"]


def test_uniform_outcome_group_reward_prefers_prompt_group_id_over_unique_uids():
    tokenizer = DummyTokenizer()
    response_texts = ["r0 a b", "r1 a b", "r2 a b", "r3 a b c"]
    batch = _build_test_batch(
        tokenizer,
        response_texts=response_texts,
        uids=["uid0", "uid1", "uid2", "uid3"],
        prompt_group_ids=["g1", "g1", "g2", "g2"],
    )
    encoded_rows = {text.split()[0]: tokenizer._encode(text)[0] for text in response_texts}
    actor_wg = DummyActorWG(
        {
            encoded_rows["r0"]: [-2.0, -4.0, -4.0, 0.0],
            encoded_rows["r1"]: [-3.0, -3.0, -3.0, 0.0],
            encoded_rows["r2"]: [-1.0, -5.0, -1.0, 0.0],
            encoded_rows["r3"]: [-2.0, -2.0, -6.0, -2.0],
        },
        max_response_len=4,
    )
    manager = _build_reward_manager(
        tokenizer,
        acc_by_prefix={"r0": 1.0, "r1": 1.0, "r2": 0.0, "r3": 0.0},
    )

    result = manager(batch, return_dict=True, actor_wg=actor_wg)
    reward_extra = result["reward_extra_info"]

    assert reward_extra["group_all_success"] == [True, True, False, False]
    assert reward_extra["group_all_failure"] == [False, False, True, True]
    assert reward_extra["used_uniform_outcome_response_logprob_reward"] == [True, True, True, True]


def test_uniform_outcome_group_reward_falls_back_to_prompt_tokens_when_ids_are_unique():
    tokenizer = DummyTokenizer()
    response_texts = ["r0 a b", "r1 a b", "r2 a b", "r3 a b c"]
    batch = _build_test_batch(
        tokenizer,
        response_texts=response_texts,
        uids=["uid0", "uid1", "uid2", "uid3"],
        prompt_group_ids=["pg0", "pg1", "pg2", "pg3"],
    )
    encoded_rows = {text.split()[0]: tokenizer._encode(text)[0] for text in response_texts}
    actor_wg = DummyActorWG(
        {
            encoded_rows["r0"]: [-2.0, -4.0, -4.0, 0.0],
            encoded_rows["r1"]: [-3.0, -3.0, -3.0, 0.0],
            encoded_rows["r2"]: [-1.0, -5.0, -1.0, 0.0],
            encoded_rows["r3"]: [-2.0, -2.0, -6.0, -2.0],
        },
        max_response_len=4,
    )
    manager = _build_reward_manager(
        tokenizer,
        acc_by_prefix={"r0": 1.0, "r1": 1.0, "r2": 0.0, "r3": 0.0},
    )

    result = manager(batch, return_dict=True, actor_wg=actor_wg)
    reward_extra = result["reward_extra_info"]

    assert reward_extra["group_all_success"] == [True, True, False, False]
    assert reward_extra["group_all_failure"] == [False, False, True, True]
    assert reward_extra["used_uniform_outcome_response_logprob_reward"] == [True, True, True, True]


def test_uniform_outcome_group_reward_ignores_precomputed_rm_scores():
    tokenizer = DummyTokenizer()
    response_texts = ["r0 a b", "r1 a b", "r2 a b", "r3 a b c"]
    batch = _build_test_batch(
        tokenizer,
        response_texts=response_texts,
        uids=["u1", "u1", "u2", "u2"],
    )
    batch.batch["rm_scores"] = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
    batch.batch["rm_scores"][0, 2] = 1.0
    batch.batch["rm_scores"][1, 2] = 1.0
    batch.meta_info["reward_extra_keys"] = ["acc"]
    batch.non_tensor_batch["acc"] = np.array([1.0, 1.0, 0.0, 0.0], dtype=object)

    encoded_rows = {text.split()[0]: tokenizer._encode(text)[0] for text in response_texts}
    actor_wg = DummyActorWG(
        {
            encoded_rows["r0"]: [-2.0, -4.0, -4.0, 0.0],
            encoded_rows["r1"]: [-3.0, -3.0, -3.0, 0.0],
            encoded_rows["r2"]: [-1.0, -5.0, -1.0, 0.0],
            encoded_rows["r3"]: [-2.0, -2.0, -6.0, -2.0],
        },
        max_response_len=4,
    )
    manager = _build_reward_manager(
        tokenizer,
        acc_by_prefix={"r0": 1.0, "r1": 1.0, "r2": 0.0, "r3": 0.0},
    )

    result = manager(batch, return_dict=True, actor_wg=actor_wg)
    reward_tensor = result["reward_tensor"]

    assert manager.disable_async_reward_loop is True
    assert reward_tensor[2, 2].item() > 0.0
    assert reward_tensor[3, 3].item() > 0.0
