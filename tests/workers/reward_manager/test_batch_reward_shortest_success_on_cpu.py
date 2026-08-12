from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    _compute_longest_success_penalty_reward_metrics,
    _compute_shortest_success_reward_metrics,
)
from verl.workers.reward_manager.batch import BatchRewardManager


class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self._vocab = {"<pad>": 0, "<eos>": 1}
        self._inverse = {0: "<pad>", 1: "<eos>"}

    def encode(self, text: str) -> list[int]:
        token_ids = []
        for token in text.split():
            if token not in self._vocab:
                token_id = len(self._vocab)
                self._vocab[token] = token_id
                self._inverse[token_id] = token
            token_ids.append(self._vocab[token])
        return token_ids

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        tokens = []
        for token_id in token_ids.tolist():
            if skip_special_tokens and token_id in {self.pad_token_id, self.eos_token_id}:
                continue
            tokens.append(self._inverse[token_id])
        return " ".join(tokens)


def _response(prefix: str, length: int) -> str:
    assert length >= 1
    return " ".join([prefix] + [f"{prefix}_token_{idx}" for idx in range(1, length)])


def _build_batch(
    tokenizer: DummyTokenizer,
    *,
    response_specs: list[tuple[str, int]],
    group_ids: list[str] | None,
    prompt_lengths: list[int] | None = None,
) -> DataProto:
    response_rows = [tokenizer.encode(_response(prefix, length)) for prefix, length in response_specs]
    if prompt_lengths is None:
        prompt_lengths = [2] * len(response_rows)
    prompt_rows = [tokenizer.encode(" ".join([f"p{idx}"] * length)) for idx, length in enumerate(prompt_lengths)]
    max_prompt_length = max(len(row) for row in prompt_rows)
    max_response_length = max(len(row) for row in response_rows)

    prompts = torch.full((len(response_rows), max_prompt_length), tokenizer.pad_token_id, dtype=torch.long)
    responses = torch.full((len(response_rows), max_response_length), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(response_rows), max_prompt_length + max_response_length), dtype=torch.long)
    response_mask = torch.zeros((len(response_rows), max_response_length), dtype=torch.long)
    for idx, (prompt_row, response_row) in enumerate(zip(prompt_rows, response_rows, strict=True)):
        prompts[idx, -len(prompt_row) :] = torch.tensor(prompt_row, dtype=torch.long)
        responses[idx, : len(response_row)] = torch.tensor(response_row, dtype=torch.long)
        attention_mask[idx, max_prompt_length - len(prompt_row) : max_prompt_length] = 1
        attention_mask[idx, max_prompt_length : max_prompt_length + len(response_row)] = 1
        response_mask[idx, : len(response_row)] = 1

    non_tensors = {
        "reward_model": np.array([{"ground_truth": "unused"} for _ in response_rows], dtype=object),
        "data_source": np.array(["math"] * len(response_rows), dtype=object),
        "extra_info": np.array([{} for _ in response_rows], dtype=object),
        "uid": np.array(group_ids or [f"uid_{idx}" for idx in range(len(response_rows))], dtype=object),
    }
    if group_ids is not None:
        non_tensors["prompt_group_id"] = np.array(group_ids, dtype=object)
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
        },
        non_tensors=non_tensors,
    )


def _manager(
    tokenizer: DummyTokenizer,
    *,
    acc_by_prefix: dict[str, float],
    margin_percent: float = 10.0,
    group_size: int = 4,
    enabled: bool = True,
) -> BatchRewardManager:
    def compute_score(data_sources, solution_strs, ground_truths, extra_infos):
        del data_sources, ground_truths, extra_infos
        return [
            {"score": acc_by_prefix[solution.split()[0]], "acc": acc_by_prefix[solution.split()[0]]}
            for solution in solution_strs
        ]

    config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": group_size}}})
    return BatchRewardManager(
        config=config,
        tokenizer=tokenizer,
        num_examine=0,
        compute_score=compute_score,
        use_shortest_success_reward=enabled,
        shortest_success_margin_percent=margin_percent,
    )


def _longest_penalty_manager(
    tokenizer: DummyTokenizer,
    *,
    acc_by_prefix: dict[str, float],
    margin_percent: float = 50.0,
    group_size: int = 4,
    enabled: bool = True,
) -> BatchRewardManager:
    def compute_score(data_sources, solution_strs, ground_truths, extra_infos):
        del data_sources, ground_truths, extra_infos
        return [
            {"score": acc_by_prefix[solution.split()[0]], "acc": acc_by_prefix[solution.split()[0]]}
            for solution in solution_strs
        ]

    config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": group_size}}})
    return BatchRewardManager(
        config=config,
        tokenizer=tokenizer,
        num_examine=0,
        compute_score=compute_score,
        use_longest_success_penalty_reward=enabled,
        longest_success_no_penalty_margin_percent=margin_percent,
    )


def test_shortest_success_rewards_only_correct_responses_within_inclusive_margin():
    tokenizer = DummyTokenizer()
    specs = [
        ("a0", 10),
        ("a1", 11),
        ("a2", 12),
        ("a3", 5),
        ("b0", 3),
        ("b1", 4),
        ("b2", 5),
        ("b3", 6),
    ]
    batch = _build_batch(
        tokenizer,
        response_specs=specs,
        group_ids=["g1"] * 4 + ["g2"] * 4,
        prompt_lengths=[1, 4, 2, 5, 3, 1, 5, 2],
    )
    manager = _manager(
        tokenizer,
        acc_by_prefix={"a0": 1.0, "a1": 1.0, "a2": 1.0, "a3": 0.0, **{f"b{i}": 0.0 for i in range(4)}},
    )

    result = manager(batch, return_dict=True)
    reward_tensor = result["reward_tensor"]
    extra = result["reward_extra_info"]

    assert reward_tensor.sum().item() == 2.0
    assert reward_tensor[0, 9].item() == 1.0
    assert reward_tensor[1, 10].item() == 1.0
    assert torch.count_nonzero(reward_tensor).item() == 2
    assert extra["acc"] == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert extra["shortest_success_reward"] == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert extra["shortest_success_response_tokens"] == [length for _, length in specs]
    assert extra["shortest_success_group_min_tokens"][:4] == [10] * 4
    assert extra["shortest_success_group_threshold_tokens"][:4] == [11.0] * 4
    assert extra["shortest_success_group_min_tokens"][4:] == [None] * 4
    assert extra["shortest_success_group_has_success"] == [True] * 4 + [False] * 4
    assert extra["shortest_success_reward_token_index"] == [length - 1 for _, length in specs]
    assert extra["shortest_success_reward_tensor_coordinate"] == ["response_only"] * 8
    assert extra["shortest_success_nonzero_reward_token_count"] == [1, 1, 0, 0, 0, 0, 0, 0]
    assert extra["shortest_success_valid_response_nonzero_reward_token_count"] == [1, 1, 0, 0, 0, 0, 0, 0]
    assert extra["shortest_success_pad_nonzero_reward_token_count"] == [0] * 8
    assert extra["shortest_success_reward_tensor_row_sum"] == extra["shortest_success_reward"]
    assert batch.batch["acc"].tolist() == extra["acc"]
    assert manager.disable_async_reward_loop is True


def test_shortest_success_zero_margin_rewards_all_exact_ties_and_is_order_independent():
    tokenizer = DummyTokenizer()
    specs = [("a0", 10), ("b0", 7), ("a1", 10), ("b1", 9), ("a2", 20), ("b2", 7), ("a3", 20), ("b3", 8)]
    groups = ["a", "b", "a", "b", "a", "b", "a", "b"]
    batch = _build_batch(tokenizer, response_specs=specs, group_ids=groups)
    manager = _manager(tokenizer, acc_by_prefix={prefix: 1.0 for prefix, _ in specs}, margin_percent=0.0)

    extra = manager(batch, return_dict=True)["reward_extra_info"]

    assert extra["shortest_success_reward"] == [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    assert extra["shortest_success_group_min_tokens"] == [10, 7, 10, 7, 10, 7, 10, 7]


def test_shortest_success_ignores_precomputed_rm_scores_but_default_path_does_not():
    tokenizer = DummyTokenizer()
    batch = _build_batch(
        tokenizer,
        response_specs=[("r0", 2), ("r1", 3), ("r2", 4), ("r3", 5)],
        group_ids=["g"] * 4,
    )
    batch.batch["rm_scores"] = torch.full_like(batch.batch["responses"], 7.0, dtype=torch.float32)

    enabled_manager = _manager(
        tokenizer,
        acc_by_prefix={"r0": 1.0, "r1": 1.0, "r2": 0.0, "r3": 0.0},
    )
    enabled = enabled_manager(batch, return_dict=True)["reward_tensor"]
    assert enabled.sum().item() == 1.0

    disabled_manager = _manager(
        tokenizer,
        acc_by_prefix={"r0": 1.0, "r1": 1.0, "r2": 0.0, "r3": 0.0},
        enabled=False,
    )
    disabled = disabled_manager(batch, return_dict=True)["reward_tensor"]
    assert torch.equal(disabled, batch.batch["rm_scores"])


def test_shortest_success_fails_closed_on_invalid_configuration_groups_and_masks():
    tokenizer = DummyTokenizer()
    with pytest.raises(ValueError, match="finite and non-negative"):
        _manager(tokenizer, acc_by_prefix={}, margin_percent=-1.0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        BatchRewardManager(
            config=OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": 1}}}),
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=lambda **_: [0.0],
            use_shortest_success_reward=True,
            use_response_logprob_reward_for_uniform_outcome_groups=True,
        )

    missing_group_batch = _build_batch(
        tokenizer,
        response_specs=[("r0", 2), ("r1", 2), ("r2", 2), ("r3", 2)],
        group_ids=None,
    )
    manager = _manager(tokenizer, acc_by_prefix={f"r{i}": 0.0 for i in range(4)})
    with pytest.raises(KeyError, match="prompt_group_id"):
        manager(missing_group_batch, return_dict=True)

    incomplete_batch = _build_batch(
        tokenizer,
        response_specs=[("r0", 2), ("r1", 2), ("r2", 2), ("r3", 2)],
        group_ids=["g", "g", "g", "other"],
    )
    with pytest.raises(ValueError, match="complete rollout groups"):
        manager(incomplete_batch, return_dict=True)

    gapped_batch = _build_batch(
        tokenizer,
        response_specs=[("r0", 3), ("r1", 3), ("r2", 3), ("r3", 3)],
        group_ids=["g"] * 4,
    )
    gapped_batch.batch["response_mask"][0] = torch.tensor([1, 0, 1])
    with pytest.raises(ValueError, match="contiguous left-aligned"):
        manager(gapped_batch, return_dict=True)


def test_shortest_success_metrics_match_per_group_fields():
    reward_extra = {
        "acc": [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "shortest_success_selected": [True, False, False, False, True, False, False, False],
        "shortest_success_response_tokens": [10, 12, 5, 7, 8, 4, 6, 9],
        "shortest_success_group_id": ["a"] * 4 + ["b"] * 4,
        "shortest_success_group_has_success": [True] * 8,
        "shortest_success_group_min_tokens": [10] * 4 + [8] * 4,
    }

    metrics = _compute_shortest_success_reward_metrics(reward_extra)

    assert metrics["reward/shortest_success/selected_fraction"] == 0.25
    assert math.isclose(metrics["reward/shortest_success/selected_per_correct"], 2.0 / 3.0)
    assert metrics["reward/shortest_success/groups_with_success_fraction"] == 1.0
    assert metrics["reward/shortest_success/mean_min_success_tokens"] == 9.0
    assert metrics["reward/shortest_success/mean_selected_tokens"] == 9.0
    assert metrics["reward/shortest_success/raw_acc_mean"] == 0.375


def test_longest_success_penalty_uses_inclusive_margin_and_penalizes_all_max_ties():
    tokenizer = DummyTokenizer()
    specs = [
        ("a0", 10),
        ("a1", 15),
        ("a2", 7),
        ("a3", 20),
        ("b0", 13),
        ("b1", 8),
        ("b2", 5),
        ("b3", 13),
        ("c0", 3),
        ("c1", 4),
        ("c2", 5),
        ("c3", 6),
    ]
    batch = _build_batch(
        tokenizer,
        response_specs=specs,
        group_ids=["boundary"] * 4 + ["wide"] * 4 + ["none"] * 4,
        prompt_lengths=[1, 6, 2, 5, 4, 1, 5, 2, 2, 7, 3, 1],
    )
    manager = _longest_penalty_manager(
        tokenizer,
        acc_by_prefix={
            "a0": 1.0,
            "a1": 1.0,
            "a2": 0.0,
            "a3": 0.0,
            "b0": 1.0,
            "b1": 1.0,
            "b2": 0.0,
            "b3": 1.0,
            **{f"c{i}": 0.0 for i in range(4)},
        },
    )

    result = manager(batch, return_dict=True)
    reward_tensor = result["reward_tensor"]
    extra = result["reward_extra_info"]

    assert extra["longest_success_penalty_reward"] == [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0] + [0.0] * 4
    assert extra["longest_success_penalized"] == [False] * 4 + [True, False, False, True] + [False] * 4
    assert extra["longest_success_group_within_margin"] == [True] * 4 + [False] * 4 + [False] * 4
    assert extra["longest_success_group_min_tokens"] == [10] * 4 + [8] * 4 + [None] * 4
    assert extra["longest_success_group_max_tokens"] == [15] * 4 + [13] * 4 + [None] * 4
    assert extra["longest_success_group_no_penalty_threshold_tokens"] == [15.0] * 4 + [12.0] * 4 + [None] * 4
    assert extra["longest_success_reward_token_index"] == [length - 1 for _, length in specs]
    assert extra["longest_success_reward_tensor_coordinate"] == ["response_only"] * 12
    assert extra["longest_success_nonzero_reward_token_count"] == [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    assert extra["longest_success_valid_response_nonzero_reward_token_count"] == [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    assert extra["longest_success_pad_nonzero_reward_token_count"] == [0] * 12
    assert extra["longest_success_reward_tensor_row_sum"] == extra["longest_success_penalty_reward"]
    assert torch.count_nonzero(reward_tensor).item() == 3
    assert reward_tensor[0, 9].item() == 1.0
    assert reward_tensor[1, 14].item() == 1.0
    assert reward_tensor[5, 7].item() == 1.0
    assert batch.batch["acc"].tolist() == extra["acc"]
    assert manager.disable_async_reward_loop is True


def test_longest_success_penalty_single_success_is_rewarded_and_selection_is_order_independent():
    tokenizer = DummyTokenizer()
    specs = [("a0", 4), ("b0", 20), ("a1", 9), ("b1", 10), ("a2", 3), ("b2", 15), ("a3", 8), ("b3", 20)]
    groups = ["single", "wide", "single", "wide", "single", "wide", "single", "wide"]
    batch = _build_batch(tokenizer, response_specs=specs, group_ids=groups)
    manager = _longest_penalty_manager(
        tokenizer,
        acc_by_prefix={
            "a0": 1.0,
            "a1": 0.0,
            "a2": 0.0,
            "a3": 0.0,
            "b0": 1.0,
            "b1": 1.0,
            "b2": 1.0,
            "b3": 1.0,
        },
    )

    extra = manager(batch, return_dict=True)["reward_extra_info"]

    assert extra["longest_success_penalty_reward"] == [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    assert extra["longest_success_penalized"] == [False, True, False, False, False, False, False, True]
    assert extra["longest_success_group_min_tokens"] == [4, 10, 4, 10, 4, 10, 4, 10]
    assert extra["longest_success_group_max_tokens"] == [4, 20, 4, 20, 4, 20, 4, 20]


def test_longest_success_penalty_ignores_precomputed_rm_scores_but_default_path_does_not():
    tokenizer = DummyTokenizer()
    batch = _build_batch(
        tokenizer,
        response_specs=[("r0", 2), ("r1", 3), ("r2", 6), ("r3", 7)],
        group_ids=["g"] * 4,
    )
    batch.batch["rm_scores"] = torch.full_like(batch.batch["responses"], 7.0, dtype=torch.float32)

    enabled_manager = _longest_penalty_manager(
        tokenizer,
        acc_by_prefix={"r0": 1.0, "r1": 1.0, "r2": 1.0, "r3": 0.0},
    )
    enabled = enabled_manager(batch, return_dict=True)["reward_tensor"]
    assert enabled.sum().item() == 2.0

    disabled_manager = _longest_penalty_manager(
        tokenizer,
        acc_by_prefix={"r0": 1.0, "r1": 1.0, "r2": 1.0, "r3": 0.0},
        enabled=False,
    )
    disabled = disabled_manager(batch, return_dict=True)["reward_tensor"]
    assert torch.equal(disabled, batch.batch["rm_scores"])


def test_longest_success_penalty_fails_closed_on_configuration_groups_and_masks():
    tokenizer = DummyTokenizer()
    with pytest.raises(ValueError, match="finite and non-negative"):
        _longest_penalty_manager(tokenizer, acc_by_prefix={}, margin_percent=-1.0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        BatchRewardManager(
            config=OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": 1}}}),
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=lambda **_: [0.0],
            use_shortest_success_reward=True,
            use_longest_success_penalty_reward=True,
        )

    missing_group_batch = _build_batch(
        tokenizer,
        response_specs=[("r0", 2), ("r1", 2), ("r2", 2), ("r3", 2)],
        group_ids=None,
    )
    manager = _longest_penalty_manager(tokenizer, acc_by_prefix={f"r{i}": 0.0 for i in range(4)})
    with pytest.raises(KeyError, match="prompt_group_id"):
        manager(missing_group_batch, return_dict=True)

    incomplete_batch = _build_batch(
        tokenizer,
        response_specs=[("r0", 2), ("r1", 2), ("r2", 2), ("r3", 2)],
        group_ids=["g", "g", "g", "other"],
    )
    with pytest.raises(ValueError, match="complete rollout groups"):
        manager(incomplete_batch, return_dict=True)

    gapped_batch = _build_batch(
        tokenizer,
        response_specs=[("r0", 3), ("r1", 3), ("r2", 3), ("r3", 3)],
        group_ids=["g"] * 4,
    )
    gapped_batch.batch["response_mask"][0] = torch.tensor([1, 0, 1])
    with pytest.raises(ValueError, match="contiguous left-aligned"):
        manager(gapped_batch, return_dict=True)

    mismatched_batch = _build_batch(
        tokenizer,
        response_specs=[("r0", 3), ("r1", 3), ("r2", 3), ("r3", 3)],
        group_ids=["g"] * 4,
    )
    mismatched_batch.batch["response_mask"][0, -1] = 0
    with pytest.raises(ValueError, match="must equal the response slice of attention_mask"):
        manager(mismatched_batch, return_dict=True)


def test_longest_success_penalty_metrics_match_group_fields():
    reward_extra = {
        "acc": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        "longest_success_penalty_reward": [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "longest_success_penalized": [False, False, False, False, False, True, True, False],
        "longest_success_response_tokens": [10, 15, 5, 7, 8, 13, 13, 9],
        "longest_success_group_id": ["a"] * 4 + ["b"] * 4,
        "longest_success_group_has_success": [True] * 8,
        "longest_success_group_within_margin": [True] * 4 + [False] * 4,
        "longest_success_group_min_tokens": [10] * 4 + [8] * 4,
        "longest_success_group_max_tokens": [15] * 4 + [13] * 4,
    }

    metrics = _compute_longest_success_penalty_reward_metrics(reward_extra)

    assert metrics["reward/longest_success_penalty/rewarded_fraction"] == 3.0 / 8.0
    assert metrics["reward/longest_success_penalty/rewarded_per_correct"] == 3.0 / 5.0
    assert metrics["reward/longest_success_penalty/penalized_per_correct"] == 2.0 / 5.0
    assert metrics["reward/longest_success_penalty/groups_with_success_fraction"] == 1.0
    assert metrics["reward/longest_success_penalty/successful_groups_within_margin_fraction"] == 0.5
    assert metrics["reward/longest_success_penalty/successful_groups_penalized_fraction"] == 0.5
    assert metrics["reward/longest_success_penalty/mean_min_success_tokens"] == 9.0
    assert metrics["reward/longest_success_penalty/mean_max_success_tokens"] == 14.0
    assert math.isclose(metrics["reward/longest_success_penalty/mean_max_to_min_ratio"], 1.5625)
    assert metrics["reward/longest_success_penalty/mean_rewarded_tokens"] == 11.0
    assert metrics["reward/longest_success_penalty/mean_penalized_tokens"] == 13.0
    assert metrics["reward/longest_success_penalty/raw_acc_mean"] == 0.625
