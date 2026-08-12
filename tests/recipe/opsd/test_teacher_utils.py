import pytest
import torch

from recipe.opsd.teacher_utils import (
    assign_prompt_crossfit_folds,
    build_teacher_sft_mask,
    find_last_token_subsequence,
    select_sdpo_caa_source_indices,
    select_sdpo_teacher_candidate_indices,
    select_sdpo_teacher_indices,
)


def test_prompt_crossfit_folds_are_stable_and_keep_rollouts_together():
    uids = ["a", "a", "b", "c", "b", "c"]
    folds, groups, count = assign_prompt_crossfit_folds(uids, seed=1234)

    assert count == 3
    assert groups == [0, 0, 1, 2, 1, 2]
    assert folds[0] == folds[1]
    assert folds[2] == folds[4]
    assert folds[3] == folds[5]
    assert assign_prompt_crossfit_folds(uids, seed=1234) == (folds, groups, count)


def test_find_last_token_subsequence_uses_final_closing_think_tag():
    assert find_last_token_subsequence([10, 90, 91, 11, 90, 91, 12], [90, 91]) == 4
    assert find_last_token_subsequence([10, 11], [90, 91]) is None


def test_teacher_sft_thinking_only_includes_final_delimiter_and_excludes_answer():
    response_ids = torch.tensor(
        [
            [10, 90, 91, 20, 90, 91, 30, 31, 0],
            [40, 90, 91, 50, 0, 0, 0, 0, 0],
        ]
    )
    attention = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )

    mask, boundaries = build_teacher_sft_mask(
        response_ids,
        attention,
        torch.tensor([True, False]),
        target_scope="thinking_only",
        think_end_token_ids=[90, 91],
    )

    # The final closing tag is selected. Its two tokens are supervised; the
    # answer following it and the unsuccessful second rollout are not.
    assert boundaries.tolist() == [6, 3]
    assert mask.tolist() == [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]


def test_teacher_sft_thinking_and_answer_uses_attention_not_pad_token_identity():
    # Token id 0 appears in an attended response position. This deliberately
    # recreates the PAD==EOS class of trap: validity must come from attention,
    # never from comparing token IDs with a padding ID.
    response_ids = torch.tensor([[10, 0, 20, 0], [30, 31, 0, 0]])
    attention = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)

    mask, boundaries = build_teacher_sft_mask(
        response_ids,
        attention,
        torch.tensor([1.0, 0.0]),
        target_scope="thinking_and_answer",
        think_end_token_ids=[90, 91],
    )

    assert boundaries.tolist() == [-1, -1]
    assert mask.tolist() == [[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]


def test_teacher_sft_thinking_only_rejects_success_without_closing_tag():
    with pytest.raises(ValueError, match="missing the configured closing-think delimiter"):
        build_teacher_sft_mask(
            torch.tensor([[10, 20, 0]]),
            torch.tensor([[1, 1, 0]], dtype=torch.float32),
            torch.tensor([True]),
            target_scope="thinking_only",
            think_end_token_ids=[90, 91],
        )


def test_teacher_sft_rejects_interior_response_padding():
    with pytest.raises(ValueError, match="not right padded"):
        build_teacher_sft_mask(
            torch.tensor([[10, 0, 20]]),
            torch.tensor([[1, 0, 1]], dtype=torch.float32),
            torch.tensor([True]),
            target_scope="thinking_and_answer",
            think_end_token_ids=[90, 91],
        )


def test_select_sdpo_teacher_indices_targets_only_failed_samples_by_default():
    teacher_indices, distill_active = select_sdpo_teacher_indices(
        uids=["a", "a", "b", "b"],
        correct=[0.0, 1.0, 1.0, 0.0],
    )

    assert teacher_indices == [1, None, None, 2]
    assert distill_active == [True, False, False, True]


def test_select_sdpo_teacher_indices_keeps_success_samples_inactive_without_peer_success():
    teacher_indices, distill_active = select_sdpo_teacher_indices(
        uids=["a", "a", "a"],
        correct=[1.0, 0.0, 0.0],
        distill_only_failed=False,
        exclude_self_success=True,
    )

    assert teacher_indices == [None, 0, 0]
    assert distill_active == [False, True, True]


def test_select_sdpo_teacher_indices_can_allow_self_teacher_for_single_success():
    teacher_indices, distill_active = select_sdpo_teacher_indices(
        uids=["a", "a", "a"],
        correct=[1.0, 0.0, 0.0],
        distill_only_failed=False,
        exclude_self_success=False,
    )

    assert teacher_indices == [0, 0, 0]
    assert distill_active == [True, True, True]


def test_select_sdpo_teacher_candidate_indices_can_keep_all_successes():
    teacher_indices, distill_active = select_sdpo_teacher_candidate_indices(
        uids=["a", "a", "a", "a"],
        correct=[0.0, 1.0, 0.0, 1.0],
        aggregation="all",
    )
    assert teacher_indices == [[1, 3], [], [1, 3], []]
    assert distill_active == [True, False, True, False]


def test_select_sdpo_caa_requires_mixed_groups_and_targets_all_outcomes():
    positives, negatives, distill_active = select_sdpo_caa_source_indices(
        uids=["a", "a", "a", "a", "b", "b", "c", "c"],
        correct=[0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
    )
    assert positives == [[1, 3], [1, 3], [1, 3], [1, 3], [], [], [], []]
    assert negatives == [[0, 2], [0, 2], [0, 2], [0, 2], [], [], [], []]
    assert distill_active == [True, True, True, True, False, False, False, False]


def test_select_sdpo_caa_can_reproduce_historical_failed_only_analysis():
    positives, negatives, distill_active = select_sdpo_caa_source_indices(
        uids=["a", "a", "a", "a"],
        correct=[0.0, 1.0, 0.0, 1.0],
        distill_only_failed=True,
    )
    assert positives == [[1, 3], [], [1, 3], []]
    assert negatives == [[0, 2], [], [0, 2], []]
    assert distill_active == [True, False, True, False]
