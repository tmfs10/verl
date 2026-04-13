# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

from unittest.mock import patch

import pytest

from verl.utils.reward_score import default_compute_score


def test_math_entropy_bonus_is_zero_for_repeated_wrong_answer():
    with patch(
        "verl.utils.reward_score.math_dapo.compute_score",
        return_value={"score": -1.0, "acc": False, "pred": "41"},
    ):
        result = default_compute_score(
            data_source="math",
            solution_str="wrong answer",
            ground_truth="42",
            extra_info={"answer_history": ["41", "41"], "entropy_bonus_coef": 0.5},
        )

    assert result["score"] == 0.0
    assert result["normalized_answer_entropy"] == 0.0
    assert result["answer_entropy_bonus"] == 0.0


def test_math_entropy_bonus_reaches_half_for_two_distinct_wrong_answers():
    with patch(
        "verl.utils.reward_score.math_dapo.compute_score",
        return_value={"score": -1.0, "acc": False, "pred": "43"},
    ):
        result = default_compute_score(
            data_source="math",
            solution_str="wrong answer",
            ground_truth="42",
            extra_info={"answer_history": ["41", "43"], "entropy_bonus_coef": 0.5},
        )

    assert result["score"] == pytest.approx(0.5)
    assert result["normalized_answer_entropy"] == pytest.approx(1.0)
    assert result["answer_entropy_bonus"] == pytest.approx(0.5)


def test_math_entropy_bonus_is_disabled_when_coefficient_is_zero():
    with patch(
        "verl.utils.reward_score.math_dapo.compute_score",
        return_value={"score": -1.0, "acc": False, "pred": "43"},
    ):
        result = default_compute_score(
            data_source="math",
            solution_str="wrong answer",
            ground_truth="42",
            extra_info={"answer_history": ["41", "43"], "entropy_bonus_coef": 0.0},
        )

    assert result["score"] == 0.0
    assert result["answer_entropy_bonus"] == 0.0


def test_math_entropy_bonus_preserves_correct_final_answer():
    with patch(
        "verl.utils.reward_score.math_dapo.compute_score",
        return_value={"score": 1.0, "acc": True, "pred": "42"},
    ):
        result = default_compute_score(
            data_source="math",
            solution_str="correct answer",
            ground_truth="42",
            extra_info={"answer_history": ["41", "42"], "entropy_bonus_coef": 0.5},
        )

    assert result == {"score": 1.0, "acc": True, "pred": "42"}


def test_last_completed_turn_reward_uses_last_completed_answer():
    result = default_compute_score(
        data_source="math",
        solution_str="interrupted final answer",
        ground_truth="42",
        extra_info={
            "reward_mode": "last_completed_turn",
            "has_last_completed_answer": True,
            "last_completed_answer": "42",
            "last_completed_answer_correct": True,
        },
    )

    assert result == {"score": 1.0, "acc": True, "pred": "42", "reward_mode": "last_completed_turn"}


def test_last_completed_turn_reward_is_zero_without_completed_correct_answer():
    result = default_compute_score(
        data_source="math",
        solution_str="interrupted final answer",
        ground_truth="42",
        extra_info={
            "reward_mode": "last_completed_turn",
            "has_last_completed_answer": False,
            "last_completed_answer": None,
            "last_completed_answer_correct": False,
        },
    )

    assert result == {"score": 0.0, "acc": False, "pred": None, "reward_mode": "last_completed_turn"}
