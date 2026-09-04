# Copyright 2026 NVIDIA Corporation
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

import random

import pytest

from verl.trainer.config import RandomContinuationBaselineConfig
from verl.trainer.ppo.random_continuation_baseline import (
    clustered_rate,
    select_structurally_valid_random_marks,
    strict_candidate_bounds,
)


class _CharTokenizer:
    eos_token_id = 0

    def decode(self, token_ids, **_kwargs):
        return "".join(chr(token) for token in token_ids)


def _ids(text: str) -> list[int]:
    return [ord(char) for char in text]


def test_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="points_per_rollout"):
        RandomContinuationBaselineConfig(points_per_rollout=0)
    with pytest.raises(ValueError, match="min_prefix_fraction"):
        RandomContinuationBaselineConfig(min_prefix_fraction=1.0)


def test_strict_bounds_exclude_exact_ten_percent_and_preserve_suffix_budget():
    low, high = strict_candidate_bounds(
        100,
        min_prefix_fraction=0.10,
        response_budget=100,
        min_continuation_tokens=20,
    )
    assert (low, high) == (11, 80)


def test_structural_selection_skips_open_math_and_is_deterministic():
    # Positions after the opening delimiter remain invalid until the closing delimiter.
    tokens = _ids("abcdefghijk$$inside$$tail")
    first = select_structurally_valid_random_marks(
        tokens,
        tokenizer=_CharTokenizer(),
        points_per_rollout=8,
        min_prefix_fraction=0.10,
        response_budget=100,
        min_continuation_tokens=1,
        rng=random.Random(7),
    )
    second = select_structurally_valid_random_marks(
        tokens,
        tokenizer=_CharTokenizer(),
        points_per_rollout=8,
        min_prefix_fraction=0.10,
        response_budget=100,
        min_continuation_tokens=1,
        rng=random.Random(7),
    )
    assert first == second
    assert len(first.marks) == 8
    for mark in first.marks:
        prefix = _CharTokenizer().decode(tokens[:mark])
        assert not (prefix.count("$$") % 2)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("prefix\n```python\nvalue = 1", "branch_inside_code_fence"),
        ("prefix $$ value", "branch_inside_display_math"),
        (r"prefix \[ value", "branch_inside_display_math"),
        (r"prefix \begin{aligned} value", "branch_inside_latex_environment"),
    ],
)
def test_selection_records_production_structural_rejections(text: str, reason: str):
    tokens = _ids(text)
    result = select_structurally_valid_random_marks(
        tokens,
        tokenizer=_CharTokenizer(),
        points_per_rollout=len(tokens),
        min_prefix_fraction=0.0,
        response_budget=10_000,
        min_continuation_tokens=1,
        rng=random.Random(1),
    )
    assert result.rejection_counts[reason] > 0
    assert len(result.marks) < len(tokens) - 1


def test_selection_reports_shortfall_instead_of_using_invalid_positions():
    tokens = _ids("a$$never closes")
    result = select_structurally_valid_random_marks(
        tokens,
        tokenizer=_CharTokenizer(),
        points_per_rollout=8,
        min_prefix_fraction=0.0,
        response_budget=10_000,
        min_continuation_tokens=1,
        rng=random.Random(2),
    )
    assert len(result.marks) < 8
    assert result.inspected == len(tokens) - 1


def test_clustered_rate_uses_prompt_clusters():
    result = clustered_rate([[1.0] * 8, [0.0]], bootstrap_samples=1000, seed=3)
    assert result["attempt_weighted"] == pytest.approx(8 / 9)
    assert result["prompt_weighted"] == pytest.approx(0.5)
    assert result["prompts"] == 2
    assert result["attempts"] == 9
