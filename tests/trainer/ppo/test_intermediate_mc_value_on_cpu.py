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
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from verl.trainer.config import INTERMEDIATE_MC_CRITIQUE_PROMPT, IntermediateMCValueConfig
from verl.trainer.ppo.intermediate_mc_value import (
    BetaValueLossComponents,
    VarianceCandidate,
    aggregate_mark_targets,
    beta_value_loss_components,
    build_critic_context,
    candidate_bounds,
    critique_accuracy_reward,
    critique_group_advantages,
    masked_whiten,
    scalar_value_loss_components,
    select_random_marks,
    select_variance_marks,
    stable_rng,
    terminal_index,
    token_gae,
    validate_reward,
)


def test_only_two_closed_recipes_are_accepted() -> None:
    scalar = IntermediateMCValueConfig(recipe="scalar_random")
    beta = IntermediateMCValueConfig(recipe="beta_variance")
    assert (scalar.critic_head, scalar.mark_selector, scalar.num_critic_labels) == ("scalar", "random", 1)
    assert (beta.critic_head, beta.mark_selector, beta.num_critic_labels) == ("beta", "variance", 2)
    with pytest.raises(ValueError, match="EMA"):
        IntermediateMCValueConfig(recipe="ema_scalar")


def test_zero_marks_is_a_supported_eos_only_configuration() -> None:
    config = IntermediateMCValueConfig(max_marks=0)
    assert config.max_marks == 0
    assert (
        select_random_marks(
            10,
            k=0,
            min_gap=1,
            start_fraction=0.0,
            end_fraction=1.0,
            rng=random.Random(0),
        )
        == []
    )


def test_composed_critique_prompt_preserves_exact_newlines() -> None:
    config_dir = Path(__file__).parents[3] / "verl" / "trainer" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="ppo_trainer")
    assert config.algorithm.intermediate_mc_value.critique_prompt == INTERMEDIATE_MC_CRITIQUE_PROMPT


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_critiques": 0}, "num_critiques"),
        ({"num_critiques": 4.0}, "num_critiques"),
        ({"critic_warmup_updates": -1}, "critic_warmup_updates"),
        ({"selection_seed": 1.0}, "selection_seed"),
        ({"mark_start_fraction": 0.8, "mark_end_fraction": 0.2}, "mark fractions"),
        ({"variance_scope": "queue"}, "variance_scope"),
        ({"max_reward": 0.0}, "max_reward"),
        ({"beta_target_epsilon": 0.5}, "beta_target_epsilon"),
        ({"critique_prompt": "different"}, "exactly match"),
    ],
)
def test_invalid_recipe_parameters_fail_closed(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        IntermediateMCValueConfig(**kwargs)


@pytest.mark.parametrize("reward", [True, float("nan"), -0.1, 1.1])
def test_invalid_environment_rewards_fail_closed(reward) -> None:
    with pytest.raises(ValueError, match="reward"):
        validate_reward(reward, 1.0)


def test_random_marks_are_deterministic_bounded_and_gap_feasible() -> None:
    kwargs = dict(
        response_length=100,
        k=4,
        min_gap=10,
        start_fraction=0.1,
        end_fraction=0.8,
    )
    first = select_random_marks(**kwargs, rng=stable_rng(7, 3, "sample", 1))
    second = select_random_marks(**kwargs, rng=stable_rng(7, 3, "sample", 1))
    assert first == second
    assert len(first) == 4
    assert all(10 <= mark <= 80 for mark in first)
    assert all(right - left >= 10 for left, right in zip(first, first[1:], strict=False))


def test_random_marks_return_largest_feasible_subset_and_never_terminal() -> None:
    assert candidate_bounds(1) == (1, 0)
    marks = select_random_marks(
        10,
        k=8,
        min_gap=4,
        start_fraction=0.0,
        end_fraction=1.0,
        rng=random.Random(4),
    )
    assert len(marks) == 3
    assert max(marks) < 10
    assert all(right - left >= 4 for left, right in zip(marks, marks[1:], strict=False))


def test_variance_selection_ties_scope_gap_and_random_fallback() -> None:
    candidates = [
        VarianceCandidate(0, "a", 10, 0.3),
        VarianceCandidate(0, "a", 20, 0.3),
        VarianceCandidate(1, "b", 5, 0.3),
    ]
    selected = select_variance_marks(
        candidates,
        k=2,
        min_gap=32,
        random_probability=0.0,
        rng=random.Random(0),
    )
    assert [(item.candidate.rollout_id, item.candidate.token) for item in selected] == [("a", 10), ("b", 5)]
    fallback = select_variance_marks(
        candidates,
        k=1,
        min_gap=32,
        random_probability=1.0,
        rng=random.Random(0),
    )
    assert fallback[0].reason == "random_fallback"


def test_mark_rewards_average_within_mark_then_across_applicable_marks() -> None:
    per_mark, dense = aggregate_mark_targets({2: [0.0, 1.0], 4: [1.0, 1.0]})
    assert per_mark == {2: 0.5, 4: 1.0}
    assert dense == {1: 0.75, 2: 0.75, 3: 1.0, 4: 1.0}
    assert aggregate_mark_targets({}) == ({}, {})
    with pytest.raises(ValueError, match="at least one"):
        aggregate_mark_targets({2: []})


def test_token_gae_uses_delimiter_value_and_terminal_reward() -> None:
    # [V(s0), V(s1), V(s2), V(s3)] for actions x1, x2, x3.
    values = [0.25, 0.4, 0.6, 0.9]
    assert token_gae(values, 1.0, gamma=1.0, gae_lambda=1.0) == pytest.approx([0.75, 0.6, 0.4])


def test_masked_whitening_does_not_mix_masked_or_critique_tokens() -> None:
    values = torch.tensor([[1.0, 2.0, 100.0], [3.0, 200.0, 300.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    result = masked_whiten(values, mask)
    torch.testing.assert_close(result[mask.bool()].mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(result[mask.bool()].var(unbiased=False), torch.tensor(1.0), atol=1e-6, rtol=0)
    assert torch.equal(result[~mask.bool()], torch.zeros_like(result[~mask.bool()]))


def test_critique_reward_and_population_normalization() -> None:
    assert critique_accuracy_reward([1.0, 1.0], [1.0, 1.0], max_reward=1.0) == 1.0
    assert critique_accuracy_reward([0.0, 0.0], [1.0, 1.0], max_reward=1.0) == 0.0
    assert critique_group_advantages([0.2, 0.8], 1e-8) == pytest.approx([-1.0, 1.0])
    assert critique_group_advantages([0.7], 1e-8) == [0.0]


def test_scalar_mse_and_bce_share_the_same_value_clip() -> None:
    logits = torch.tensor([[0.0, 2.0]])
    targets = torch.tensor([[0.25, 0.75]])
    old = torch.tensor([[0.4, 0.4]])
    mse = scalar_value_loss_components(
        logits,
        targets,
        old,
        max_reward=1.0,
        value_clip_epsilon=0.2,
        target_loss="mse",
    )
    bce = scalar_value_loss_components(
        logits,
        targets,
        old,
        max_reward=1.0,
        value_clip_epsilon=0.2,
        target_loss="bce",
    )
    expected_values = torch.sigmoid(logits)
    expected_clip = old + torch.clamp(expected_values - old, -0.2, 0.2)
    torch.testing.assert_close(mse.values, expected_values)
    torch.testing.assert_close(mse.clipped_values, expected_clip)
    torch.testing.assert_close(mse.current_loss, (expected_values - targets).square())
    torch.testing.assert_close(
        bce.current_loss,
        torch.nn.functional.binary_cross_entropy(expected_values, targets, reduction="none"),
    )


def test_beta_parameterization_endpoints_and_gradients_are_finite() -> None:
    logits = torch.tensor([[[20.0, -2.0], [-20.0, 2.0]]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0]])
    old = torch.tensor([[1.0, 0.0]])
    components: BetaValueLossComponents = beta_value_loss_components(
        logits,
        targets,
        old,
        max_reward=1.0,
        value_clip_epsilon=0.2,
        beta_target_epsilon=1e-4,
    )
    assert torch.all((components.mean > 0.0) & (components.mean < 1.0))
    torch.testing.assert_close(components.q, torch.sigmoid(logits[..., 1].float()))
    torch.testing.assert_close(
        components.variance,
        components.q * components.mean * (1.0 - components.mean),
    )
    torch.testing.assert_close(components.alpha + components.beta, components.kappa)
    torch.testing.assert_close(components.clipped_alpha + components.clipped_beta, components.kappa)
    loss = torch.maximum(components.current_loss, components.clipped_loss).sum()
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_reward_scale_is_applied_to_mean_and_variance() -> None:
    logits = torch.zeros((1, 1, 2))
    components = beta_value_loss_components(
        logits,
        torch.tensor([[1.0]]),
        torch.tensor([[1.0]]),
        max_reward=2.0,
        value_clip_epsilon=0.2,
        beta_target_epsilon=1e-4,
    )
    torch.testing.assert_close(components.mean, torch.tensor([[1.0]]))
    torch.testing.assert_close(components.variance, torch.tensor([[0.5]]))


def test_exact_critic_boundaries_include_solution_newline_and_solution_ids() -> None:
    context = build_critic_context(
        [1, 2],
        [3, 4],
        [5, 6, 7],
        critique_delimiter_ids=[10, 11],
        solution_delimiter_ids=[20, 21, 22],
    )
    assert context.token_ids == [1, 2, 10, 11, 3, 4, 20, 21, 22, 5, 6, 7]
    assert context.pre_solution_position == 8
    assert context.solution_positions == [9, 10, 11]
    assert [context.token_ids[index] for index in context.solution_positions] == [5, 6, 7]


def test_terminal_index_accepts_literal_or_length_capped_terminal() -> None:
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    assert terminal_index(mask).tolist() == [1, 3]
    with pytest.raises(ValueError, match="at least one"):
        terminal_index(torch.zeros((1, 4)))
