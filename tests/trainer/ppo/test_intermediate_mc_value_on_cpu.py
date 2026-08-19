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
from verl.trainer.ppo.core_algos import compute_gae_advantage_return
from verl.trainer.ppo.intermediate_mc_value import (
    CRITIQUE_DELIMITER,
    FP32_EPSILON,
    SOLUTION_DELIMITER,
    BetaValueLossComponents,
    VarianceCandidate,
    aggregate_mark_targets,
    beta_value_loss_components,
    build_critic_context,
    build_unconditioned_critic_context,
    candidate_bounds,
    critique_accuracy_reward,
    critique_group_advantages,
    initial_state_target,
    scalar_value_loss_components,
    select_ema_marks,
    select_random_marks,
    select_variance_marks,
    stable_rng,
    terminal_index,
    validate_reward,
)


def test_head_and_selector_are_independent_and_variance_requires_beta() -> None:
    scalar_random = IntermediateMCValueConfig(critic_head="scalar", mark_selector="random")
    scalar_ema = IntermediateMCValueConfig(critic_head="scalar", mark_selector="ema")
    beta_random = IntermediateMCValueConfig(critic_head="beta", mark_selector="random")
    beta_variance = IntermediateMCValueConfig(critic_head="beta", mark_selector="variance")
    assert scalar_random.num_critic_labels == 1
    assert scalar_ema.resolved_max_marks == 4
    assert beta_random.num_critic_labels == 2
    assert beta_variance.resolved_max_marks == 1
    with pytest.raises(ValueError, match="requires critic_head=beta"):
        IntermediateMCValueConfig(critic_head="scalar", mark_selector="variance")


def test_explicit_zero_marks_zero_critiques_and_one_critique_are_supported() -> None:
    assert IntermediateMCValueConfig(max_marks=0).resolved_max_marks == 0
    without_critiques = IntermediateMCValueConfig(num_critiques=0)
    assert without_critiques.num_critiques == 0
    assert without_critiques.num_critic_streams == 1
    with pytest.warns(UserWarning, match="exactly zero"):
        config = IntermediateMCValueConfig(num_critiques=1)
    assert config.num_critiques == 1
    assert config.num_critic_streams == 1


def test_hydra_base_and_feature_preset_preserve_native_actor_loss_override() -> None:
    config_dir = Path(__file__).parents[3] / "verl" / "trainer" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        base = compose(config_name="ppo_trainer")
        preset = compose(config_name="intermediate_mc_ppo_trainer")
        overridden = compose(
            config_name="intermediate_mc_ppo_trainer",
            overrides=["actor_rollout_ref.actor.policy_loss.loss_mode=vanilla"],
        )
    assert base.algorithm.intermediate_mc_value.enable is False
    assert base.algorithm.intermediate_mc_value.critique_prompt == INTERMEDIATE_MC_CRITIQUE_PROMPT
    assert preset.algorithm.intermediate_mc_value.enable is True
    assert preset.trainer.critic_warmup == 30
    assert preset.critic.cliprange_value == 0.2
    assert preset.actor_rollout_ref.actor.policy_loss.loss_mode == "dppo_tv"
    assert overridden.actor_rollout_ref.actor.policy_loss.loss_mode == "vanilla"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"critic_head": "categorical"}, "critic_head"),
        ({"mark_selector": "queue"}, "mark_selector"),
        ({"num_critiques": -1}, "num_critiques"),
        ({"num_critiques": True}, "num_critiques"),
        ({"num_critiques": 4.0}, "num_critiques"),
        ({"max_marks": -1}, "max_marks"),
        ({"selection_seed": 1.0}, "selection_seed"),
        ({"ema_alpha": 0.0}, "ema_alpha"),
        ({"ema_ratio_up": 1.0}, "ema_ratio_up"),
        ({"ema_ratio_down": 1.0}, "ema_ratio_down"),
        ({"mark_start_fraction": 0.8, "mark_end_fraction": 0.2}, "mark fractions"),
        ({"variance_scope": "queue"}, "variance_scope"),
        ({"max_reward": 0.0}, "max_reward"),
        ({"beta_target_epsilon": 0.5}, "beta_target_epsilon"),
        ({"critique_prompt": "different"}, "exactly match"),
    ],
)
def test_invalid_parameters_fail_closed(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        IntermediateMCValueConfig(**kwargs)


@pytest.mark.parametrize("reward", [True, float("nan"), -0.1, 1.1])
def test_invalid_environment_rewards_fail_closed(reward) -> None:
    with pytest.raises(ValueError, match="reward"):
        validate_reward(reward, 1.0)


def test_random_marks_are_deterministic_bounded_and_gap_feasible() -> None:
    kwargs = dict(response_length=100, k=4, min_gap=10, start_fraction=0.1, end_fraction=0.8)
    first = select_random_marks(**kwargs, rng=stable_rng(7, 3, "sample", 1))
    second = select_random_marks(**kwargs, rng=stable_rng(7, 3, "sample", 1))
    assert first == second
    assert len(first) == 4
    assert all(10 <= mark <= 80 for mark in first)
    assert all(right - left >= 10 for left, right in zip(first, first[1:], strict=False))


def test_random_marks_return_largest_feasible_nonterminal_subset() -> None:
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


def test_ema_uses_float64_critic_values_and_chronological_reference_updates() -> None:
    values = [0.2, 0.5, 0.4, 0.1, 0.1]
    selections, ema_values = select_ema_marks(
        values,
        k=4,
        min_gap=1,
        start_fraction=0.0,
        end_fraction=1.0,
        alpha=1.0,
        baseline_token=1,
        floor=1e-4,
        ratio_up=2.0,
        ratio_down=0.5,
    )
    assert ema_values == pytest.approx(values)
    assert [(item.token, item.direction) for item in selections] == [(2, "up"), (4, "down")]
    assert [item.value for item in selections] == pytest.approx([0.5, 0.1])
    assert selections[1].reference == pytest.approx(0.5)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1])
def test_ema_rejects_invalid_critic_value(invalid) -> None:
    with pytest.raises(ValueError, match="finite"):
        select_ema_marks(
            [0.0, invalid],
            k=1,
            min_gap=1,
            start_fraction=0.0,
            end_fraction=1.0,
            alpha=0.1,
            baseline_token=1,
            floor=1e-4,
            ratio_up=2.0,
            ratio_down=0.5,
        )


def test_ema_keeps_configured_baseline_when_candidate_window_starts_later() -> None:
    values = [0.9] * 49 + [0.1] * 51
    selections, _ = select_ema_marks(
        values,
        k=1,
        min_gap=1,
        start_fraction=0.5,
        end_fraction=0.8,
        alpha=1.0,
        baseline_token=2,
        floor=1e-4,
        ratio_up=2.0,
        ratio_down=0.5,
    )
    assert [selection.token for selection in selections] == [50]
    assert selections[0].reference == pytest.approx(0.9)


def test_ema_min_gap_is_anchored_at_the_configured_baseline() -> None:
    selections, _ = select_ema_marks(
        [0.2, 0.8, 0.8, 0.8, 0.8],
        k=1,
        min_gap=3,
        start_fraction=0.0,
        end_fraction=1.0,
        alpha=1.0,
        baseline_token=1,
        floor=1e-4,
        ratio_up=2.0,
        ratio_down=0.5,
    )
    assert [selection.token for selection in selections] == [4]


def test_variance_selection_ties_gap_and_random_fallback() -> None:
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


def test_mark_targets_and_trained_initial_state_target() -> None:
    per_mark, dense = aggregate_mark_targets({2: [0.0, 1.0], 4: [1.0, 1.0]})
    assert per_mark == {2: 0.5, 4: 1.0}
    assert dense == {1: 0.75, 2: 0.75, 3: 1.0, 4: 1.0}
    assert initial_state_target(0.0, per_mark) == pytest.approx(0.5)
    assert initial_state_target(0.75, {}) == pytest.approx(0.75)


def test_native_gae_uses_pre_action_values_and_terminal_reward() -> None:
    values = torch.tensor([[0.25, 0.4, 0.6]])
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    mask = torch.ones_like(values)
    advantages, returns = compute_gae_advantage_return(rewards, values, mask, gamma=1.0, lam=1.0)
    raw = torch.tensor([[0.75, 0.6, 0.4]])
    # Match VeRL's native masked_whiten, which uses Bessel-corrected variance.
    expected = (raw - raw.mean()) / torch.sqrt(raw.var(unbiased=True) + 1e-8)
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, raw + values)


def test_critique_reward_and_population_normalization() -> None:
    assert critique_accuracy_reward([1.0, 1.0], [1.0, 1.0], max_reward=1.0) == 1.0
    assert critique_accuracy_reward([0.0, 0.0], [1.0, 1.0], max_reward=1.0) == 0.0
    assert critique_group_advantages([0.2, 0.8], 1e-8) == pytest.approx([-1.0, 1.0])
    assert critique_group_advantages([0.7], 1e-8) == [0.0]


def test_scalar_mse_and_bce_use_reward_normalized_value_clip() -> None:
    logits = torch.tensor([[0.0, 2.0]])
    targets = torch.tensor([[0.5, 1.5]])
    old = torch.tensor([[0.8, 0.8]])
    mse = scalar_value_loss_components(
        logits,
        targets,
        old,
        max_reward=2.0,
        cliprange_value=0.2,
        target_loss="mse",
    )
    bce = scalar_value_loss_components(
        logits,
        targets,
        old,
        max_reward=2.0,
        cliprange_value=0.2,
        target_loss="bce",
    )
    expected_values = 2.0 * torch.sigmoid(logits)
    expected_clip = old + 2.0 * torch.clamp((expected_values - old) / 2.0, -0.2, 0.2)
    torch.testing.assert_close(mse.values, expected_values)
    torch.testing.assert_close(mse.clipped_values, expected_clip)
    torch.testing.assert_close(mse.current_loss, (expected_values - targets).square())
    assert torch.isfinite(bce.current_loss).all()


def test_beta_parameterization_mean_only_clip_and_gradients() -> None:
    logits = torch.tensor([[[20.0, -2.0], [-20.0, 2.0]]], requires_grad=True)
    components: BetaValueLossComponents = beta_value_loss_components(
        logits,
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.5, 0.5]]),
        max_reward=1.0,
        cliprange_value=0.2,
        beta_target_epsilon=1e-4,
    )
    assert torch.all((components.mean >= FP32_EPSILON) & (components.mean <= 1.0 - FP32_EPSILON))
    torch.testing.assert_close(components.q, torch.sigmoid(logits[..., 1].float()))
    torch.testing.assert_close(components.variance, components.q * components.mean * (1.0 - components.mean))
    torch.testing.assert_close(components.alpha + components.beta, components.kappa)
    torch.testing.assert_close(components.clipped_alpha + components.clipped_beta, components.kappa)
    assert torch.all((components.clipped_values - 0.5).abs() <= 0.2 + 1e-6)
    loss = torch.maximum(components.current_loss, components.clipped_loss).sum()
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_beta_value_clip_scales_with_reward_range() -> None:
    logits = torch.tensor([[[2.0, 0.0]]])
    old = torch.tensor([[0.5]])
    components = beta_value_loss_components(
        logits,
        torch.tensor([[1.0]]),
        old,
        max_reward=2.0,
        cliprange_value=0.2,
        beta_target_epsilon=1e-4,
    )
    expected = old + 2.0 * torch.clamp((components.mean - old) / 2.0, -0.2, 0.2)
    torch.testing.assert_close(components.clipped_values, expected)
    torch.testing.assert_close(components.clipped_values, torch.tensor([[0.9]]))


def test_beta_prediction_clamp_is_independent_of_target_transform_epsilon() -> None:
    logits = torch.tensor([[[-10.0, 0.0]]], requires_grad=True)
    components = beta_value_loss_components(
        logits,
        torch.tensor([[0.0]]),
        torch.tensor([[0.5]]),
        max_reward=1.0,
        cliprange_value=0.2,
        beta_target_epsilon=0.1,
    )
    torch.testing.assert_close(components.mean, torch.sigmoid(logits[..., 0]))
    components.mean.sum().backward()
    assert logits.grad is not None
    assert logits.grad[..., 0].item() > 0.0


def test_exact_critic_boundaries_include_s0_and_solution_states() -> None:
    context = build_critic_context(
        [1, 2],
        [3, 4],
        [5, 6, 7],
        critique_delimiter_ids=[10, 11],
        solution_delimiter_ids=[20, 21, 22],
    )
    assert context.token_ids == [1, 2, 10, 11, 3, 4, 20, 21, 22, 5, 6, 7]
    assert context.pre_solution_position == 8
    assert context.value_positions == [8, 9, 10, 11]
    assert [context.token_ids[index] for index in context.solution_positions] == [5, 6, 7]


def test_unconditioned_critic_context_uses_one_stream_and_no_critique_boundary() -> None:
    context = build_unconditioned_critic_context(
        [1, 2],
        [5, 6, 7],
        solution_delimiter_ids=[20, 21, 22],
    )
    assert context.token_ids == [1, 2, 20, 21, 22, 5, 6, 7]
    assert context.prompt_range == (0, 2)
    assert context.critique_delimiter_range == (2, 2)
    assert context.critique_range == (2, 2)
    assert context.pre_solution_position == 4
    assert context.value_positions == [4, 5, 6, 7]
    assert [context.token_ids[index] for index in context.solution_positions] == [5, 6, 7]


def test_literal_critic_delimiters_match_the_required_multiline_context() -> None:
    assert CRITIQUE_DELIMITER == "\n\nCritique:\n"
    assert SOLUTION_DELIMITER == "\n\nSolution:\n"
    assert f"q{CRITIQUE_DELIMITER}c{SOLUTION_DELIMITER}x" == "q\n\nCritique:\nc\n\nSolution:\nx"


def test_terminal_index_accepts_literal_or_length_capped_terminal() -> None:
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    assert terminal_index(mask).tolist() == [1, 3]
    with pytest.raises(ValueError, match="at least one"):
        terminal_index(torch.zeros((1, 4)))
