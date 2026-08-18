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
"""Pure synchronous helpers for intermediate Monte Carlo value supervision."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn.functional as F

FP32_EPSILON = torch.finfo(torch.float32).eps
CRITIQUE_DELIMITER = "\n\nCritique:\n"
SOLUTION_DELIMITER = "\n\nSolution:\n"


@dataclass(frozen=True)
class VarianceCandidate:
    rollout_order: int
    rollout_id: str
    token: int
    variance: float


@dataclass(frozen=True)
class VarianceSelection:
    candidate: VarianceCandidate
    reason: Literal["variance", "random_fallback"]
    draw: float


@dataclass(frozen=True)
class CriticContext:
    """Exact conditioned critic sequence and its explicit boundaries."""

    token_ids: list[int]
    prompt_range: tuple[int, int]
    critique_delimiter_range: tuple[int, int]
    critique_range: tuple[int, int]
    solution_delimiter_range: tuple[int, int]
    solution_range: tuple[int, int]

    @property
    def pre_solution_position(self) -> int:
        return self.solution_delimiter_range[1] - 1

    @property
    def solution_positions(self) -> list[int]:
        return list(range(*self.solution_range))

    @property
    def value_positions(self) -> list[int]:
        return [self.pre_solution_position, *self.solution_positions]


@dataclass(frozen=True)
class ScalarValueLossComponents:
    values: torch.Tensor
    clipped_values: torch.Tensor
    current_loss: torch.Tensor
    clipped_loss: torch.Tensor


@dataclass(frozen=True)
class BetaValueLossComponents:
    mean: torch.Tensor
    q: torch.Tensor
    variance: torch.Tensor
    kappa: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    clipped_values: torch.Tensor
    clipped_alpha: torch.Tensor
    clipped_beta: torch.Tensor
    current_loss: torch.Tensor
    clipped_loss: torch.Tensor


def stable_rng(seed: int, *parts: object) -> random.Random:
    key = ":".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def validate_reward(reward: object, max_reward: float) -> float:
    if isinstance(reward, bool) or not isinstance(reward, int | float):
        raise ValueError(f"intermediate MC requires a finite scalar reward, got {reward!r}")
    result = float(reward)
    if not math.isfinite(result) or not 0.0 <= result <= max_reward:
        raise ValueError(f"intermediate MC reward must be in [0, {max_reward}], got {reward!r}")
    return result


def candidate_bounds(
    response_length: int,
    start_fraction: float = 0.05,
    end_fraction: float = 0.90,
) -> tuple[int, int]:
    """Return inclusive one-indexed nonterminal mark bounds."""

    if response_length < 2:
        return 1, 0
    low = max(1, math.ceil(start_fraction * response_length))
    high = min(response_length - 1, math.floor(end_fraction * response_length))
    return low, high


def select_random_marks(
    response_length: int,
    *,
    k: int,
    min_gap: int,
    start_fraction: float,
    end_fraction: float,
    rng: random.Random,
) -> list[int]:
    """Uniformly sample a feasible subset using the standard gap compression bijection."""

    low, high = candidate_bounds(response_length, start_fraction, end_fraction)
    if k == 0 or low > high:
        return []
    feasible_k = min(k, 1 + (high - low) // min_gap)
    compressed_high = high - (feasible_k - 1) * (min_gap - 1)
    compressed = sorted(rng.sample(range(low, compressed_high + 1), feasible_k))
    return [token + index * (min_gap - 1) for index, token in enumerate(compressed)]


def select_variance_marks(
    candidates: Sequence[VarianceCandidate],
    *,
    k: int,
    min_gap: int,
    random_probability: float,
    rng: random.Random,
) -> list[VarianceSelection]:
    """Greedily select variance candidates with a per-slot random fallback."""

    if k < 0:
        raise ValueError("k must be non-negative")
    if min_gap <= 0:
        raise ValueError("min_gap must be positive")
    if not 0.0 <= random_probability <= 1.0:
        raise ValueError("random_probability must be in [0, 1]")
    if any(not math.isfinite(candidate.variance) or candidate.variance < 0.0 for candidate in candidates):
        raise ValueError("variance candidates must be finite and non-negative")
    eligible = list(candidates)
    selected: list[VarianceSelection] = []
    while eligible and len(selected) < k:
        draw = rng.random()
        if draw < random_probability:
            chosen = eligible[rng.randrange(len(eligible))]
            reason: Literal["variance", "random_fallback"] = "random_fallback"
        else:
            chosen = min(eligible, key=lambda item: (-item.variance, item.rollout_order, item.token))
            reason = "variance"
        selected.append(VarianceSelection(candidate=chosen, reason=reason, draw=draw))
        eligible = [
            candidate
            for candidate in eligible
            if candidate != chosen
            and not (candidate.rollout_id == chosen.rollout_id and abs(candidate.token - chosen.token) < min_gap)
        ]
    return selected


def aggregate_mark_targets(mark_rewards: dict[int, Sequence[float]]) -> tuple[dict[int, float], dict[int, float]]:
    """Average samples within marks and then applicable marks for every earlier token."""

    if not mark_rewards:
        return {}, {}
    per_mark: dict[int, float] = {}
    for mark, rewards in mark_rewards.items():
        if mark < 1 or not rewards:
            raise ValueError(f"mark {mark} must contain at least one successful continuation reward")
        numeric_rewards = [float(reward) for reward in rewards]
        if not all(math.isfinite(reward) for reward in numeric_rewards):
            raise ValueError(f"mark {mark} contains a non-finite continuation reward")
        per_mark[mark] = sum(numeric_rewards) / len(numeric_rewards)
    largest_mark = max(per_mark)
    dense = {
        token: sum(value for mark, value in per_mark.items() if mark >= token)
        / sum(1 for mark in per_mark if mark >= token)
        for token in range(1, largest_mark + 1)
    }
    return per_mark, dense


def critique_group_advantages(rewards: Sequence[float], epsilon: float) -> list[float]:
    if not rewards:
        return []
    mean = sum(float(reward) for reward in rewards) / len(rewards)
    variance = sum((float(reward) - mean) ** 2 for reward in rewards) / len(rewards)
    denominator = max(math.sqrt(variance), epsilon)
    return [(float(reward) - mean) / denominator for reward in rewards]


def critique_accuracy_reward(
    predictions: Sequence[float],
    targets: Sequence[float],
    *,
    max_reward: float,
) -> float:
    if not predictions or len(predictions) != len(targets):
        raise ValueError("critique reward needs equal non-empty prediction and target lists")
    mean_error = sum(
        ((float(prediction) - float(target)) / max_reward) ** 2
        for prediction, target in zip(predictions, targets, strict=True)
    ) / len(predictions)
    return min(1.0, max(0.0, 1.0 - mean_error))


def token_gae(
    state_values: Sequence[float],
    terminal_reward: float,
    *,
    gamma: float,
    gae_lambda: float,
) -> list[float]:
    """Compute action advantages from [V(s0), ..., V(sT)] critic means.

    The terminal action uses the environment reward and V(sT) is deliberately
    absent from its delta. It remains useful as the terminal critic label.
    """

    if len(state_values) < 2:
        raise ValueError("token GAE needs V(s0) and at least one solution-token value")
    action_count = len(state_values) - 1
    deltas = [
        (
            terminal_reward - float(state_values[action_index])
            if action_index == action_count - 1
            else gamma * float(state_values[action_index + 1]) - float(state_values[action_index])
        )
        for action_index in range(action_count)
    ]
    advantages = [0.0] * action_count
    running = 0.0
    for action_index in range(action_count - 1, -1, -1):
        running = deltas[action_index] + gamma * gae_lambda * running
        advantages[action_index] = running
    return advantages


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Whiten valid tokens while leaving every masked element exactly zero."""

    valid = mask.to(dtype=torch.bool)
    if not torch.any(valid):
        return torch.zeros_like(values)
    selected = values[valid].float()
    mean = selected.mean()
    variance = selected.var(unbiased=False)
    whitened = (values.float() - mean) * torch.rsqrt(variance + epsilon)
    return torch.where(valid, whitened, torch.zeros_like(whitened)).to(values.dtype)


def scalar_value_loss_components(
    critic_logits: torch.Tensor,
    value_targets: torch.Tensor,
    old_predictions: torch.Tensor,
    *,
    max_reward: float,
    value_clip_epsilon: float,
    target_loss: Literal["mse", "bce"],
) -> ScalarValueLossComponents:
    values = max_reward * torch.sigmoid(critic_logits.float())
    clipped_values = old_predictions.float() + max_reward * torch.clamp(
        (values - old_predictions.float()) / max_reward,
        -value_clip_epsilon,
        value_clip_epsilon,
    )
    if target_loss == "mse":
        current_loss = (values - value_targets.float()).square()
        clipped_loss = (clipped_values - value_targets.float()).square()
    elif target_loss == "bce":
        normalized_targets = value_targets.float() / max_reward
        current_loss = F.binary_cross_entropy(
            (values / max_reward).clamp(1e-6, 1.0 - 1e-6), normalized_targets, reduction="none"
        )
        clipped_loss = F.binary_cross_entropy(
            (clipped_values / max_reward).clamp(1e-6, 1.0 - 1e-6), normalized_targets, reduction="none"
        )
    else:
        raise ValueError(f"unsupported scalar critic loss {target_loss!r}")
    return ScalarValueLossComponents(values, clipped_values, current_loss, clipped_loss)


def beta_value_loss_components(
    critic_logits: torch.Tensor,
    value_targets: torch.Tensor,
    old_predictions: torch.Tensor,
    *,
    max_reward: float,
    value_clip_epsilon: float,
    beta_target_epsilon: float,
) -> BetaValueLossComponents:
    if critic_logits.ndim < 1 or critic_logits.shape[-1] != 2:
        raise ValueError(f"Beta critic logits must end in width 2, got {tuple(critic_logits.shape)}")
    z_mu = critic_logits[..., 0].float()
    z_q = critic_logits[..., 1].float()
    normalized_mean = torch.sigmoid(z_mu).clamp(FP32_EPSILON, 1.0 - FP32_EPSILON)
    mean = max_reward * normalized_mean
    q = torch.sigmoid(z_q)
    variance = q * mean * (max_reward - mean)
    kappa = torch.exp(-z_q)
    alpha = normalized_mean * kappa
    beta = (1.0 - normalized_mean) * kappa

    raw_clipped = old_predictions.float() + max_reward * torch.clamp(
        (mean - old_predictions.float()) / max_reward,
        -value_clip_epsilon,
        value_clip_epsilon,
    )
    clipped_normalized_mean = (raw_clipped / max_reward).clamp(FP32_EPSILON, 1.0 - FP32_EPSILON)
    clipped_values = max_reward * clipped_normalized_mean
    clipped_alpha = clipped_normalized_mean * kappa
    clipped_beta = (1.0 - clipped_normalized_mean) * kappa
    transformed_targets = beta_target_epsilon + (1.0 - 2.0 * beta_target_epsilon) * (value_targets.float() / max_reward)

    def beta_nll(alpha_value: torch.Tensor, beta_value: torch.Tensor) -> torch.Tensor:
        return -(
            (alpha_value - 1.0) * torch.log(transformed_targets)
            + (beta_value - 1.0) * torch.log1p(-transformed_targets)
            - (torch.lgamma(alpha_value) + torch.lgamma(beta_value) - torch.lgamma(alpha_value + beta_value))
        )

    current_loss = beta_nll(alpha, beta)
    clipped_loss = beta_nll(clipped_alpha, clipped_beta)
    return BetaValueLossComponents(
        mean=mean,
        q=q,
        variance=variance,
        kappa=kappa,
        alpha=alpha,
        beta=beta,
        clipped_values=clipped_values,
        clipped_alpha=clipped_alpha,
        clipped_beta=clipped_beta,
        current_loss=current_loss,
        clipped_loss=clipped_loss,
    )


def build_critic_context(
    prompt_ids: Sequence[int],
    critique_ids: Sequence[int],
    solution_ids: Sequence[int],
    *,
    critique_delimiter_ids: Sequence[int],
    solution_delimiter_ids: Sequence[int],
) -> CriticContext:
    if not prompt_ids or not critique_ids or not solution_ids:
        raise ValueError("critic prompt, critique, and solution token ranges must be non-empty")
    if not critique_delimiter_ids or not solution_delimiter_ids:
        raise ValueError("critic delimiters must tokenize to non-empty sequences")
    cursor = 0
    prompt_range = (cursor, cursor + len(prompt_ids))
    cursor = prompt_range[1]
    critique_delimiter_range = (cursor, cursor + len(critique_delimiter_ids))
    cursor = critique_delimiter_range[1]
    critique_range = (cursor, cursor + len(critique_ids))
    cursor = critique_range[1]
    solution_delimiter_range = (cursor, cursor + len(solution_delimiter_ids))
    cursor = solution_delimiter_range[1]
    solution_range = (cursor, cursor + len(solution_ids))
    token_ids = [
        *map(int, prompt_ids),
        *map(int, critique_delimiter_ids),
        *map(int, critique_ids),
        *map(int, solution_delimiter_ids),
        *map(int, solution_ids),
    ]
    context = CriticContext(
        token_ids=token_ids,
        prompt_range=prompt_range,
        critique_delimiter_range=critique_delimiter_range,
        critique_range=critique_range,
        solution_delimiter_range=solution_delimiter_range,
        solution_range=solution_range,
    )
    if [context.token_ids[position] for position in context.solution_positions] != list(solution_ids):
        raise RuntimeError("critic solution positions do not preserve exact actor solution token IDs")
    return context


def terminal_index(response_mask: torch.Tensor) -> torch.Tensor:
    """Return the last valid response index, including length-capped responses."""

    if response_mask.ndim != 2:
        raise ValueError("response_mask must have shape [batch, response_length]")
    lengths = response_mask.to(dtype=torch.long).sum(dim=-1)
    if torch.any(lengths <= 0):
        raise ValueError("every intermediate MC solution must contain at least one valid response token")
    return lengths - 1
