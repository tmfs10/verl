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
"""Selection and statistics for the random-prefix continuation baseline."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from verl.trainer.ppo.branch_revision_grpo import branch_prefix_open_block_reason, decode_exact


@dataclass(frozen=True)
class RandomMarkSelection:
    marks: tuple[int, ...]
    candidate_low: int
    candidate_high: int
    inspected: int
    rejection_counts: dict[str, int]


def stable_random(seed: int, *parts: object) -> random.Random:
    return random.Random(stable_seed(seed, *parts))


def stable_seed(seed: int, *parts: object) -> int:
    """Return a deterministic nonnegative per-request seed."""

    payload = "\x1f".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def strict_candidate_bounds(
    response_length: int,
    *,
    min_prefix_fraction: float,
    response_budget: int,
    min_continuation_tokens: int,
) -> tuple[int, int]:
    """Return bounds satisfying m/T > fraction, m < T, and suffix headroom."""

    if response_length < 2:
        return 1, 0
    low = max(1, math.floor(min_prefix_fraction * response_length) + 1)
    high = min(response_length - 1, response_budget - min_continuation_tokens)
    return low, high


def select_structurally_valid_random_marks(
    solution_ids: Sequence[int],
    *,
    tokenizer: Any,
    points_per_rollout: int,
    min_prefix_fraction: float,
    response_budget: int,
    min_continuation_tokens: int,
    rng: random.Random,
    structural_boundaries_only: bool = True,
    structural_reason_fn: Callable[[str], str | None] = branch_prefix_open_block_reason,
) -> RandomMarkSelection:
    """Uniformly sample valid marks without decoding every valid prefix.

    A uniformly shuffled candidate list is scanned until K valid positions are
    found. The first K valid elements of a uniform permutation are an exact
    uniform K-subset of the structurally valid candidate set.
    """

    low, high = strict_candidate_bounds(
        len(solution_ids),
        min_prefix_fraction=min_prefix_fraction,
        response_budget=response_budget,
        min_continuation_tokens=min_continuation_tokens,
    )
    if low > high:
        return RandomMarkSelection((), low, high, 0, {})
    candidates = list(range(low, high + 1))
    rng.shuffle(candidates)
    marks: list[int] = []
    rejection_counts: Counter[str] = Counter()
    inspected = 0
    for mark in candidates:
        inspected += 1
        if structural_boundaries_only:
            reason = structural_reason_fn(decode_exact(solution_ids[:mark], tokenizer))
            if reason is not None:
                rejection_counts[str(reason)] += 1
                continue
        marks.append(mark)
        if len(marks) == points_per_rollout:
            break
    return RandomMarkSelection(
        tuple(sorted(marks)),
        low,
        high,
        inspected,
        dict(sorted(rejection_counts.items())),
    )


def descriptive(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {"count": 0, "mean": None, "std": None, "min": None, "median": None, "max": None}
    result = {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }
    for name, quantile in (
        ("p01", 0.01),
        ("p05", 0.05),
        ("p10", 0.10),
        ("p25", 0.25),
        ("p75", 0.75),
        ("p90", 0.90),
        ("p95", 0.95),
        ("p99", 0.99),
    ):
        result[name] = float(np.quantile(array, quantile))
    return result


def clustered_rate(
    prompt_attempts: Sequence[Sequence[float]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, float | int | None]:
    """Return attempt/prompt rates and a prompt-cluster bootstrap interval."""

    clusters = [np.asarray(values, dtype=np.float64) for values in prompt_attempts if values]
    attempts = np.concatenate(clusters) if clusters else np.asarray([], dtype=np.float64)
    if not clusters:
        return {
            "prompts": 0,
            "attempts": 0,
            "successes": 0,
            "attempt_weighted": None,
            "prompt_weighted": None,
            "cluster_bootstrap_ci95_low": None,
            "cluster_bootstrap_ci95_high": None,
        }
    prompt_rates = np.asarray([cluster.mean() for cluster in clusters], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(clusters), size=(bootstrap_samples, len(clusters)))
    bootstrap = prompt_rates[draws].mean(axis=1)
    return {
        "prompts": len(clusters),
        "attempts": int(attempts.size),
        "successes": int(attempts.sum()),
        "attempt_weighted": float(attempts.mean()),
        "prompt_weighted": float(prompt_rates.mean()),
        "cluster_bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
        "cluster_bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
    }


def clustered_mean(
    prompt_values: Sequence[Sequence[float]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, float | int | None]:
    """Summarize arbitrary values with prompts as the bootstrap clusters."""

    clusters = [np.asarray(values, dtype=np.float64) for values in prompt_values if values]
    values = np.concatenate(clusters) if clusters else np.asarray([], dtype=np.float64)
    if not clusters:
        return {
            "prompts": 0,
            "observations": 0,
            "observation_weighted": None,
            "prompt_weighted": None,
            "cluster_bootstrap_ci95_low": None,
            "cluster_bootstrap_ci95_high": None,
        }
    prompt_means = np.asarray([cluster.mean() for cluster in clusters], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(clusters), size=(bootstrap_samples, len(clusters)))
    bootstrap = prompt_means[draws].mean(axis=1)
    return {
        "prompts": len(clusters),
        "observations": int(values.size),
        "observation_weighted": float(values.mean()),
        "prompt_weighted": float(prompt_means.mean()),
        "cluster_bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
        "cluster_bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
    }
