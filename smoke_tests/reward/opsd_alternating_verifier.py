"""Deterministic verifier used only by the OPSD separate-teacher smoke test.

Each prompt must have exactly two sampled responses. The first receives reward
zero and the second reward one, guaranteeing non-uniform GRPO advantages while
remaining independent of generation quality. The submit wrapper deliberately
disables the async per-response reward loop so this function receives the full
rollout batch. This is intentionally a plumbing test; production experiments
should use their real verifier.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any


def _prompt_group(extra_info: dict[str, Any], fallback: int) -> str:
    value = extra_info.get("line_number", fallback)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    return str(value)


def compute_score(data_sources, solution_strs, ground_truths, extra_infos=None, **reward_kwargs):
    del ground_truths
    if extra_infos is None:
        raise ValueError("The OPSD alternating smoke verifier requires extra_info.line_number.")
    if not (len(solution_strs) == len(extra_infos)):
        raise ValueError("solution_strs and extra_infos must have identical lengths.")

    expected_group_size = int(reward_kwargs.get("opsd_smoke_group_size", 2))
    group_counts: dict[str, int] = defaultdict(int)
    results = []
    for index, (solution, extra_info) in enumerate(zip(solution_strs, extra_infos, strict=True)):
        group = _prompt_group(extra_info, index)
        slot = group_counts[group]
        group_counts[group] += 1
        reward = float(slot % 2)
        results.append(
            {
                "score": reward,
                "acc": reward,
                "smoke_reward_slot": float(slot),
                "pred": solution,
                "data_source": data_sources[index] if data_sources is not None else None,
            }
        )

    bad_groups = {group: count for group, count in group_counts.items() if count != expected_group_size}
    if bad_groups:
        raise ValueError(
            "The OPSD alternating smoke verifier expected exactly "
            f"{expected_group_size} responses per prompt, got {bad_groups}."
        )
    return results
