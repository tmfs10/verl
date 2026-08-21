#!/usr/bin/env python3
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
"""Verify one collected live branch-revision GRPO smoke run."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    if not rows:
        raise ValueError(f"empty JSONL evidence file: {path}")
    return rows


def _only(events: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [event for event in events if event.get("event") == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name!r} event, got {len(matches)}")
    return matches[0]


def _require_binary(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number not in {0.0, 1.0}:
        raise ValueError(f"{label} must be exactly binary, got {value!r}")
    return number


def verify(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    completed = _read_json(root / "completed.json")
    if completed.get("status") != "completed":
        raise ValueError(f"training did not complete: {completed!r}")
    if (root / "failed.json").exists():
        raise ValueError("failed.json exists beside completed.json")

    audit_files = sorted((root / "audit").glob("step_*.jsonl"))
    if len(audit_files) != 1:
        raise ValueError(f"expected exactly one step-scoped audit file, got {audit_files!r}")
    events = _read_jsonl(audit_files[0])
    event_counts = Counter(str(event.get("event")) for event in events)
    iteration = _only(events, "iteration")
    actor_batch = _only(events, "actor_batch")
    if int(iteration["originals"]) != 16:
        raise ValueError(f"expected 16 original rollouts, got {iteration['originals']!r}")
    original_rewards = [_require_binary(value, "original reward") for value in iteration["original_rewards"]]
    if len(original_rewards) != 16:
        raise ValueError("iteration audit must retain all 16 original binary rewards")
    incorrect = int(iteration["incorrect"])
    if incorrect != original_rewards.count(0.0) or incorrect <= 0:
        raise ValueError("smoke must contain and exactly count at least one incorrect original rollout")

    critiques = [event for event in events if event.get("event") == "critique"]
    continuations = [event for event in events if event.get("event") == "continuation"]
    expected_critiques = incorrect * 2
    if len(critiques) != expected_critiques:
        raise ValueError(f"expected {expected_critiques} IID critiques, got {len(critiques)}")
    if not continuations:
        raise ValueError("smoke produced no strictly parsed branch revision or continuation")

    critique_keys: set[tuple[str, int]] = set()
    valid_keys: set[tuple[str, int]] = set()
    per_rollout: defaultdict[str, set[int]] = defaultdict(set)
    for critique in critiques:
        key = (str(critique["rollout_id"]), int(critique["critique_index"]))
        if key in critique_keys:
            raise ValueError(f"duplicate critique evidence for {key!r}")
        critique_keys.add(key)
        per_rollout[key[0]].add(key[1])
        outcome = _require_binary(critique["continuation_outcome"], "critique continuation outcome")
        baseline = float(critique["prompt_pass_at_1"])
        reward = float(critique["reward"])
        if not 0.0 <= baseline <= 1.0 or not math.isclose(reward, outcome - baseline, abs_tol=1e-9):
            raise ValueError(
                f"critique {key!r} reward must equal continuation_outcome - prompt_pass_at_1; "
                f"got {reward!r}, {outcome!r}, {baseline!r}"
            )
        if critique["parse_reason"] == "valid":
            valid_keys.add(key)
            if not str(critique["branch"]).strip() or not str(critique["new_continuation"]).strip():
                raise ValueError(f"valid critique {key!r} omitted an edit boundary")
    if any(indices != {0, 1} for indices in per_rollout.values()):
        raise ValueError(f"each incorrect rollout must have critique indices 0 and 1: {dict(per_rollout)!r}")

    continuation_keys: set[tuple[str, int]] = set()
    for continuation in continuations:
        key = (str(continuation["rollout_id"]), int(continuation["critique_index"]))
        if key in continuation_keys:
            raise ValueError(f"duplicate continuation evidence for {key!r}")
        continuation_keys.add(key)
        _require_binary(continuation["reward"], "continuation reward")
        if not continuation["revised_prefix_ids"] or not continuation["continuation_ids"]:
            raise ValueError(f"continuation {key!r} lacks its revised prefix or generated suffix")
        if len(continuation["continuation_ids"]) != len(continuation["continuation_log_probs"]):
            raise ValueError(f"continuation {key!r} token/log-prob lengths differ")
    if continuation_keys != valid_keys:
        raise ValueError("valid critiques and continuation evidence must be one-to-one")

    expected_actor_rows = 16 + expected_critiques + len(continuations)
    expected_batch = {
        "rows": expected_actor_rows,
        "original": 16,
        "critiques": expected_critiques,
        "continuations": len(continuations),
        "policy_loss_mode": "dppo_tv",
    }
    for key, expected in expected_batch.items():
        if actor_batch.get(key) != expected:
            raise ValueError(f"actor batch {key} mismatch: {actor_batch.get(key)!r} != {expected!r}")
    padding = int(actor_batch["padding"])
    if not 0 <= padding < 8 or (expected_actor_rows + padding) % 8:
        raise ValueError(f"invalid data-parallel padding count: {padding}")

    metrics = _read_jsonl(root / "metrics.jsonl")
    step_rows = [row for row in metrics if int(row.get("step", -1)) == 1]
    if not step_rows:
        raise ValueError("file logger contains no global step 1 metrics")
    merged_metrics: dict[str, Any] = {}
    for row in step_rows:
        merged_metrics.update(row.get("data", {}))
    required_metrics = {
        "branch_revision/originals": 16.0,
        "branch_revision/incorrect_originals": float(incorrect),
        "branch_revision/critiques": float(expected_critiques),
        "branch_revision/valid_edits": float(len(continuations)),
        "branch_revision/continuations": float(len(continuations)),
        "branch_revision/policy_loss_is_dppo_tv": 1.0,
    }
    for key, expected in required_metrics.items():
        actual = float(merged_metrics.get(key, float("nan")))
        if actual != expected:
            raise ValueError(f"metric {key} mismatch: {actual!r} != {expected!r}")
    grad_keys = [key for key in merged_metrics if key.endswith("actor/grad_norm") or key == "actor/grad_norm"]
    if not grad_keys:
        raise ValueError("optimizer-step grad_norm metric is missing")
    if not any(math.isfinite(float(merged_metrics[key])) for key in grad_keys):
        raise ValueError("optimizer-step grad_norm metrics are all non-finite")

    return {
        "status": "verified",
        "audit_file": str(audit_files[0]),
        "event_counts": dict(sorted(event_counts.items())),
        "incorrect_originals": incorrect,
        "valid_edits": len(continuations),
        "successful_revisions": sum(float(event["reward"]) for event in continuations),
        "actor_rows": expected_actor_rows,
        "padding_rows": padding,
        "wall_seconds": float(completed["wall_seconds"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    result = verify(args.root)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
