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
    resolved_config = _read_json(root / "resolved_config.json")
    branch_config = resolved_config["algorithm"]["branch_revision_grpo"]
    num_critiques = int(branch_config["num_critiques"])
    if not bool(branch_config["enable_positive_compression"]):
        raise ValueError("live smoke did not enable positive-rollout compression")
    num_positive_critiques = int(branch_config["num_positive_critiques"])
    min_continuation_tokens = int(branch_config["min_continuation_tokens"])
    expected_originals = int(resolved_config["data"]["train_batch_size"]) * int(
        resolved_config["actor_rollout_ref"]["rollout"]["n"]
    )
    loss_mode = str(resolved_config["actor_rollout_ref"]["actor"]["policy_loss"]["loss_mode"])
    if loss_mode not in {"dppo_tv", "vanilla"}:
        raise ValueError(f"unsupported actor policy loss in smoke evidence: {loss_mode!r}")

    audit_files = sorted((root / "audit").glob("step_*.jsonl"))
    if len(audit_files) != 1:
        raise ValueError(f"expected exactly one step-scoped audit file, got {audit_files!r}")
    events = _read_jsonl(audit_files[0])
    event_counts = Counter(str(event.get("event")) for event in events)
    iteration = _only(events, "iteration")
    actor_batch = _only(events, "actor_batch")
    if int(iteration["originals"]) != expected_originals:
        raise ValueError(f"expected {expected_originals} original rollouts, got {iteration['originals']!r}")
    original_rewards = [_require_binary(value, "original reward") for value in iteration["original_rewards"]]
    if len(original_rewards) != expected_originals:
        raise ValueError(f"iteration audit must retain all {expected_originals} original binary rewards")
    incorrect = int(iteration["incorrect"])
    if incorrect != original_rewards.count(0.0) or incorrect <= 0:
        raise ValueError("smoke must contain and exactly count at least one incorrect original rollout")
    correct = int(iteration["correct"])
    if correct != original_rewards.count(1.0) or correct <= 0:
        raise ValueError("smoke must contain and exactly count at least one correct original rollout")
    audited_prompt_pass_at_1 = {str(key): float(value) for key, value in iteration.get("prompt_pass_at_1", {}).items()}
    expected_prompt_groups = expected_originals // int(resolved_config["actor_rollout_ref"]["rollout"]["n"])
    if len(audited_prompt_pass_at_1) != expected_prompt_groups or any(
        not 0.0 <= value <= 1.0 for value in audited_prompt_pass_at_1.values()
    ):
        raise ValueError("iteration audit must retain one valid original-rollout pass@1 per prompt")

    critiques = [event for event in events if event.get("event") == "critique"]
    continuations = [event for event in events if event.get("event") == "continuation"]
    learnability_events = [event for event in events if event.get("event") == "learnability"]
    expected_critiques = incorrect * num_critiques + correct * num_positive_critiques
    if len(critiques) != expected_critiques:
        raise ValueError(f"expected {expected_critiques} IID critiques, got {len(critiques)}")
    if not continuations:
        raise ValueError("smoke produced no strictly parsed branch revision or continuation")

    critique_keys: set[tuple[str, int]] = set()
    critique_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    structurally_valid_keys: set[tuple[str, int]] = set()
    accepted_keys: set[tuple[str, int]] = set()
    per_rollout: defaultdict[str, set[int]] = defaultdict(set)
    rollout_objectives: dict[str, str] = {}
    critique_prompts: defaultdict[str, set[tuple[int, ...]]] = defaultdict(set)
    for critique in critiques:
        key = (str(critique["rollout_id"]), int(critique["critique_index"]))
        if key in critique_keys:
            raise ValueError(f"duplicate critique evidence for {key!r}")
        critique_keys.add(key)
        critique_by_key[key] = critique
        per_rollout[key[0]].add(key[1])
        objective = str(critique.get("objective"))
        if objective not in {"recovery", "compression"}:
            raise ValueError(f"critique {key!r} has invalid objective {objective!r}")
        previous_objective = rollout_objectives.setdefault(key[0], objective)
        if previous_objective != objective:
            raise ValueError(f"critique group {key[0]!r} mixes objectives")
        critique_prompt_ids = tuple(int(token) for token in critique.get("critique_prompt_ids", ()))
        critique_ids = tuple(int(token) for token in critique.get("critique_ids", ()))
        critique_log_probs = tuple(float(value) for value in critique.get("critique_log_probs", ()))
        if not critique_prompt_ids:
            raise ValueError(f"critique {key!r} omitted its exact behavior-policy prompt IDs")
        if not critique_ids or len(critique_ids) != len(critique_log_probs):
            raise ValueError(f"critique {key!r} token/log-prob lengths differ")
        critique_prompts[key[0]].add(critique_prompt_ids)
        outcome = _require_binary(critique["continuation_outcome"], "critique continuation outcome")
        prompt_group_id = str(critique.get("prompt_group_id"))
        baseline = float(critique["prompt_pass_at_1"])
        reward = float(critique["reward"])
        learnability_weight = float(critique["learnability_weight"])
        accepted = bool(critique["learnability_accepted"])
        objective_credit = float(critique["objective_credit"])
        if not 0.0 <= learnability_weight <= 1.0:
            raise ValueError(f"critique {key!r} has invalid learnability weight")
        if prompt_group_id not in audited_prompt_pass_at_1 or not math.isclose(
            baseline, audited_prompt_pass_at_1[prompt_group_id], abs_tol=1e-9
        ):
            raise ValueError(f"critique {key!r} uses a pass@1 from the wrong original prompt group")
        expected_reward = (
            outcome * learnability_weight - baseline
            if objective == "recovery"
            else objective_credit * learnability_weight
        )
        if not math.isclose(reward, expected_reward, abs_tol=1e-9):
            raise ValueError(
                f"critique {key!r} reward does not match its objective and learnability credit; "
                f"got {reward!r}, expected {expected_reward!r}"
            )
        if objective == "recovery" and not math.isclose(objective_credit, outcome, abs_tol=1e-9):
            raise ValueError(f"recovery critique {key!r} objective credit differs from its binary outcome")
        if objective == "compression" and not math.isclose(
            objective_credit,
            float(critique.get("compression_credit") or 0.0),
            abs_tol=1e-9,
        ):
            raise ValueError(f"compression critique {key!r} objective credit differs from its length credit")
        if critique["parse_reason"] == "valid":
            structurally_valid_keys.add(key)
            if not str(critique["branch"]).strip() or not str(critique["new_continuation"]).strip():
                raise ValueError(f"valid critique {key!r} omitted an edit boundary")
            if accepted:
                accepted_keys.add(key)
        elif accepted or learnability_weight != 0.0:
            raise ValueError(f"structurally invalid critique {key!r} received learnability credit")
    for rollout_id, indices in per_rollout.items():
        count = num_critiques if rollout_objectives[rollout_id] == "recovery" else num_positive_critiques
        expected_indices = set(range(count))
        if indices != expected_indices:
            raise ValueError(
                f"rollout {rollout_id!r} must have critique indices {sorted(expected_indices)}; got {sorted(indices)}"
            )
    if Counter(rollout_objectives.values()) != Counter(recovery=incorrect, compression=correct):
        raise ValueError("critique objective counts do not match original correctness counts")
    if any(len(prompts) != 1 for prompts in critique_prompts.values()):
        raise ValueError("IID critiques for one original rollout used different behavior-policy prompt IDs")

    learnability_by_key = {
        (str(event["rollout_id"]), int(event["critique_index"])): event for event in learnability_events
    }
    if len(learnability_by_key) != len(learnability_events):
        raise ValueError("duplicate learnability evidence")
    if set(learnability_by_key) != structurally_valid_keys:
        raise ValueError("every structurally valid edit must have exactly one learnability assessment")
    for key, event in learnability_by_key.items():
        if event.get("score_source") != "vllm_prompt_logprobs":
            raise ValueError(f"learnability event {key!r} did not use vLLM prompt log probabilities")
        if not 0.0 <= float(event["percentile"]) <= 1.0 or not 0.0 <= float(event["reward_weight"]) <= 1.0:
            raise ValueError(f"learnability event {key!r} has an invalid percentile or weight")
        if int(event["seed_tokens"]) <= 0:
            raise ValueError(f"learnability event {key!r} has no replacement seed")
        if bool(event["accepted"]) and int(event["sampled_windows"]) <= 0:
            raise ValueError(f"accepted learnability event {key!r} lacks a length-matched reference")
        critique = critique_by_key[key]
        if bool(event["accepted"]) != bool(critique["learnability_accepted"]) or not math.isclose(
            float(event["reward_weight"]), float(critique["learnability_weight"]), abs_tol=1e-9
        ):
            raise ValueError(f"learnability event {key!r} differs from its trained critique reward")

    continuation_keys: set[tuple[str, int]] = set()
    for continuation in continuations:
        key = (str(continuation["rollout_id"]), int(continuation["critique_index"]))
        if key in continuation_keys:
            raise ValueError(f"duplicate continuation evidence for {key!r}")
        continuation_keys.add(key)
        continuation_reward = _require_binary(continuation["reward"], "continuation reward")
        objective = str(continuation.get("objective"))
        if objective != rollout_objectives.get(key[0]):
            raise ValueError(f"continuation {key!r} objective differs from its critique group")
        if not continuation["revised_prefix_ids"] or not continuation["continuation_ids"]:
            raise ValueError(f"continuation {key!r} lacks its revised prefix or generated suffix")
        if len(continuation["continuation_ids"]) != len(continuation["continuation_log_probs"]):
            raise ValueError(f"continuation {key!r} token/log-prob lengths differ")
        if int(continuation.get("continuation_max_tokens", 0)) < min_continuation_tokens:
            raise ValueError(f"continuation {key!r} did not receive its configured minimum token budget")
        if objective == "compression":
            fraction = float(continuation["compression_fraction"])
            credit = float(continuation["compression_credit"])
            target = float(branch_config["positive_compression_target"])
            if fraction < 0.0 or not math.isclose(credit, continuation_reward * min(fraction / target, 1.0)):
                raise ValueError(f"compression continuation {key!r} has inconsistent length credit")
    if continuation_keys != accepted_keys:
        raise ValueError("learnability-accepted critiques and rewarded continuation evidence must be one-to-one")

    expected_actor_rows = expected_originals + expected_critiques + len(continuations)
    expected_batch = {
        "rows": expected_actor_rows,
        "original": expected_originals,
        "critiques": expected_critiques,
        "continuations": len(continuations),
        "policy_loss_mode": loss_mode,
    }
    for key, expected in expected_batch.items():
        if actor_batch.get(key) != expected:
            raise ValueError(f"actor batch {key} mismatch: {actor_batch.get(key)!r} != {expected!r}")
    actor_rows = actor_batch.get("actor_rows")
    if not isinstance(actor_rows, list) or len(actor_rows) != expected_actor_rows:
        raise ValueError("actor-batch audit must retain every non-padding row's kind, group, and reward")
    actor_kind_counts = Counter(str(row.get("kind")) for row in actor_rows)
    if actor_kind_counts != Counter(
        original=expected_originals,
        critique=expected_critiques,
        continuation=len(continuations),
    ):
        raise ValueError(f"actor-row kind counts mismatch: {actor_kind_counts!r}")
    original_actor_rows = [row for row in actor_rows if row["kind"] == "original"]
    critique_actor_rows = [row for row in actor_rows if row["kind"] == "critique"]
    continuation_actor_rows = [row for row in actor_rows if row["kind"] == "continuation"]
    if [float(row["reward"]) for row in original_actor_rows] != original_rewards:
        raise ValueError("original actor rows do not retain their binary environment outcomes")
    continuation_actor_rewards = sorted(
        _require_binary(row["reward"], "continuation actor reward") for row in continuation_actor_rows
    )
    continuation_audit_rewards = sorted(float(event["reward"]) for event in continuations)
    if continuation_actor_rewards != continuation_audit_rewards:
        raise ValueError("revised solution actor rows must use their binary continuation outcomes")
    critique_actor_rewards = sorted(float(row["reward"]) for row in critique_actor_rows)
    critique_audit_rewards = sorted(float(event["reward"]) for event in critiques)
    if len(critique_actor_rewards) != len(critique_audit_rewards) or any(
        not math.isclose(actual, expected, abs_tol=1e-9)
        for actual, expected in zip(critique_actor_rewards, critique_audit_rewards, strict=True)
    ):
        raise ValueError("critique actor rows do not use their audited objective rewards")
    original_solution_groups = Counter(str(row["group_id"]) for row in original_actor_rows)
    rollout_n = int(resolved_config["actor_rollout_ref"]["rollout"]["n"])
    expected_prompt_groups = expected_originals // rollout_n
    if len(original_solution_groups) != expected_prompt_groups or any(
        count != rollout_n for count in original_solution_groups.values()
    ):
        raise ValueError(
            f"original solution GRPO groups must contain {rollout_n} rollouts per prompt: {original_solution_groups!r}"
        )
    if not any(
        len({float(row["reward"]) for row in original_actor_rows if str(row["group_id"]) == group_id}) > 1
        for group_id in original_solution_groups
    ):
        raise ValueError("smoke has no nonuniform original-solution GRPO reward group")
    recomputed_prompt_pass_at_1: dict[str, float] = {}
    for group_id in original_solution_groups:
        prompt_group_id = group_id.removeprefix("solution:")
        group_rewards = [float(row["reward"]) for row in original_actor_rows if str(row["group_id"]) == group_id]
        recomputed_prompt_pass_at_1[prompt_group_id] = sum(group_rewards) / len(group_rewards)
    if recomputed_prompt_pass_at_1 != audited_prompt_pass_at_1:
        raise ValueError("audited prompt pass@1 values do not match original solution outcomes")
    if any(str(row["group_id"]) not in original_solution_groups for row in continuation_actor_rows):
        raise ValueError("revised solutions must join their original prompt's solution GRPO group")
    critique_groups = Counter(str(row["group_id"]) for row in critique_actor_rows)
    if len(critique_groups) != incorrect + correct:
        raise ValueError("every selected original must have one critique GRPO group")
    for group_id, count in critique_groups.items():
        rollout_id = group_id.removeprefix("critique:")
        expected_count = num_critiques if rollout_objectives[rollout_id] == "recovery" else num_positive_critiques
        if count != expected_count:
            raise ValueError(f"critique group {group_id!r} must contain {expected_count} IID critiques, got {count}")
    for objective in ("recovery", "compression"):
        objective_groups = [
            group_id
            for group_id in critique_groups
            if rollout_objectives[group_id.removeprefix("critique:")] == objective
        ]
        if not any(
            len({float(row["reward"]) for row in critique_actor_rows if str(row["group_id"]) == group_id}) > 1
            for group_id in objective_groups
        ):
            raise ValueError(f"smoke has no nonuniform {objective} critique GRPO reward group")
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
        "branch_revision/originals": float(expected_originals),
        "branch_revision/incorrect_originals": float(incorrect),
        "branch_revision/correct_originals": float(correct),
        "branch_revision/critiques": float(expected_critiques),
        "branch_revision/recovery_critiques": float(incorrect * num_critiques),
        "branch_revision/compression_critiques": float(correct * num_positive_critiques),
        "branch_revision/valid_edits": float(len(structurally_valid_keys)),
        "branch_revision/learnability_accepted_edits": float(len(accepted_keys)),
        "branch_revision/continuations": float(len(continuations)),
        "branch_revision/policy_loss_is_dppo_tv": float(loss_mode == "dppo_tv"),
    }
    for key, expected in required_metrics.items():
        actual = float(merged_metrics.get(key, float("nan")))
        if actual != expected:
            raise ValueError(f"metric {key} mismatch: {actual!r} != {expected!r}")
    grad_keys = [key for key in merged_metrics if key.endswith("actor/grad_norm") or key == "actor/grad_norm"]
    if not grad_keys:
        raise ValueError("optimizer-step grad_norm metric is missing")
    finite_grad_norms = [float(merged_metrics[key]) for key in grad_keys if math.isfinite(float(merged_metrics[key]))]
    if not finite_grad_norms or not any(value > 0.0 for value in finite_grad_norms):
        raise ValueError("optimizer-step grad_norm metrics contain no finite positive learning signal")
    pg_loss = float(merged_metrics.get("actor/pg_loss", float("nan")))
    if not math.isfinite(pg_loss):
        raise ValueError("optimizer-step actor/pg_loss metric is missing or non-finite")
    successful_revisions = sum(float(event["reward"]) for event in continuations)
    if successful_revisions <= 0.0:
        raise ValueError("smoke has no successful revised continuation")
    successful_recoveries = sum(
        float(event["reward"]) for event in continuations if event.get("objective") == "recovery"
    )
    successful_compressions = sum(
        float(event.get("compression_credit") or 0.0)
        for event in continuations
        if event.get("objective") == "compression"
    )
    if successful_recoveries <= 0.0:
        raise ValueError("smoke has no successful recovery continuation")
    if successful_compressions <= 0.0:
        raise ValueError("smoke has no successful positive-rollout compression")

    return {
        "status": "verified",
        "audit_file": str(audit_files[0]),
        "event_counts": dict(sorted(event_counts.items())),
        "incorrect_originals": incorrect,
        "correct_originals": correct,
        "valid_edits": len(structurally_valid_keys),
        "learnability_accepted_edits": len(accepted_keys),
        "successful_revisions": successful_revisions,
        "successful_compression_credit": successful_compressions,
        "policy_loss_mode": loss_mode,
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
