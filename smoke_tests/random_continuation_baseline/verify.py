#!/usr/bin/env python3
"""Streaming integrity checks for random-continuation evidence."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from verl.trainer.ppo.branch_revision_grpo import branch_prefix_open_block_reason


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        rows.append(value)
    return rows


def _close(actual: Any, expected: float, name: str) -> None:
    if actual is None or not math.isclose(float(actual), expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"{name}={actual!r}, expected {expected!r}")


def verify(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    audit_path = root / "audit" / "random_continuation_baseline.jsonl"
    summary_path = root / "audit" / "summary.json"
    config_path = root / "resolved_config.json"
    metrics_path = root / "metrics.jsonl"
    for path in (audit_path, summary_path, config_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    contract = config["random_continuation_run"]
    feature = config["algorithm"]["random_continuation_baseline"]
    prompts = int(contract["n_prompts"])
    rollouts = int(contract.get("rollouts_per_prompt", config["actor_rollout_ref"]["rollout"]["n"]))
    points = int(contract["points_per_rollout"])
    continuations_per_mark = int(contract.get("continuations_per_mark", feature.get("continuations_per_mark", 1)))
    expected_originals = prompts * rollouts
    expected_marks_count = expected_originals * points
    expected_continuations = expected_marks_count * continuations_per_mark

    configuration: dict[str, Any] | None = None
    summary_event: dict[str, Any] | None = None
    original_by_id: dict[str, dict[str, Any]] = {}
    samples_by_prompt: dict[str, set[int]] = defaultdict(set)
    originals_by_prompt: dict[str, list[str]] = defaultdict(list)
    expected_keys: set[tuple[str, int, int]] = set()
    failed_keys: set[tuple[str, int, int]] = set()
    observed_keys: set[tuple[str, int, int]] = set()
    continuation_rewards: dict[str, list[float]] = defaultdict(list)
    decile_counts = {False: Counter(), True: Counter()}
    successes = 0
    recovery_attempts = recovery_successes = 0
    retention_attempts = retention_successes = 0

    with audit_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{audit_path}:{line_number} is not an object")
            event = row.get("event")
            if event == "configuration":
                if configuration is not None:
                    raise ValueError("audit contains duplicate configuration events")
                configuration = row
                continue
            if event == "summary":
                if summary_event is not None:
                    raise ValueError("audit contains duplicate summary events")
                summary_event = row
                continue
            if event == "original":
                rollout_id = str(row["rollout_id"])
                if rollout_id in original_by_id:
                    raise ValueError("duplicate original rollout ID")
                solution_ids = [int(token) for token in row["solution_ids"]]
                editable_length = int(row["editable_solution_length"])
                if not 0 < editable_length <= len(solution_ids):
                    raise ValueError("invalid editable original length")
                reward = float(row["original_reward"])
                if reward not in {0.0, 1.0}:
                    raise ValueError("original reward is not binary")
                selection = row["selection"]
                marks = [int(mark) for mark in selection["marks"]]
                if marks != sorted(set(marks)) or len(marks) != points:
                    raise ValueError(f"original must have exactly {points} unique sorted marks")
                prompt_group_id = str(row["prompt_group_id"])
                sample_index = int(row.get("original_sample_index", 0))
                if sample_index in samples_by_prompt[prompt_group_id]:
                    raise ValueError("duplicate original sample index within a prompt")
                samples_by_prompt[prompt_group_id].add(sample_index)
                originals_by_prompt[prompt_group_id].append(rollout_id)
                for mark in marks:
                    if not int(selection["candidate_low"]) <= mark <= int(selection["candidate_high"]):
                        raise ValueError("selected mark is outside audited bounds")
                    if mark / editable_length <= 0.10 or not mark < editable_length or 8192 - mark < 128:
                        raise ValueError("selected mark violates numeric eligibility")
                    for continuation_index in range(continuations_per_mark):
                        expected_keys.add((rollout_id, mark, continuation_index))
                for mark, continuation_index, _reason in row["failures"]:
                    key = (rollout_id, int(mark), int(continuation_index))
                    if key in failed_keys:
                        raise ValueError("duplicate failed continuation identity")
                    failed_keys.add(key)
                original_by_id[rollout_id] = {
                    "prompt_group_id": prompt_group_id,
                    "sample_index": sample_index,
                    "solution_ids": solution_ids,
                    "editable_length": editable_length,
                    "reward": reward,
                }
                continue
            if event != "continuation":
                raise ValueError(f"unexpected audit event {event!r}")

            rollout_id = str(row["rollout_id"])
            original = original_by_id.get(rollout_id)
            if original is None:
                raise ValueError("continuation precedes or refers to an unknown original")
            mark = int(row["mark"])
            sample_index = int(row.get("sample_index", 0))
            key = (rollout_id, mark, sample_index)
            if key in observed_keys or key not in expected_keys:
                raise ValueError("continuation identity is duplicate or was not selected")
            observed_keys.add(key)
            prefix_ids = [int(token) for token in row["prefix_ids"]]
            if prefix_ids != original["solution_ids"][:mark]:
                raise ValueError("continuation prefix does not match the original rollout")
            if branch_prefix_open_block_reason(str(row["prefix_text"])) is not None:
                raise ValueError("continuation begins inside a production structural block")
            suffix_ids = row["continuation_ids"]
            suffix_log_probs = row["continuation_log_probs"]
            if not suffix_ids or len(suffix_ids) != len(suffix_log_probs):
                raise ValueError("continuation tokens and log probabilities are empty or misaligned")
            if not all(math.isfinite(float(value)) for value in suffix_log_probs):
                raise ValueError("continuation has non-finite log probabilities")
            if int(row["continuation_max_tokens"]) != 8192 - mark:
                raise ValueError("continuation budget is not the remaining response budget")
            if mark + len(suffix_ids) > 8192:
                raise ValueError("completed response exceeds 8192 tokens")
            reward = float(row["reward"])
            if reward not in {0.0, 1.0}:
                raise ValueError("continuation reward is not binary")
            continuation_rewards[rollout_id].append(reward)
            successes += int(reward)
            original_correct = bool(row["original_correct"])
            if original_correct:
                retention_attempts += 1
                retention_successes += int(reward)
            else:
                recovery_attempts += 1
                recovery_successes += int(reward)
            lower = min(9, int((mark / original["editable_length"]) * 10))
            decile_counts[original_correct][f"{lower / 10:.1f}-{(lower + 1) / 10:.1f}"] += 1

    if configuration is None or summary_event is None:
        raise ValueError("audit must contain one configuration and one summary event")
    expected_configuration = {
        "schema_version": 1,
        "points_per_rollout": points,
        "min_prefix_fraction": 0.10,
        "min_continuation_tokens": 128,
        "structural_boundaries_only": True,
        "temperature": 1.0,
        "max_prompt_length": 2048,
        "max_response_length": 8192,
        "max_model_len": 32768,
    }
    for key, expected in expected_configuration.items():
        if configuration.get(key) != expected:
            raise ValueError(f"configuration {key}={configuration.get(key)!r}, expected {expected!r}")
    if "rollouts_per_prompt" in contract:
        if configuration.get("rollouts_per_prompt") != rollouts:
            raise ValueError("configuration rollout multiplicity is wrong")
        if configuration.get("continuations_per_mark") != continuations_per_mark:
            raise ValueError("configuration continuation multiplicity is wrong")
    if len(original_by_id) != expected_originals or len(originals_by_prompt) != prompts:
        raise ValueError("audit original or prompt-group count is wrong")
    expected_sample_indices = set(range(rollouts))
    if any(samples != expected_sample_indices for samples in samples_by_prompt.values()):
        raise ValueError("each prompt must contain exactly the requested original sample indices")
    if len(expected_keys) != expected_continuations:
        raise ValueError("audit selected-mark cardinality is wrong")
    if failed_keys:
        raise ValueError(f"expected zero generation failures, got {len(failed_keys)}")
    if observed_keys != expected_keys:
        raise ValueError("selected continuation identities are not fully conserved")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary_event["generated_attempts"] != expected_continuations:
        raise ValueError("audit summary generated-attempt count is wrong")
    if summary["generated_attempts"] != expected_continuations or summary["selected_points"] != expected_marks_count:
        raise ValueError("summary cardinalities are wrong")
    if summary["overall"]["successes"] != successes or summary["failed_generations"] != 0:
        raise ValueError("summary success/failure counts are wrong")
    if summary["selection_shortfall"] != 0:
        raise ValueError("summary reports a selected-mark shortfall")
    if summary["recovery_original_incorrect"]["attempts"] != recovery_attempts:
        raise ValueError("summary recovery denominator is wrong")
    if summary["recovery_original_incorrect"]["successes"] != recovery_successes:
        raise ValueError("summary recovery numerator is wrong")
    if summary["retention_original_correct"]["attempts"] != retention_attempts:
        raise ValueError("summary retention denominator is wrong")
    if summary["retention_original_correct"]["successes"] != retention_successes:
        raise ValueError("summary retention numerator is wrong")
    for original_correct, summary_key in (
        (False, "success_by_prefix_decile_original_incorrect"),
        (True, "success_by_prefix_decile_original_correct"),
    ):
        observed = {key: int(value["attempts"]) for key, value in summary[summary_key].items()}
        if observed != dict(sorted(decile_counts[original_correct].items())):
            raise ValueError(f"{summary_key} attempt counts are wrong")

    if rollouts > 1:
        delta_by_prompt: dict[str, list[float]] = defaultdict(list)
        baseline_by_prompt: dict[str, list[float]] = defaultdict(list)
        for prompt_group_id, rollout_ids in originals_by_prompt.items():
            for rollout_id in rollout_ids:
                original = original_by_id[rollout_id]
                if original["reward"] != 0.0:
                    continue
                other_rewards = [original_by_id[other]["reward"] for other in rollout_ids if other != rollout_id]
                baseline = sum(other_rewards) / len(other_rewards)
                continuation_rate = sum(continuation_rewards[rollout_id]) / len(continuation_rewards[rollout_id])
                baseline_by_prompt[prompt_group_id].append(baseline)
                delta_by_prompt[prompt_group_id].append(continuation_rate - baseline)
        if delta_by_prompt:
            expected_baseline = sum(sum(values) / len(values) for values in baseline_by_prompt.values()) / len(
                baseline_by_prompt
            )
            expected_delta = sum(sum(values) / len(values) for values in delta_by_prompt.values()) / len(
                delta_by_prompt
            )
            comparison = summary["iid_comparison_original_incorrect"]
            _close(comparison["iid_expected_success"]["prompt_weighted"], expected_baseline, "IID baseline")
            _close(comparison["continuation_minus_iid"]["prompt_weighted"], expected_delta, "continuation-IID delta")

    if config["trainer"]["total_training_steps"] != 1 or config["trainer"]["save_freq"] != -1:
        raise ValueError("resolved config is not one-step/checkpoint-free")
    if config["critic"]["enable"] is not False:
        raise ValueError("resolved config enabled the critic")
    metric_rows = _load_jsonl(metrics_path)
    if len(metric_rows) != 1:
        raise ValueError(f"expected one metrics row, got {len(metric_rows)}")
    metric_data = metric_rows[0].get("data", {})
    if float(metric_data.get("random_continuation/actor_updated", -1)) != 0.0:
        raise ValueError("evaluation reports an actor update")
    forbidden = [key for key in metric_data if key.startswith("actor/") and ("loss" in key or "grad" in key)]
    if forbidden:
        raise ValueError(f"evaluation emitted actor optimizer metrics: {forbidden!r}")
    return {
        "status": "verified",
        "prompts": prompts,
        "originals": len(original_by_id),
        "marks": expected_marks_count,
        "continuations": len(observed_keys),
        "successes": successes,
        "recovery": {"successes": recovery_successes, "attempts": recovery_attempts},
        "retention": {"successes": retention_successes, "attempts": retention_attempts},
        "failed_generations": 0,
    }
