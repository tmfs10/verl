#!/usr/bin/env python3
"""Independent integrity checks for random-continuation evidence."""

from __future__ import annotations

import json
import math
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


def verify(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    audit_path = root / "audit" / "random_continuation_baseline.jsonl"
    summary_path = root / "audit" / "summary.json"
    config_path = root / "resolved_config.json"
    metrics_path = root / "metrics.jsonl"
    for path in (audit_path, summary_path, config_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = _load_jsonl(audit_path)
    configurations = [row for row in rows if row.get("event") == "configuration"]
    originals = [row for row in rows if row.get("event") == "original"]
    continuations = [row for row in rows if row.get("event") == "continuation"]
    summaries = [row for row in rows if row.get("event") == "summary"]
    if len(configurations) != 1 or len(summaries) != 1:
        raise ValueError("audit must contain exactly one configuration and summary event")
    configuration = configurations[0]
    expected_configuration = {
        "schema_version": 1,
        "points_per_rollout": 8,
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
    if len(originals) != 256:
        raise ValueError(f"expected 256 originals, got {len(originals)}")
    original_by_id = {str(row["rollout_id"]): row for row in originals}
    if len(original_by_id) != len(originals):
        raise ValueError("duplicate original rollout IDs")
    expected_marks: set[tuple[str, int]] = set()
    failed = 0
    for original in originals:
        solution_ids = [int(token) for token in original["solution_ids"]]
        editable_length = int(original["editable_solution_length"])
        if not 0 < editable_length <= len(solution_ids):
            raise ValueError("invalid editable original length")
        if float(original["original_reward"]) not in {0.0, 1.0}:
            raise ValueError("original reward is not binary")
        selection = original["selection"]
        marks = [int(mark) for mark in selection["marks"]]
        if marks != sorted(set(marks)) or len(marks) > 8:
            raise ValueError("selected marks are not a unique sorted set of at most eight")
        for mark in marks:
            if not int(selection["candidate_low"]) <= mark <= int(selection["candidate_high"]):
                raise ValueError("selected mark is outside audited bounds")
            if mark / editable_length <= 0.10 or not mark < editable_length or 8192 - mark < 128:
                raise ValueError("selected mark violates numeric eligibility")
            expected_marks.add((str(original["rollout_id"]), mark))
        failed += len(original["failures"])
    observed_marks: set[tuple[str, int]] = set()
    successes = 0
    recovery_attempts = recovery_successes = 0
    retention_attempts = retention_successes = 0
    for continuation in continuations:
        rollout_id = str(continuation["rollout_id"])
        original = original_by_id.get(rollout_id)
        if original is None:
            raise ValueError("continuation refers to an unknown original")
        mark = int(continuation["mark"])
        key = (rollout_id, mark)
        if key in observed_marks or key not in expected_marks:
            raise ValueError("continuation mark is duplicate or was not selected")
        observed_marks.add(key)
        prefix_ids = [int(token) for token in continuation["prefix_ids"]]
        if prefix_ids != [int(token) for token in original["solution_ids"][:mark]]:
            raise ValueError("continuation prefix does not match the original rollout")
        if branch_prefix_open_block_reason(str(continuation["prefix_text"])) is not None:
            raise ValueError("continuation begins inside a production structural block")
        suffix_ids = continuation["continuation_ids"]
        suffix_log_probs = continuation["continuation_log_probs"]
        if not suffix_ids or len(suffix_ids) != len(suffix_log_probs):
            raise ValueError("continuation tokens and log probabilities are empty or misaligned")
        if not all(math.isfinite(float(value)) for value in suffix_log_probs):
            raise ValueError("continuation has non-finite log probabilities")
        if int(continuation["continuation_max_tokens"]) != 8192 - mark:
            raise ValueError("continuation budget is not the remaining response budget")
        if mark + len(suffix_ids) > 8192:
            raise ValueError("completed response exceeds 8192 tokens")
        reward = float(continuation["reward"])
        if reward not in {0.0, 1.0}:
            raise ValueError("continuation reward is not binary")
        successes += int(reward)
        if bool(continuation["original_correct"]):
            retention_attempts += 1
            retention_successes += int(reward)
        else:
            recovery_attempts += 1
            recovery_successes += int(reward)
    if observed_marks | {
        (str(original["rollout_id"]), int(mark))
        for original in originals
        for mark, _sample, _reason in original["failures"]
    } != expected_marks:
        raise ValueError("selected marks are not conserved by successes and failures")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summaries[0]["generated_attempts"] != len(continuations) or summary["generated_attempts"] != len(continuations):
        raise ValueError("summary generated-attempt count is wrong")
    if summary["overall"]["successes"] != successes or summary["failed_generations"] != failed:
        raise ValueError("summary success/failure counts are wrong")
    if summary["recovery_original_incorrect"]["attempts"] != recovery_attempts:
        raise ValueError("summary recovery denominator is wrong")
    if summary["recovery_original_incorrect"]["successes"] != recovery_successes:
        raise ValueError("summary recovery numerator is wrong")
    if summary["retention_original_correct"]["attempts"] != retention_attempts:
        raise ValueError("summary retention denominator is wrong")
    if summary["retention_original_correct"]["successes"] != retention_successes:
        raise ValueError("summary retention numerator is wrong")
    config = json.loads(config_path.read_text(encoding="utf-8"))
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
        "originals": len(originals),
        "continuations": len(continuations),
        "successes": successes,
        "recovery": {"successes": recovery_successes, "attempts": recovery_attempts},
        "retention": {"successes": retention_successes, "attempts": retention_attempts},
        "failed_generations": failed,
    }
