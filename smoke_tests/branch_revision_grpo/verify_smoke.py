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
"""Verify one collected multi-step branch-revision GRPO stress smoke run."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_AUDIT_SCHEMA_VERSION = 5
_SUPPORTED_AUDIT_SCHEMA_VERSIONS = {2, 3, 4, _AUDIT_SCHEMA_VERSION}


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


def _canonical_sha256(values: Any, *, dtype: str) -> str:
    array = np.asarray(values, dtype=np.dtype(dtype))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _optional_float_matches(actual: object, expected: float | None, *, abs_tol: float = 1e-12) -> bool:
    if expected is None:
        return actual is None
    try:
        value = float(actual)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and math.isclose(value, expected, rel_tol=0.0, abs_tol=abs_tol)


def _float32_values(values: Any) -> list[float]:
    result = np.asarray(list(values), dtype=np.float32)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError("audited log probabilities must be a finite one-dimensional sequence")
    return [float(value) for value in result.tolist()]


def _aggregate(values: Any, statistic: str) -> float:
    normalized = _float32_values(values)
    if not normalized:
        raise ValueError("cannot aggregate empty audited log probabilities")
    if statistic == "mean":
        return math.fsum(normalized) / len(normalized)
    if statistic == "min":
        return min(normalized)
    raise ValueError(f"unsupported learnability log-probability statistic: {statistic!r}")


def _exhaustive_reference(
    originals: list[dict[str, Any]],
    *,
    window_size: int,
    statistic: str,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    counts: list[dict[str, Any]] = []
    rows: list[np.ndarray] = []
    for original in originals:
        editable_length = int(original["editable_solution_length"])
        values = np.asarray(
            _float32_values(original["solution_log_probs"][:editable_length]),
            dtype=np.float32,
        )
        candidate_count = int(values.size) - window_size + 1
        if candidate_count <= 0:
            continue
        if statistic == "mean":
            prefix = np.concatenate([np.zeros(1, dtype=np.float64), np.cumsum(values, dtype=np.float64)])
            scores = (prefix[window_size:] - prefix[:-window_size]) / float(window_size)
        elif statistic == "min":
            scores = np.lib.stride_tricks.sliding_window_view(values, window_size).min(axis=1).astype(np.float64)
        else:
            raise ValueError(f"unsupported learnability log-probability statistic: {statistic!r}")
        if scores.size != candidate_count:
            raise ValueError("exhaustive audit reconstruction produced an incorrect window count")
        counts.append({"rollout_id": str(original["rollout_id"]), "windows": candidate_count})
        rows.append(np.asarray(scores, dtype=np.float64))
    if not rows:
        return counts, np.empty(0, dtype=np.float64)
    return counts, np.concatenate(rows)


def _expected_runtime_config(saved_config: dict[str, Any]) -> dict[str, Any]:
    """Apply the native, deterministic mutations made before the first audit event."""

    expected = copy.deepcopy(saved_config)
    trainer = expected.get("trainer", {})
    reward_model = expected.get("reward", {}).get("reward_model")
    if isinstance(reward_model, dict) and not bool(reward_model.get("enable_resource_pool", False)):
        if "nnodes" in trainer and "nnodes" in reward_model:
            reward_model["nnodes"] = trainer["nnodes"]
        if "n_gpus_per_node" in trainer and "n_gpus_per_node" in reward_model:
            reward_model["n_gpus_per_node"] = trainer["n_gpus_per_node"]
    total_training_steps = trainer.get("total_training_steps")
    if total_training_steps is not None:
        actor_optim = expected.get("actor_rollout_ref", {}).get("actor", {}).get("optim")
        if isinstance(actor_optim, dict) and "total_training_steps" in actor_optim:
            actor_optim["total_training_steps"] = total_training_steps
        critic_optim = expected.get("critic", {}).get("optim")
        if isinstance(critic_optim, dict) and "total_training_steps" in critic_optim:
            critic_optim["total_training_steps"] = total_training_steps
    return expected


def _verify_prompt_logprob_admission_step(
    events: list[dict[str, Any]],
    *,
    capacity: int | None,
) -> tuple[int, int, int]:
    """Validate every admission record in one step and return its pressure key."""

    learnability_events = [event for event in events if event.get("event") == "learnability"]
    summaries = [event for event in events if event.get("event") == "prompt_logprob_admission_summary"]
    if capacity is None:
        declared_tokens = sum(int(event.get("prompt_tokens", 0)) for event in summaries)
        return declared_tokens, 0, len(learnability_events)

    admissions: list[dict[str, Any]] = []
    request_keys: set[tuple[str, int]] = set()
    for event in learnability_events:
        admission = event.get("prompt_logprob_admission")
        if not isinstance(admission, dict):
            raise ValueError("learnability event omitted prompt-logprob admission evidence")
        scoring_prompt_ids = event.get("scoring_prompt_ids")
        if not isinstance(scoring_prompt_ids, list) or not scoring_prompt_ids:
            raise ValueError("learnability event omitted its scoring prompt")
        prompt_tokens = len(scoring_prompt_ids)
        charged_tokens = min(prompt_tokens, capacity)
        oversized = prompt_tokens > capacity
        server_id = admission.get("server_id")
        numeric_fields = (
            "request_sequence",
            "prompt_tokens",
            "charged_tokens",
            "inflight_prompt_tokens_at_grant",
            "inflight_charged_tokens_at_grant",
            "high_water_prompt_tokens",
            "high_water_charged_tokens",
        )
        try:
            numeric = {name: int(admission[name]) for name in numeric_fields}
            admitted_capacity = int(admission["capacity"])
            wait_seconds = float(admission["wait_seconds"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("learnability event has incomplete prompt-logprob admission evidence") from error
        if (
            not isinstance(server_id, str)
            or not server_id
            or admitted_capacity != capacity
            or numeric["request_sequence"] <= 0
            or numeric["prompt_tokens"] != prompt_tokens
            or numeric["charged_tokens"] != charged_tokens
            or numeric["inflight_prompt_tokens_at_grant"] < prompt_tokens
            or numeric["inflight_charged_tokens_at_grant"] < charged_tokens
            or numeric["inflight_charged_tokens_at_grant"] > capacity
            or numeric["high_water_prompt_tokens"] < numeric["inflight_prompt_tokens_at_grant"]
            or numeric["high_water_charged_tokens"] < numeric["inflight_charged_tokens_at_grant"]
            or numeric["high_water_charged_tokens"] > capacity
            or not isinstance(admission.get("oversized"), bool)
            or admission["oversized"] != oversized
            or not math.isfinite(wait_seconds)
            or wait_seconds < 0.0
        ):
            raise ValueError("learnability event violates the prompt-logprob admission budget")
        if oversized and (
            numeric["inflight_prompt_tokens_at_grant"] != prompt_tokens
            or numeric["inflight_charged_tokens_at_grant"] != capacity
        ):
            raise ValueError("oversized learnability request did not run alone")
        request_key = (server_id, numeric["request_sequence"])
        if request_key in request_keys:
            raise ValueError("step reused a per-server prompt-logprob admission request sequence")
        request_keys.add(request_key)
        admissions.append(admission)

    if not admissions:
        if summaries:
            raise ValueError("prompt-logprob admission summary exists without any scored edit")
        return 0, 0, 0

    if len(summaries) != 1:
        raise ValueError(f"expected exactly one prompt-logprob admission summary, got {len(summaries)}")
    summary_event = summaries[0]
    if (
        int(summary_event.get("capacity", -1)) != capacity
        or int(summary_event.get("requests", -1)) != len(admissions)
        or int(summary_event.get("prompt_tokens", -1)) != sum(int(item["prompt_tokens"]) for item in admissions)
    ):
        raise ValueError("prompt-logprob admission summary has incorrect global totals")

    expected_per_server: dict[str, dict[str, int | float]] = {}
    for admission in admissions:
        server_id = str(admission["server_id"])
        server = expected_per_server.setdefault(
            server_id,
            {
                "requests": 0,
                "prompt_tokens": 0,
                "max_inflight_prompt_tokens": 0,
                "max_inflight_charged_tokens": 0,
                "max_wait_seconds": 0.0,
            },
        )
        server["requests"] = int(server["requests"]) + 1
        server["prompt_tokens"] = int(server["prompt_tokens"]) + int(admission["prompt_tokens"])
        server["max_inflight_prompt_tokens"] = max(
            int(server["max_inflight_prompt_tokens"]),
            int(admission["high_water_prompt_tokens"]),
        )
        server["max_inflight_charged_tokens"] = max(
            int(server["max_inflight_charged_tokens"]),
            int(admission["high_water_charged_tokens"]),
        )
        server["max_wait_seconds"] = max(
            float(server["max_wait_seconds"]),
            float(admission["wait_seconds"]),
        )
    if summary_event.get("per_server") != expected_per_server:
        raise ValueError("prompt-logprob admission summary does not match per-request evidence")
    total_prompt_tokens = sum(int(item["prompt_tokens"]) for item in admissions)
    max_inflight_prompt_tokens = max(int(item["high_water_prompt_tokens"]) for item in admissions)
    return total_prompt_tokens, max_inflight_prompt_tokens, len(admissions)


def verify(root: Path, *, require_algorithm_signal: bool = True) -> dict[str, Any]:
    root = root.expanduser().resolve()
    status = _read_json(root / "status.json")
    if status.get("status") != "completed":
        raise ValueError(f"current smoke invocation did not complete: {status!r}")
    completed = _read_json(root / "completed.json")
    if completed.get("status") != "completed":
        raise ValueError(f"training did not complete: {completed!r}")
    if completed.get("invocation_id") != status.get("invocation_id"):
        raise ValueError("completed and status evidence refer to different smoke invocations")
    failed_path = root / "failed.json"
    if failed_path.exists() and _read_json(failed_path).get("invocation_id") == status.get("invocation_id"):
        raise ValueError("the current completed smoke invocation also has failure evidence")
    resolved_config = _read_json(root / "resolved_config.json")
    branch_config = resolved_config["algorithm"]["branch_revision_grpo"]
    num_critiques = int(branch_config["num_critiques"])
    if not bool(branch_config["enable_positive_compression"]):
        raise ValueError("live smoke did not enable positive-rollout compression")
    num_positive_critiques = int(branch_config["num_positive_critiques"])
    min_continuation_tokens = int(branch_config["min_continuation_tokens"])
    statistic = str(branch_config["learnability_logprob_statistic"])
    if statistic not in {"mean", "min"}:
        raise ValueError(f"unsupported learnability statistic in smoke evidence: {statistic!r}")
    expected_originals = int(resolved_config["data"]["train_batch_size"]) * int(
        resolved_config["actor_rollout_ref"]["rollout"]["n"]
    )
    prompt_logprob_capacity = resolved_config["actor_rollout_ref"]["rollout"].get("prompt_logprob_max_inflight_tokens")
    if prompt_logprob_capacity is not None:
        prompt_logprob_capacity = int(prompt_logprob_capacity)
        if prompt_logprob_capacity <= 0:
            raise ValueError("smoke evidence has an invalid prompt-logprob admission capacity")
    loss_mode = str(resolved_config["actor_rollout_ref"]["actor"]["policy_loss"]["loss_mode"])
    if loss_mode not in {"dppo_tv", "vanilla"}:
        raise ValueError(f"unsupported actor policy loss in smoke evidence: {loss_mode!r}")

    attempt_id = str(completed.get("audit_attempt_id", ""))
    if not attempt_id:
        raise ValueError("completed smoke evidence omitted its audit attempt ID")
    attempt_dir = root / "audit" / f"attempt_{attempt_id}"
    attempt = _read_json(attempt_dir / "attempt.json")
    audit_schema_version = int(attempt.get("schema_version", -1))
    if audit_schema_version not in _SUPPORTED_AUDIT_SCHEMA_VERSIONS or attempt.get("attempt_id") != attempt_id:
        raise ValueError("audit attempt metadata has the wrong schema or attempt ID")
    threshold_mode = (
        str(branch_config.get("learnability_threshold_mode", "percentile"))
        if audit_schema_version >= 5
        else "percentile"
    )
    if threshold_mode not in {"stddev", "percentile"}:
        raise ValueError(f"unsupported learnability threshold mode in smoke evidence: {threshold_mode!r}")
    max_seed_window_stddevs = float(branch_config.get("max_seed_window_stddevs", 15.0))
    if not math.isfinite(max_seed_window_stddevs) or max_seed_window_stddevs < 0.0:
        raise ValueError("smoke evidence has an invalid standard-deviation cutoff")
    runtime_config = attempt.get("resolved_config")
    if not isinstance(runtime_config, dict):
        raise ValueError("audit attempt metadata omitted its exact runtime configuration")
    runtime_config_json = json.dumps(runtime_config, sort_keys=True, default=str, ensure_ascii=False)
    if attempt.get("resolved_config_sha256") != hashlib.sha256(runtime_config_json.encode("utf-8")).hexdigest():
        raise ValueError("audit attempt runtime configuration does not match its recorded hash")
    if runtime_config != _expected_runtime_config(resolved_config):
        raise ValueError("audit attempt metadata does not match the resolved configuration")
    audit_files = sorted(attempt_dir.glob("step_*.jsonl"))
    expected_training_steps = int(resolved_config["trainer"].get("total_training_steps", 1))
    starting_global_step = int(attempt.get("starting_global_step", -1))
    if not 1 <= starting_global_step <= expected_training_steps:
        raise ValueError("audit attempt starting_global_step must fall inside the configured training-step range")
    expected_step_numbers = list(range(starting_global_step, expected_training_steps + 1))
    actual_step_numbers: list[int] = []
    audited_steps: list[tuple[Path, list[dict[str, Any]], tuple[int, int, int]]] = []
    for audit_file in audit_files:
        step_events = _read_jsonl(audit_file)
        try:
            filename_step = int(audit_file.stem.removeprefix("step_"))
        except ValueError as error:
            raise ValueError(f"invalid step audit filename: {audit_file}") from error
        actual_step_numbers.append(filename_step)
        if any(
            int(event.get("schema_version", -1)) != audit_schema_version
            or event.get("attempt_id") != attempt_id
            or int(event.get("global_step", -1)) != filename_step
            for event in step_events
        ):
            raise ValueError("step audit mixes schema versions, attempt IDs, or global steps")
        if (
            step_events[-1].get("event") != "step_complete"
            or sum(event.get("event") == "step_complete" for event in step_events) != 1
        ):
            raise ValueError("selected audit step is incomplete")
        _only(step_events, "iteration")
        _only(step_events, "actor_batch")
        pressure = _verify_prompt_logprob_admission_step(
            step_events,
            capacity=prompt_logprob_capacity,
        )
        audited_steps.append((audit_file, step_events, pressure))
    if actual_step_numbers != expected_step_numbers:
        raise ValueError(
            "completed step-scoped audit files do not match the attempt range: "
            f"expected={expected_step_numbers!r} actual={actual_step_numbers!r}"
        )

    # Deeply validate the completed step with the heaviest prompt-logprob
    # workload. Every step's admission evidence was already validated above;
    # the selected step receives the more expensive end-to-end reconstruction.
    selected_audit_file, events, _ = max(
        audited_steps,
        key=lambda item: item[2],
    )
    selected_step = int(events[0]["global_step"])
    event_counts = Counter(str(event.get("event")) for event in events)
    iteration = _only(events, "iteration")
    actor_batch = _only(events, "actor_batch")
    if int(iteration["originals"]) != expected_originals:
        raise ValueError(f"expected {expected_originals} original rollouts, got {iteration['originals']!r}")
    original_rewards = [_require_binary(value, "original reward") for value in iteration["original_rewards"]]
    if len(original_rewards) != expected_originals:
        raise ValueError(f"iteration audit must retain all {expected_originals} original binary rewards")
    if iteration.get("learnability_logprob_statistic") != statistic:
        raise ValueError("iteration audit used a different learnability statistic than the resolved config")
    if audit_schema_version >= 5 and (
        iteration.get("learnability_threshold_mode") != threshold_mode
        or not math.isclose(
            float(iteration.get("max_seed_window_stddevs", float("nan"))),
            max_seed_window_stddevs,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("iteration audit used a different learnability threshold than the resolved config")
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

    originals = [event for event in events if event.get("event") == "original"]
    if len(originals) != expected_originals:
        raise ValueError("audit must contain one source event per original rollout")
    original_by_rollout = {str(event["rollout_id"]): event for event in originals}
    if len(original_by_rollout) != len(originals):
        raise ValueError("duplicate original-rollout audit evidence")
    for original in originals:
        prompt_ids = [int(token) for token in original.get("prompt_ids", ())]
        solution_ids = [int(token) for token in original.get("solution_ids", ())]
        solution_log_probs = _float32_values(original.get("solution_log_probs", ()))
        editable_length = int(original.get("editable_solution_length", -1))
        if not prompt_ids or not solution_ids or len(solution_ids) != len(solution_log_probs):
            raise ValueError("original audit token/log-probability evidence is incomplete")
        if not 0 < editable_length <= len(solution_ids):
            raise ValueError("original audit has an invalid editable solution length")
        _require_binary(original.get("reward"), "audited original reward")

    critiques = [event for event in events if event.get("event") == "critique"]
    continuations = [event for event in events if event.get("event") == "continuation"]
    reference_events = [event for event in events if event.get("event") == "learnability_reference"]
    learnability_events = [event for event in events if event.get("event") == "learnability"]
    expected_critiques = incorrect * num_critiques + correct * num_positive_critiques
    if len(critiques) != expected_critiques:
        raise ValueError(f"expected {expected_critiques} IID critiques, got {len(critiques)}")
    if require_algorithm_signal and not continuations:
        raise ValueError("smoke produced no learnability-accepted revision and therefore no rewarded continuation")

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
        original = original_by_rollout.get(key[0])
        if original is None:
            raise ValueError(f"critique {key!r} has no original-rollout evidence")
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
        original_prefix = tuple(
            [
                *[int(token) for token in original["prompt_ids"]],
                *[int(token) for token in original["solution_ids"][: int(original["editable_solution_length"])]],
            ]
        )
        if (
            len(critique_prompt_ids) <= len(original_prefix)
            or critique_prompt_ids[: len(original_prefix)] != original_prefix
        ):
            raise ValueError(f"critique {key!r} prompt does not preserve the exact original prompt/solution")
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
            if accepted:
                accepted_keys.add(key)
            branch_prefix_ids = [int(token) for token in critique.get("branch_prefix_ids", ())]
            replacement_ids = [int(token) for token in critique.get("new_continuation_ids", ())]
            replacement_log_probs = _float32_values(critique.get("new_continuation_log_probs", ()))
            revised_prefix_ids = [int(token) for token in critique.get("revised_prefix_ids", ())]
            generated_ids = [int(token) for token in critique.get("generated_continuation_ids", ())]
            generated_log_probs = _float32_values(critique.get("generated_continuation_log_probs", ()))
            if audit_schema_version >= 4:
                prefix = str(critique.get("prefix", ""))
                joint = str(critique.get("prefix_plus_new_continuation", ""))
                new_continuation = str(critique.get("new_continuation", ""))
                prefix_ids = [int(token) for token in critique.get("prefix_ids", ())]
                continuation_prefix_ids = [int(token) for token in critique.get("continuation_prefix_ids", ())]
                if (
                    not prefix.strip()
                    or not new_continuation.strip()
                    or joint != prefix + new_continuation
                    or not prefix_ids
                    or continuation_prefix_ids != [*branch_prefix_ids, *prefix_ids]
                    or revised_prefix_ids != [*continuation_prefix_ids, *replacement_ids]
                ):
                    raise ValueError(f"valid critique {key!r} has inconsistent prefix/joint boundaries")
            else:
                if not str(critique.get("branch", "")).strip() or not str(critique.get("new_continuation", "")).strip():
                    raise ValueError(f"valid critique {key!r} omitted an edit boundary")
                if (
                    not branch_prefix_ids
                    or not replacement_ids
                    or revised_prefix_ids != [*branch_prefix_ids, *replacement_ids]
                ):
                    raise ValueError(f"valid critique {key!r} has inconsistent replacement boundaries")
            if len(replacement_ids) != len(replacement_log_probs):
                raise ValueError(f"valid critique {key!r} replacement token/log-probability lengths differ")
            if accepted:
                if not generated_ids or len(generated_ids) != len(generated_log_probs):
                    raise ValueError(f"accepted critique {key!r} generated continuation evidence is incomplete")
            elif generated_ids or generated_log_probs:
                raise ValueError(f"learnability-rejected critique {key!r} unexpectedly generated a continuation")
            if bool(critique.get("continuation_reward_evaluated")) != accepted:
                raise ValueError(f"critique {key!r} reward-evaluation flag differs from learnability acceptance")
        else:
            invalid_boundary_fields = (
                "branch_prefix_ids",
                "new_continuation_ids",
                "new_continuation_log_probs",
                "revised_prefix_ids",
                "generated_continuation_ids",
                "generated_continuation_log_probs",
            )
            if audit_schema_version >= 4:
                invalid_boundary_fields = (
                    *invalid_boundary_fields,
                    "prefix_ids",
                    "continuation_prefix_ids",
                )
            if any(critique.get(field) for field in invalid_boundary_fields):
                raise ValueError(f"structurally invalid critique {key!r} retained edit token boundaries")
            if accepted or learnability_weight != 0.0:
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

    references = {str(event.get("reference_key")): event for event in reference_events}
    if len(references) != len(reference_events):
        raise ValueError("duplicate learnability-reference evidence")
    exhaustive_scores_by_reference: dict[str, np.ndarray] = {}
    for reference_key, reference in references.items():
        if reference.get("logprob_statistic") != statistic:
            raise ValueError(f"learnability reference {reference_key!r} used the wrong statistic")
        seed_tokens = int(reference.get("seed_tokens", 0))
        if reference_key != f"{statistic}:{seed_tokens}" or seed_tokens <= 0:
            raise ValueError(f"learnability reference {reference_key!r} has an invalid identity")
        eligible_rollouts = int(reference.get("eligible_rollouts", 0))
        if audit_schema_version == 2:
            windows = reference.get("windows")
            sampled_windows = int(reference.get("sampled_windows", -1))
            if not isinstance(windows, list) or sampled_windows != len(windows) or eligible_rollouts <= 0:
                raise ValueError(f"learnability reference {reference_key!r} has inconsistent legacy counts")
            mass_by_rollout: defaultdict[str, float] = defaultdict(float)
            for window in windows:
                rollout_id = str(window.get("rollout_id"))
                original = original_by_rollout.get(rollout_id)
                if original is None:
                    raise ValueError(f"learnability reference {reference_key!r} names an unknown rollout")
                start = int(window.get("start", -1))
                editable_length = int(original["editable_solution_length"])
                if start < 0 or start + seed_tokens > editable_length:
                    raise ValueError(f"learnability reference {reference_key!r} has an invalid window boundary")
                source_values = original["solution_log_probs"][start : start + seed_tokens]
                expected_score = _aggregate(source_values, statistic)
                if not math.isclose(float(window.get("score")), expected_score, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(f"learnability reference {reference_key!r} has a corrupted window score")
                weight = float(window.get("weight"))
                if not math.isfinite(weight) or weight <= 0.0:
                    raise ValueError(f"learnability reference {reference_key!r} has an invalid window weight")
                mass_by_rollout[rollout_id] += weight
            if len(mass_by_rollout) != eligible_rollouts or any(
                not math.isclose(mass, 1.0 / eligible_rollouts, rel_tol=0.0, abs_tol=1e-12)
                for mass in mass_by_rollout.values()
            ):
                raise ValueError(f"learnability reference {reference_key!r} does not give equal mass per rollout")
            continue

        if reference.get("window_weighting") != "uniform_per_window":
            raise ValueError(f"learnability reference {reference_key!r} does not use uniform per-window mass")
        expected_counts, exhaustive_scores = _exhaustive_reference(
            originals,
            window_size=seed_tokens,
            statistic=statistic,
        )
        total_windows = int(reference.get("total_windows", -1))
        if (
            reference.get("rollout_window_counts") != expected_counts
            or eligible_rollouts != len(expected_counts)
            or total_windows != int(exhaustive_scores.size)
            or total_windows != sum(int(row["windows"]) for row in expected_counts)
        ):
            raise ValueError(f"learnability reference {reference_key!r} is not exhaustive")
        if reference.get("window_scores_sha256") != _canonical_sha256(exhaustive_scores, dtype="<f8"):
            raise ValueError(f"learnability reference {reference_key!r} has a corrupted exhaustive score hash")
        if audit_schema_version >= 5:
            expected_mean = float(np.mean(exhaustive_scores, dtype=np.float64)) if exhaustive_scores.size else None
            expected_stddev = (
                float(np.std(exhaustive_scores, dtype=np.float64, ddof=0)) if exhaustive_scores.size else None
            )
            if not _optional_float_matches(reference.get("population_mean"), expected_mean) or not (
                _optional_float_matches(reference.get("population_stddev"), expected_stddev)
            ):
                raise ValueError(f"learnability reference {reference_key!r} has corrupted population statistics")
        exhaustive_scores_by_reference[reference_key] = exhaustive_scores

    learnability_by_key = {
        (str(event["rollout_id"]), int(event["critique_index"])): event for event in learnability_events
    }
    if len(learnability_by_key) != len(learnability_events):
        raise ValueError("duplicate learnability evidence")
    if set(learnability_by_key) != structurally_valid_keys:
        raise ValueError("every structurally valid edit must have exactly one learnability assessment")
    prompt_logprob_admissions: list[dict[str, Any]] = []
    for key, event in learnability_by_key.items():
        if event.get("score_source") != "vllm_prompt_logprobs":
            raise ValueError(f"learnability event {key!r} did not use vLLM prompt log probabilities")
        if event.get("logprob_statistic") != statistic:
            raise ValueError(f"learnability event {key!r} used the wrong log-probability statistic")
        if not 0.0 <= float(event["percentile"]) <= 1.0 or not 0.0 <= float(event["reward_weight"]) <= 1.0:
            raise ValueError(f"learnability event {key!r} has an invalid percentile or weight")
        seed_tokens = int(event["seed_tokens"])
        if seed_tokens <= 0:
            raise ValueError(f"learnability event {key!r} has no replacement seed")
        critique = critique_by_key[key]
        reference = references.get(str(event.get("reference_key")))
        if reference is None or int(reference["seed_tokens"]) != seed_tokens:
            raise ValueError(f"learnability event {key!r} lacks its exact length-matched reference")
        reference_window_field = "sampled_windows" if audit_schema_version == 2 else "total_windows"
        if int(event["eligible_rollouts"]) != int(reference["eligible_rollouts"]) or int(
            event[reference_window_field]
        ) != int(reference[reference_window_field]):
            raise ValueError(f"learnability event {key!r} reference counts disagree")
        scored_ids = [int(token) for token in event.get("scored_token_ids", ())]
        scored_log_probs = _float32_values(event.get("scored_token_log_probs", ()))
        scoring_prompt_ids = [int(token) for token in event.get("scoring_prompt_ids", ())]
        prompt_logprob_start = int(event.get("prompt_logprob_start", -1))
        original = original_by_rollout[key[0]]
        if audit_schema_version >= 4:
            scoring_prefix_ids = [int(token) for token in critique["continuation_prefix_ids"]]
        else:
            scoring_prefix_ids = [int(token) for token in critique["branch_prefix_ids"]]
        expected_prompt = [
            *[int(token) for token in original["prompt_ids"]],
            *scoring_prefix_ids,
            *[int(token) for token in critique["new_continuation_ids"]],
        ]
        expected_start = len(original["prompt_ids"]) + len(scoring_prefix_ids)
        if (
            scoring_prompt_ids != expected_prompt
            or prompt_logprob_start != expected_start
            or scored_ids != [int(token) for token in critique["new_continuation_ids"]]
            or scored_log_probs != _float32_values(critique["new_continuation_log_probs"])
            or scoring_prompt_ids[prompt_logprob_start:] != scored_ids
            or len(scored_ids) != len(scored_log_probs)
        ):
            raise ValueError(f"learnability event {key!r} has a corrupted prompt-scoring slice")
        if prompt_logprob_capacity is not None:
            admission = event.get("prompt_logprob_admission")
            if not isinstance(admission, dict):
                raise ValueError(f"learnability event {key!r} omitted prompt-logprob admission evidence")
            prompt_tokens = len(expected_prompt)
            charged_tokens = min(prompt_tokens, prompt_logprob_capacity)
            oversized = prompt_tokens > prompt_logprob_capacity
            server_id = admission.get("server_id")
            numeric_fields = (
                "request_sequence",
                "prompt_tokens",
                "charged_tokens",
                "inflight_prompt_tokens_at_grant",
                "inflight_charged_tokens_at_grant",
                "high_water_prompt_tokens",
                "high_water_charged_tokens",
            )
            try:
                numeric = {name: int(admission[name]) for name in numeric_fields}
                wait_seconds = float(admission["wait_seconds"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"learnability event {key!r} has incomplete admission evidence") from error
            if (
                not isinstance(server_id, str)
                or not server_id
                or int(admission.get("capacity", -1)) != prompt_logprob_capacity
                or numeric["request_sequence"] <= 0
                or numeric["prompt_tokens"] != prompt_tokens
                or numeric["charged_tokens"] != charged_tokens
                or numeric["inflight_prompt_tokens_at_grant"] < prompt_tokens
                or numeric["inflight_charged_tokens_at_grant"] < charged_tokens
                or numeric["inflight_charged_tokens_at_grant"] > prompt_logprob_capacity
                or numeric["high_water_prompt_tokens"] < numeric["inflight_prompt_tokens_at_grant"]
                or numeric["high_water_charged_tokens"] < numeric["inflight_charged_tokens_at_grant"]
                or numeric["high_water_charged_tokens"] > prompt_logprob_capacity
                or bool(admission.get("oversized")) != oversized
                or not math.isfinite(wait_seconds)
                or wait_seconds < 0.0
            ):
                raise ValueError(f"learnability event {key!r} violates the prompt-logprob admission budget")
            if oversized and (
                numeric["inflight_prompt_tokens_at_grant"] != prompt_tokens
                or numeric["inflight_charged_tokens_at_grant"] != prompt_logprob_capacity
            ):
                raise ValueError(f"oversized learnability request {key!r} did not run alone")
            prompt_logprob_admissions.append(admission)
        expected_seed_score = _aggregate(scored_log_probs, statistic)
        if not math.isclose(float(event["seed_score"]), expected_seed_score, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"learnability event {key!r} has a corrupted replacement score")
        if audit_schema_version == 2:
            expected_percentile = sum(
                float(window["weight"])
                for window in reference["windows"]
                if float(window["score"]) <= expected_seed_score
            )
        else:
            exhaustive_scores = exhaustive_scores_by_reference[str(event["reference_key"])]
            expected_percentile = (
                float(np.count_nonzero(exhaustive_scores <= expected_seed_score)) / float(exhaustive_scores.size)
                if exhaustive_scores.size
                else 0.0
            )
        expected_percentile = min(max(expected_percentile, 0.0), 1.0)
        minimum = float(branch_config["min_seed_window_percentile"])
        full_credit = float(branch_config["full_credit_seed_window_percentile"])
        if audit_schema_version >= 5:
            exhaustive_scores = exhaustive_scores_by_reference[str(event["reference_key"])]
            reference_mean = float(np.mean(exhaustive_scores, dtype=np.float64)) if exhaustive_scores.size else None
            reference_stddev = (
                float(np.std(exhaustive_scores, dtype=np.float64, ddof=0)) if exhaustive_scores.size else None
            )
            expected_stddevs = (
                (reference_mean - expected_seed_score) / reference_stddev
                if reference_mean is not None and reference_stddev is not None and reference_stddev > 0.0
                else None
            )
            if threshold_mode == "stddev":
                expected_floor = (
                    reference_mean - max_seed_window_stddevs * reference_stddev
                    if reference_mean is not None and reference_stddev is not None
                    else None
                )
                expected_accepted = expected_floor is not None and expected_seed_score >= expected_floor
                expected_weight = float(expected_accepted)
            else:
                expected_floor = None
                expected_accepted = exhaustive_scores.size > 0 and expected_percentile >= minimum
                expected_weight = min(max((expected_percentile - minimum) / (full_credit - minimum), 0.0), 1.0)
            if (
                event.get("threshold_mode") != threshold_mode
                or not _optional_float_matches(event.get("reference_mean"), reference_mean)
                or not _optional_float_matches(event.get("reference_stddev"), reference_stddev)
                or not _optional_float_matches(event.get("stddevs_below_mean"), expected_stddevs)
                or not _optional_float_matches(event.get("acceptance_floor"), expected_floor)
                or not math.isclose(
                    float(event.get("max_seed_window_stddevs", float("nan"))),
                    max_seed_window_stddevs,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(f"learnability event {key!r} has corrupted standard-deviation evidence")
        else:
            expected_accepted = expected_percentile >= minimum
            expected_weight = min(max((expected_percentile - minimum) / (full_credit - minimum), 0.0), 1.0)
        if (
            not math.isclose(float(event["percentile"]), expected_percentile, rel_tol=0.0, abs_tol=1e-12)
            or bool(event["accepted"]) != expected_accepted
            or not math.isclose(float(event["reward_weight"]), expected_weight, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError(f"learnability event {key!r} does not match its audited reference distribution")
        if bool(event["accepted"]) != bool(critique["learnability_accepted"]) or not math.isclose(
            float(event["reward_weight"]), float(critique["learnability_weight"]), abs_tol=1e-9
        ):
            raise ValueError(f"learnability event {key!r} differs from its trained critique reward")

    if prompt_logprob_capacity is not None and prompt_logprob_admissions:
        admission_summary = _only(events, "prompt_logprob_admission_summary")
        if (
            int(admission_summary.get("capacity", -1)) != prompt_logprob_capacity
            or int(admission_summary.get("requests", -1)) != len(prompt_logprob_admissions)
            or int(admission_summary.get("prompt_tokens", -1))
            != sum(int(item["prompt_tokens"]) for item in prompt_logprob_admissions)
        ):
            raise ValueError("prompt-logprob admission summary has incorrect global totals")
        expected_per_server: dict[str, dict[str, int | float]] = {}
        for admission in prompt_logprob_admissions:
            server_id = str(admission["server_id"])
            summary = expected_per_server.setdefault(
                server_id,
                {
                    "requests": 0,
                    "prompt_tokens": 0,
                    "max_inflight_prompt_tokens": 0,
                    "max_inflight_charged_tokens": 0,
                    "max_wait_seconds": 0.0,
                },
            )
            summary["requests"] = int(summary["requests"]) + 1
            summary["prompt_tokens"] = int(summary["prompt_tokens"]) + int(admission["prompt_tokens"])
            summary["max_inflight_prompt_tokens"] = max(
                int(summary["max_inflight_prompt_tokens"]),
                int(admission["high_water_prompt_tokens"]),
            )
            summary["max_inflight_charged_tokens"] = max(
                int(summary["max_inflight_charged_tokens"]),
                int(admission["high_water_charged_tokens"]),
            )
            summary["max_wait_seconds"] = max(
                float(summary["max_wait_seconds"]),
                float(admission["wait_seconds"]),
            )
        if admission_summary.get("per_server") != expected_per_server:
            raise ValueError("prompt-logprob admission summary does not match per-request evidence")
    elif prompt_logprob_capacity is not None and any(
        event.get("event") == "prompt_logprob_admission_summary" for event in events
    ):
        raise ValueError("prompt-logprob admission summary exists without any scored edit")

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
    padding = int(actor_batch["padding"])
    trainer_config = resolved_config.get("trainer", {})
    data_parallel_size = int(trainer_config.get("nnodes", 0)) * int(trainer_config.get("n_gpus_per_node", 0))
    if data_parallel_size <= 0:
        raise ValueError(f"invalid actor data-parallel size: {data_parallel_size}")
    if not 0 <= padding < data_parallel_size or (expected_actor_rows + padding) % data_parallel_size:
        raise ValueError(f"invalid data-parallel padding count: {padding}")
    if not isinstance(actor_rows, list) or len(actor_rows) != expected_actor_rows + padding:
        raise ValueError("actor-batch audit must retain every balanced row")
    actor_kind_counts = Counter(str(row.get("kind")) for row in actor_rows)
    if actor_kind_counts != Counter(
        original=expected_originals,
        critique=expected_critiques,
        continuation=len(continuations),
        padding=padding,
    ):
        raise ValueError(f"actor-row kind counts mismatch: {actor_kind_counts!r}")

    expected_sources: dict[str, dict[str, Any]] = {}
    for rollout_id, original in original_by_rollout.items():
        expected_sources[f"original:{rollout_id}"] = {
            "kind": "original",
            "group_id": f"solution:{original['prompt_group_id']}",
            "reward": float(original["reward"]),
            "full_ids": [*original["prompt_ids"], *original["solution_ids"]],
            "train_start": len(original["prompt_ids"]),
            "behavior_log_probs": _float32_values(original["solution_log_probs"]),
        }
    for key, critique in critique_by_key.items():
        row_id = str(critique["actor_row_id"])
        expected_sources[row_id] = {
            "kind": "critique",
            "group_id": f"critique:{key[0]}",
            "reward": float(critique["reward"]),
            "full_ids": [*critique["critique_prompt_ids"], *critique["critique_ids"]],
            "train_start": len(critique["critique_prompt_ids"]),
            "behavior_log_probs": _float32_values(critique["critique_log_probs"]),
        }
    for continuation in continuations:
        rollout_id = str(continuation["rollout_id"])
        original = original_by_rollout[rollout_id]
        row_id = str(continuation["actor_row_id"])
        expected_sources[row_id] = {
            "kind": "continuation",
            "group_id": f"solution:{original['prompt_group_id']}",
            "reward": float(continuation["reward"]),
            "full_ids": [
                *original["prompt_ids"],
                *continuation["revised_prefix_ids"],
                *continuation["continuation_ids"],
            ],
            "train_start": len(original["prompt_ids"]) + len(continuation["revised_prefix_ids"]),
            "behavior_log_probs": _float32_values(continuation["continuation_log_probs"]),
        }
    if len(expected_sources) != expected_actor_rows:
        raise ValueError("source audit evidence does not map one-to-one onto non-padding actor rows")

    seen_source_ids: set[str] = set()
    balanced_indices: set[int] = set()
    response_widths = {int(row["response_width"]) for row in actor_rows}
    if len(response_widths) != 1:
        raise ValueError("balanced actor rows disagree on response width")
    response_width = response_widths.pop()
    pad_token_id = int(actor_batch["pad_token_id"])
    for row in actor_rows:
        row_id = str(row.get("actor_row_id"))
        balanced_indices.add(int(row.get("balanced_row_index", -1)))
        if row_id.startswith("padding:"):
            expected = {
                "kind": "padding",
                "group_id": row_id,
                "reward": 0.0,
                "full_ids": [pad_token_id],
                "train_start": None,
                "behavior_log_probs": [],
            }
        else:
            expected = expected_sources.get(row_id)
            if expected is None:
                raise ValueError(f"balanced actor row {row_id!r} has no source evidence")
            if row_id in seen_source_ids:
                raise ValueError(f"balanced actor row {row_id!r} is duplicated")
            seen_source_ids.add(row_id)
        full_ids = [int(token) for token in expected["full_ids"]]
        behavior = _float32_values(expected["behavior_log_probs"])
        train_start = expected["train_start"]
        if train_start is None:
            train_stop = None
            response_mask = [0] * response_width
        else:
            train_stop = int(train_start) + len(behavior)
            if not 0 < int(train_start) < train_stop == len(full_ids) or len(full_ids) - 1 > response_width:
                raise ValueError(f"source actor row {row_id!r} has an invalid train span")
            response_mask = [0] * response_width
            response_mask[int(train_start) - 1 : train_stop - 1] = [1] * len(behavior)
        if (
            str(row.get("kind")) != expected["kind"]
            or str(row.get("group_id")) != expected["group_id"]
            or not math.isclose(float(row.get("reward")), float(expected["reward"]), abs_tol=1e-6)
            or int(row.get("sequence_length")) != len(full_ids)
            or row.get("train_start") != train_start
            or row.get("train_stop") != train_stop
            or row.get("input_ids_sha256") != _canonical_sha256(full_ids, dtype="<i8")
            or row.get("response_mask_sha256") != _canonical_sha256(response_mask, dtype="u1")
            or row.get("old_log_probs_sha256") != _canonical_sha256(behavior, dtype="<f4")
            or row.get("rollout_log_probs_sha256") != _canonical_sha256(behavior, dtype="<f4")
        ):
            raise ValueError(f"balanced actor row {row_id!r} does not match its source tensors")
    if seen_source_ids != set(expected_sources) or balanced_indices != set(range(len(actor_rows))):
        raise ValueError("balanced actor audit lost, duplicated, or misindexed source rows")

    original_actor_rows = [row for row in actor_rows if row["kind"] == "original"]
    critique_actor_rows = [row for row in actor_rows if row["kind"] == "critique"]
    continuation_actor_rows = [row for row in actor_rows if row["kind"] == "continuation"]
    continuation_actor_rewards = sorted(
        _require_binary(row["reward"], "continuation actor reward") for row in continuation_actor_rows
    )
    continuation_audit_rewards = sorted(float(event["reward"]) for event in continuations)
    if continuation_actor_rewards != continuation_audit_rewards:
        raise ValueError("revised solution actor rows must use their binary continuation outcomes")
    critique_actor_rewards = sorted(_float32_values(row["reward"] for row in critique_actor_rows))
    critique_audit_rewards = sorted(_float32_values(event["reward"] for event in critiques))
    if critique_actor_rewards != critique_audit_rewards:
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
    if require_algorithm_signal and not any(
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
        if require_algorithm_signal and not any(
            len({float(row["reward"]) for row in critique_actor_rows if str(row["group_id"]) == group_id}) > 1
            for group_id in objective_groups
        ):
            raise ValueError(f"smoke has no nonuniform {objective} critique GRPO reward group")

    metrics = _read_jsonl(root / "metrics.jsonl")
    step_rows = [row for row in metrics if int(row.get("step", -1)) == selected_step]
    if not step_rows:
        raise ValueError(f"file logger contains no global step {selected_step} metrics")
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
    successful_recovery_count = sum(
        float(event["reward"]) for event in continuations if event.get("objective") == "recovery"
    )
    accepted_recovery_count = sum(rollout_objectives[key[0]] == "recovery" for key in accepted_keys)
    valid_recovery_count = sum(rollout_objectives[key[0]] == "recovery" for key in structurally_valid_keys)
    expected_rates = {
        "branch_revision/flip/success_per_valid_continuation": (
            successful_recovery_count / accepted_recovery_count if accepted_recovery_count else 0.0
        ),
        "branch_revision/flip/success_per_continuation": (
            successful_recovery_count / valid_recovery_count if valid_recovery_count else 0.0
        ),
    }
    for key, expected in expected_rates.items():
        actual = float(merged_metrics.get(key, float("nan")))
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"metric {key} mismatch: {actual!r} != {expected!r}")
    grad_keys = [key for key in merged_metrics if key.endswith("actor/grad_norm") or key == "actor/grad_norm"]
    if not grad_keys:
        raise ValueError("optimizer-step grad_norm metric is missing")
    finite_grad_norms = [float(merged_metrics[key]) for key in grad_keys if math.isfinite(float(merged_metrics[key]))]
    if not finite_grad_norms:
        raise ValueError("optimizer-step grad_norm metrics contain no finite values")
    if require_algorithm_signal and not any(value > 0.0 for value in finite_grad_norms):
        raise ValueError("optimizer-step grad_norm metrics contain no finite positive learning signal")
    pg_loss = float(merged_metrics.get("actor/pg_loss", float("nan")))
    if not math.isfinite(pg_loss):
        raise ValueError("optimizer-step actor/pg_loss metric is missing or non-finite")
    successful_revisions = sum(float(event["reward"]) for event in continuations)
    if require_algorithm_signal and successful_revisions <= 0.0:
        raise ValueError("smoke has no successful revised continuation")
    successful_recoveries = successful_recovery_count
    successful_compressions = sum(
        float(event.get("compression_credit") or 0.0)
        for event in continuations
        if event.get("objective") == "compression"
    )
    if require_algorithm_signal and successful_recoveries <= 0.0:
        raise ValueError("smoke has no successful recovery continuation")
    if require_algorithm_signal and successful_compressions <= 0.0:
        raise ValueError("smoke has no successful positive-rollout compression")

    return {
        "status": "verified" if require_algorithm_signal else "integrity-verified",
        "algorithm_signal_required": require_algorithm_signal,
        "audit_schema_version": audit_schema_version,
        "audit_attempt_id": attempt_id,
        "audit_file": str(selected_audit_file),
        "audit_files": [str(path) for path in audit_files],
        "selected_global_step": selected_step,
        "event_counts": dict(sorted(event_counts.items())),
        "incorrect_originals": incorrect,
        "correct_originals": correct,
        "valid_edits": len(structurally_valid_keys),
        "learnability_accepted_edits": len(accepted_keys),
        "successful_revisions": successful_revisions,
        "successful_compression_credit": successful_compressions,
        "policy_loss_mode": loss_mode,
        "learnability_logprob_statistic": statistic,
        "learnability_threshold_mode": threshold_mode,
        "max_seed_window_stddevs": max_seed_window_stddevs,
        "actor_rows": expected_actor_rows,
        "padding_rows": padding,
        "wall_seconds": float(completed["wall_seconds"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--integrity-only",
        action="store_true",
        help="verify complete schema-v2/v3/v4/v5 evidence without requiring nonzero revision learning signal",
    )
    args = parser.parse_args()
    result = verify(args.root, require_algorithm_signal=not args.integrity_only)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
