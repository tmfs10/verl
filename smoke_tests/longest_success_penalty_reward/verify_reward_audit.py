#!/usr/bin/env python3
"""Independently reconstruct longest-success-penalty rollout rewards."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


WIDE_PAGE_CSS = """<style>
body,
main,
article,
.markdown-body,
.rendered_html,
.jp-RenderedHTMLCommon,
.jp-MarkdownOutput {
  max-width: none !important;
  width: min(98vw, 1800px) !important;
}
table { width: 100% !important; }
</style>
"""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _assert_close(actual: float, expected: float, context: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=1e-7, abs_tol=1e-7):
        raise AssertionError(f"{context}: actual={actual}, expected={expected}")


def verify_rollouts(
    rows: list[dict[str, Any]],
    margin_percent: float,
    expected_group_size: int,
) -> dict[str, Any]:
    required = {
        "score",
        "acc",
        "rule_reward",
        "response_tokens",
        "longest_success_penalty_reward",
        "longest_success_penalized",
        "longest_success_response_tokens",
        "longest_success_group_id",
        "longest_success_group_size",
        "longest_success_group_has_success",
        "longest_success_group_within_margin",
        "longest_success_group_min_tokens",
        "longest_success_group_max_tokens",
        "longest_success_group_no_penalty_threshold_tokens",
        "longest_success_length_ratio",
        "longest_success_no_penalty_margin_percent",
        "longest_success_reward_token_index",
        "longest_success_group_key_source",
        "longest_success_reward_tensor_coordinate",
        "longest_success_reward_tensor_width",
        "longest_success_nonzero_reward_token_count",
        "longest_success_valid_response_nonzero_reward_token_count",
        "longest_success_pad_nonzero_reward_token_count",
        "longest_success_reward_tensor_row_sum",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise AssertionError(f"Rollout audit is missing fields: {missing}")

    groups: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[str(row["longest_success_group_id"])].append(idx)
    bad_sizes = {key: len(indices) for key, indices in groups.items() if len(indices) != expected_group_size}
    if bad_sizes:
        raise AssertionError(f"Incomplete rollout groups: {bad_sizes}")

    correct_count = 0
    rewarded_count = 0
    penalized_count = 0
    successful_groups = 0
    within_margin_groups = 0
    valid_response_nonzero_reward_tokens = 0
    pad_nonzero_reward_tokens = 0
    successful_group_min_tokens: list[int] = []
    successful_group_max_tokens: list[int] = []
    rewarded_response_tokens: list[int] = []
    penalized_response_tokens: list[int] = []
    group_details: list[dict[str, Any]] = []

    for group_number, (group_id, indices) in enumerate(groups.items(), start=1):
        successes = [idx for idx in indices if float(rows[idx]["acc"]) > 0.5]
        success_lengths = [int(rows[idx]["longest_success_response_tokens"]) for idx in successes]
        min_tokens = min(success_lengths, default=None)
        max_tokens = max(success_lengths, default=None)
        threshold_tokens = None if min_tokens is None else min_tokens * (1.0 + margin_percent / 100.0)
        within_margin = bool(max_tokens is not None and float(max_tokens) <= float(threshold_tokens))
        if successes:
            successful_groups += 1
            within_margin_groups += int(within_margin)
            successful_group_min_tokens.append(int(min_tokens))
            successful_group_max_tokens.append(int(max_tokens))

        group_rewards = 0
        group_penalties = 0
        for idx in indices:
            row = rows[idx]
            response_length = int(row["longest_success_response_tokens"])
            if response_length != len(row["response_tokens"]):
                raise AssertionError(
                    f"row {idx}: response-mask length {response_length} != dumped valid tokens "
                    f"{len(row['response_tokens'])}"
                )
            if row["longest_success_group_key_source"] != "prompt_group_id":
                raise AssertionError(f"row {idx}: unexpected group key source")
            if int(row["longest_success_group_size"]) != expected_group_size:
                raise AssertionError(f"row {idx}: wrong logged group size")
            _assert_close(
                row["longest_success_no_penalty_margin_percent"],
                margin_percent,
                f"row {idx} margin",
            )

            is_success = float(row["acc"]) > 0.5
            expected_penalized = bool(
                is_success and not within_margin and max_tokens is not None and response_length == max_tokens
            )
            expected_reward = 1.0 if is_success and not expected_penalized else 0.0
            expected_ratio = float(response_length) / float(min_tokens) if is_success and min_tokens else None

            _assert_close(row["rule_reward"], row["acc"], f"row {idx} raw verifier reward")
            _assert_close(row["score"], row["acc"], f"row {idx} raw dumped score")
            _assert_close(row["longest_success_penalty_reward"], expected_reward, f"row {idx} reward")
            if bool(row["longest_success_penalized"]) != expected_penalized:
                raise AssertionError(f"row {idx}: penalized flag mismatch")
            if bool(row["longest_success_group_has_success"]) != bool(successes):
                raise AssertionError(f"row {idx}: group success flag mismatch")
            if bool(row["longest_success_group_within_margin"]) != within_margin:
                raise AssertionError(f"row {idx}: within-margin flag mismatch")
            if row["longest_success_group_min_tokens"] != min_tokens:
                raise AssertionError(f"row {idx}: minimum token count mismatch")
            if row["longest_success_group_max_tokens"] != max_tokens:
                raise AssertionError(f"row {idx}: maximum token count mismatch")
            if threshold_tokens is None:
                if row["longest_success_group_no_penalty_threshold_tokens"] is not None:
                    raise AssertionError(f"row {idx}: threshold should be null")
            else:
                _assert_close(
                    row["longest_success_group_no_penalty_threshold_tokens"],
                    threshold_tokens,
                    f"row {idx} threshold",
                )
            if expected_ratio is None:
                if row["longest_success_length_ratio"] is not None:
                    raise AssertionError(f"row {idx}: failure ratio should be null")
            else:
                _assert_close(row["longest_success_length_ratio"], expected_ratio, f"row {idx} ratio")

            expected_reward_index = response_length - 1 if response_length > 0 else None
            if row["longest_success_reward_token_index"] != expected_reward_index:
                raise AssertionError(f"row {idx}: reward token index mismatch")
            if row["longest_success_reward_tensor_coordinate"] != "response_only":
                raise AssertionError(f"row {idx}: reward tensor is not response-only")
            if int(row["longest_success_reward_tensor_width"]) < response_length:
                raise AssertionError(f"row {idx}: response exceeds reward-tensor width")
            expected_nonzero = int(expected_reward > 0)
            if int(row["longest_success_nonzero_reward_token_count"]) != expected_nonzero:
                raise AssertionError(f"row {idx}: wrong nonzero reward-token count")
            if int(row["longest_success_valid_response_nonzero_reward_token_count"]) != expected_nonzero:
                raise AssertionError(f"row {idx}: nonzero reward is not on a valid response token")
            if int(row["longest_success_pad_nonzero_reward_token_count"]) != 0:
                raise AssertionError(f"row {idx}: nonzero reward leaked onto PAD")
            _assert_close(row["longest_success_reward_tensor_row_sum"], expected_reward, f"row {idx} row sum")

            correct_count += int(is_success)
            rewarded_count += int(expected_reward)
            penalized_count += int(expected_penalized)
            if expected_reward > 0:
                rewarded_response_tokens.append(response_length)
            if expected_penalized:
                penalized_response_tokens.append(response_length)
            group_rewards += int(expected_reward)
            group_penalties += int(expected_penalized)
            valid_response_nonzero_reward_tokens += int(
                row["longest_success_valid_response_nonzero_reward_token_count"]
            )
            pad_nonzero_reward_tokens += int(row["longest_success_pad_nonzero_reward_token_count"])

        print(
            f"[audit] group {group_number}/{len(groups)} id={group_id} successes={len(successes)} "
            f"min={min_tokens} max={max_tokens} within_margin={within_margin} "
            f"rewarded={group_rewards} penalized={group_penalties}"
        )
        group_details.append(
            {
                "group_id": group_id,
                "successes": len(successes),
                "minimum_success_tokens": min_tokens,
                "maximum_success_tokens": max_tokens,
                "inclusive_no_penalty_threshold": threshold_tokens,
                "within_margin": within_margin,
                "rewarded_successes": group_rewards,
                "penalized_longest_ties": group_penalties,
                "success_response_lengths": success_lengths,
            }
        )

    penalized_groups = successful_groups - within_margin_groups
    return {
        "rollouts": len(rows),
        "groups": len(groups),
        "successful_groups": successful_groups,
        "within_margin_successful_groups": within_margin_groups,
        "penalized_successful_groups": penalized_groups,
        "correct_rollouts": correct_count,
        "rewarded_rollouts": rewarded_count,
        "penalized_longest_rollouts": penalized_count,
        "effective_training_reward_sum": rewarded_count,
        "rewarded_fraction": rewarded_count / len(rows),
        "rewarded_per_correct": rewarded_count / correct_count if correct_count else 0.0,
        "penalized_per_correct": penalized_count / correct_count if correct_count else 0.0,
        "groups_with_success_fraction": successful_groups / len(groups),
        "successful_groups_within_margin_fraction": (
            within_margin_groups / successful_groups if successful_groups else 0.0
        ),
        "successful_groups_penalized_fraction": penalized_groups / successful_groups if successful_groups else 0.0,
        "mean_min_success_tokens": (
            sum(successful_group_min_tokens) / len(successful_group_min_tokens)
            if successful_group_min_tokens
            else 0.0
        ),
        "mean_max_success_tokens": (
            sum(successful_group_max_tokens) / len(successful_group_max_tokens)
            if successful_group_max_tokens
            else 0.0
        ),
        "mean_max_to_min_ratio": (
            sum(maximum / minimum for minimum, maximum in zip(
                successful_group_min_tokens,
                successful_group_max_tokens,
                strict=True,
            ))
            / len(successful_group_min_tokens)
            if successful_group_min_tokens
            else 0.0
        ),
        "mean_rewarded_tokens": (
            sum(rewarded_response_tokens) / len(rewarded_response_tokens) if rewarded_response_tokens else 0.0
        ),
        "mean_penalized_tokens": (
            sum(penalized_response_tokens) / len(penalized_response_tokens)
            if penalized_response_tokens
            else 0.0
        ),
        "raw_acc_mean": correct_count / len(rows),
        "prompt_token_reward_count": 0,
        "valid_response_nonzero_reward_token_count": valid_response_nonzero_reward_tokens,
        "pad_token_reward_count": pad_nonzero_reward_tokens,
        "group_details": group_details,
    }


def verify_trainer_metrics(trainer_log: Path, rollout_summary: dict[str, Any]) -> dict[str, Any]:
    metric_to_summary = {
        "reward/longest_success_penalty/rewarded_fraction": "rewarded_fraction",
        "reward/longest_success_penalty/rewarded_per_correct": "rewarded_per_correct",
        "reward/longest_success_penalty/penalized_per_correct": "penalized_per_correct",
        "reward/longest_success_penalty/groups_with_success_fraction": "groups_with_success_fraction",
        "reward/longest_success_penalty/successful_groups_within_margin_fraction": (
            "successful_groups_within_margin_fraction"
        ),
        "reward/longest_success_penalty/successful_groups_penalized_fraction": (
            "successful_groups_penalized_fraction"
        ),
        "reward/longest_success_penalty/mean_min_success_tokens": "mean_min_success_tokens",
        "reward/longest_success_penalty/mean_max_success_tokens": "mean_max_success_tokens",
        "reward/longest_success_penalty/mean_max_to_min_ratio": "mean_max_to_min_ratio",
        "reward/longest_success_penalty/mean_rewarded_tokens": "mean_rewarded_tokens",
        "reward/longest_success_penalty/mean_penalized_tokens": "mean_penalized_tokens",
        "reward/longest_success_penalty/raw_acc_mean": "raw_acc_mean",
    }
    step_line = None
    with trainer_log.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "step:1 -" in line and "reward/longest_success_penalty/" in line:
                step_line = line
    if step_line is None:
        raise AssertionError(f"No step-1 longest-success metric line found in {trainer_log}")

    logged: dict[str, float] = {}
    for metric, summary_key in metric_to_summary.items():
        match = re.search(rf"(?:^| - ){re.escape(metric)}:([^ ]+)", step_line)
        if match is None:
            raise AssertionError(f"Trainer step-1 line is missing {metric}")
        logged_value = float(match.group(1))
        expected_value = float(rollout_summary[summary_key])
        _assert_close(logged_value, expected_value, metric)
        logged[metric] = logged_value
    return {
        "trainer_step": 1,
        "trainer_metrics_verified": len(logged),
        "trainer_logged_metrics": logged,
    }


def verify_validation(rows: list[dict[str, Any]], expected_rows: int) -> dict[str, Any]:
    if len(rows) != expected_rows:
        raise AssertionError(f"validation rows={len(rows)}, expected={expected_rows}")
    for idx, row in enumerate(rows):
        if "acc" not in row:
            raise AssertionError(f"validation row {idx} has no acc field")
        _assert_close(row["score"], row["acc"], f"validation row {idx}")
        if any(key.startswith("longest_success_") for key in row):
            raise AssertionError(f"validation row {idx} contains training-only reward fields")
    return {
        "validation_rows": len(rows),
        "validation_accuracy": sum(float(row["acc"]) for row in rows) / len(rows),
        "validation_training_reward_fields": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-jsonl", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--trainer-log", type=Path, required=True)
    parser.add_argument("--margin-percent", type=float, default=50.0)
    parser.add_argument("--expected-group-size", type=int, default=8)
    parser.add_argument("--expected-validation-rows", type=int, default=1024)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()

    rollout_summary = verify_rollouts(
        _load_jsonl(args.rollout_jsonl),
        margin_percent=args.margin_percent,
        expected_group_size=args.expected_group_size,
    )
    validation_summary = verify_validation(
        _load_jsonl(args.validation_jsonl),
        expected_rows=args.expected_validation_rows,
    )
    trainer_summary = verify_trainer_metrics(args.trainer_log, rollout_summary)
    summary = {
        "status": "PASS",
        "margin_percent": args.margin_percent,
        "expected_group_size": args.expected_group_size,
        **rollout_summary,
        **validation_summary,
        **trainer_summary,
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "audit_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = WIDE_PAGE_CSS + "\n# Longest-success-penalty reward audit\n\n"
    report += "| Metric | Value |\n| --- | ---: |\n"
    report += "\n".join(
        f"| {key} | {value} |" for key, value in summary.items() if key != "group_details"
    ) + "\n"
    report += "\n## Per-group reconstruction\n\n"
    report += (
        "| Group ID | Successes | Min tokens | Max tokens | Inclusive threshold | Within margin | "
        "Rewarded | Penalized max ties | Success lengths |\n"
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |\n"
    )
    for group in summary["group_details"]:
        report += (
            f"| {group['group_id']} | {group['successes']} | {group['minimum_success_tokens']} | "
            f"{group['maximum_success_tokens']} | {group['inclusive_no_penalty_threshold']} | "
            f"{group['within_margin']} | {group['rewarded_successes']} | "
            f"{group['penalized_longest_ties']} | {group['success_response_lengths']} |\n"
        )
    (args.report_dir / "audit_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
