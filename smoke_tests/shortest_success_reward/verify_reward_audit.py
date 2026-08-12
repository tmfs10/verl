#!/usr/bin/env python3
"""Independently reconstruct shortest-success rewards from dumped generations."""

from __future__ import annotations

import argparse
import json
import math
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
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
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


def verify_rollouts(rows: list[dict[str, Any]], margin_percent: float, expected_group_size: int) -> dict[str, Any]:
    required = {
        "score",
        "acc",
        "rule_reward",
        "response_tokens",
        "shortest_success_reward",
        "shortest_success_selected",
        "shortest_success_response_tokens",
        "shortest_success_group_id",
        "shortest_success_group_size",
        "shortest_success_group_has_success",
        "shortest_success_group_min_tokens",
        "shortest_success_group_threshold_tokens",
        "shortest_success_length_ratio",
        "shortest_success_margin_percent",
        "shortest_success_reward_token_index",
        "shortest_success_group_key_source",
        "shortest_success_reward_tensor_coordinate",
        "shortest_success_reward_tensor_width",
        "shortest_success_nonzero_reward_token_count",
        "shortest_success_valid_response_nonzero_reward_token_count",
        "shortest_success_pad_nonzero_reward_token_count",
        "shortest_success_reward_tensor_row_sum",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise AssertionError(f"Rollout audit is missing fields: {missing}")

    groups: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[str(row["shortest_success_group_id"])].append(idx)

    if any(len(indices) != expected_group_size for indices in groups.values()):
        sizes = {key: len(indices) for key, indices in groups.items()}
        raise AssertionError(f"Incomplete rollout groups: {sizes}")

    selected_count = 0
    correct_count = 0
    successful_groups = 0
    successful_group_min_tokens: list[int] = []
    selected_response_tokens: list[int] = []
    valid_response_nonzero_reward_tokens = 0
    pad_nonzero_reward_tokens = 0
    group_details: list[dict[str, Any]] = []
    for group_number, (group_id, indices) in enumerate(groups.items(), start=1):
        successes = [idx for idx in indices if float(rows[idx]["acc"]) > 0.5]
        min_tokens = min((int(rows[idx]["shortest_success_response_tokens"]) for idx in successes), default=None)
        threshold_tokens = None if min_tokens is None else min_tokens * (1.0 + margin_percent / 100.0)
        if successes:
            successful_groups += 1
            successful_group_min_tokens.append(int(min_tokens))

        for idx in indices:
            row = rows[idx]
            response_length = int(row["shortest_success_response_tokens"])
            if response_length != len(row["response_tokens"]):
                raise AssertionError(
                    f"row {idx}: response-mask length {response_length} != dumped valid tokens {len(row['response_tokens'])}"
                )
            if row["shortest_success_group_key_source"] != "prompt_group_id":
                raise AssertionError(f"row {idx}: unexpected group key source")
            if int(row["shortest_success_group_size"]) != expected_group_size:
                raise AssertionError(f"row {idx}: wrong logged group size")
            _assert_close(row["shortest_success_margin_percent"], margin_percent, f"row {idx} margin")

            is_success = float(row["acc"]) > 0.5
            expected_selected = bool(
                is_success and threshold_tokens is not None and response_length <= threshold_tokens
            )
            expected_reward = 1.0 if expected_selected else 0.0
            expected_ratio = float(response_length) / float(min_tokens) if is_success and min_tokens else None

            _assert_close(row["rule_reward"], row["acc"], f"row {idx} raw verifier reward")
            # The generic rollout dumper retains the verifier's raw scalar in
            # `score`. The effective training reward is the independently
            # audited reward-tensor row sum below.
            _assert_close(row["score"], row["acc"], f"row {idx} raw dumped score")
            _assert_close(row["shortest_success_reward"], expected_reward, f"row {idx} shaped reward")
            if bool(row["shortest_success_selected"]) != expected_selected:
                raise AssertionError(f"row {idx}: selected flag mismatch")
            if bool(row["shortest_success_group_has_success"]) != bool(successes):
                raise AssertionError(f"row {idx}: group success flag mismatch")
            if row["shortest_success_group_min_tokens"] != min_tokens:
                raise AssertionError(f"row {idx}: minimum token count mismatch")
            if threshold_tokens is None:
                if row["shortest_success_group_threshold_tokens"] is not None:
                    raise AssertionError(f"row {idx}: threshold should be null")
            else:
                _assert_close(
                    row["shortest_success_group_threshold_tokens"], threshold_tokens, f"row {idx} threshold"
                )
            if expected_ratio is None:
                if row["shortest_success_length_ratio"] is not None:
                    raise AssertionError(f"row {idx}: failure ratio should be null")
            else:
                _assert_close(row["shortest_success_length_ratio"], expected_ratio, f"row {idx} ratio")
            expected_reward_index = response_length - 1 if response_length > 0 else None
            if row["shortest_success_reward_token_index"] != expected_reward_index:
                raise AssertionError(f"row {idx}: reward token index mismatch")
            if row["shortest_success_reward_tensor_coordinate"] != "response_only":
                raise AssertionError(f"row {idx}: reward tensor is not response-only")
            if int(row["shortest_success_reward_tensor_width"]) < response_length:
                raise AssertionError(f"row {idx}: response exceeds logged reward-tensor width")
            expected_nonzero = int(expected_selected)
            if int(row["shortest_success_nonzero_reward_token_count"]) != expected_nonzero:
                raise AssertionError(f"row {idx}: wrong nonzero reward-token count")
            if int(row["shortest_success_valid_response_nonzero_reward_token_count"]) != expected_nonzero:
                raise AssertionError(f"row {idx}: nonzero reward is not on a valid response token")
            if int(row["shortest_success_pad_nonzero_reward_token_count"]) != 0:
                raise AssertionError(f"row {idx}: nonzero reward leaked onto PAD")
            _assert_close(row["shortest_success_reward_tensor_row_sum"], expected_reward, f"row {idx} row sum")
            valid_response_nonzero_reward_tokens += int(
                row["shortest_success_valid_response_nonzero_reward_token_count"]
            )
            pad_nonzero_reward_tokens += int(row["shortest_success_pad_nonzero_reward_token_count"])

            selected_count += int(expected_selected)
            correct_count += int(is_success)
            if expected_selected:
                selected_response_tokens.append(response_length)
        print(f"[audit] verified group {group_number}/{len(groups)} id={group_id} successes={len(successes)}")
        selected_lengths = [
            int(rows[idx]["shortest_success_response_tokens"])
            for idx in successes
            if threshold_tokens is not None
            and int(rows[idx]["shortest_success_response_tokens"]) <= threshold_tokens
        ]
        group_details.append(
            {
                "group_id": group_id,
                "successes": len(successes),
                "minimum_success_tokens": min_tokens,
                "threshold_tokens": threshold_tokens,
                "selected": len(selected_lengths),
                "selected_response_tokens": selected_lengths,
            }
        )

    return {
        "rollouts": len(rows),
        "groups": len(groups),
        "successful_groups": successful_groups,
        "correct_rollouts": correct_count,
        "selected_rollouts": selected_count,
        "effective_training_reward_sum": selected_count,
        "selected_fraction": selected_count / len(rows),
        "selected_per_correct": selected_count / correct_count if correct_count else 0.0,
        "groups_with_success_fraction": successful_groups / len(groups),
        "mean_min_success_tokens": (
            sum(successful_group_min_tokens) / len(successful_group_min_tokens)
            if successful_group_min_tokens
            else 0.0
        ),
        "mean_selected_tokens": (
            sum(selected_response_tokens) / len(selected_response_tokens) if selected_response_tokens else 0.0
        ),
        "raw_acc_mean": correct_count / len(rows),
        "prompt_token_reward_count": 0,
        "valid_response_nonzero_reward_token_count": valid_response_nonzero_reward_tokens,
        "pad_token_reward_count": pad_nonzero_reward_tokens,
        "group_details": group_details,
    }


def verify_validation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for idx, row in enumerate(rows):
        if "acc" not in row:
            raise AssertionError(f"validation row {idx} has no acc field")
        _assert_close(row["score"], row["acc"], f"validation row {idx}")
        if any(key.startswith("shortest_success_") for key in row):
            raise AssertionError(f"validation row {idx} contains training-only shortest-success fields")
    return {
        "validation_rows": len(rows),
        "validation_accuracy": sum(float(row["acc"]) for row in rows) / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-jsonl", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--margin-percent", type=float, default=10.0)
    parser.add_argument("--expected-group-size", type=int, default=8)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()

    rollout_summary = verify_rollouts(
        _load_jsonl(args.rollout_jsonl),
        margin_percent=args.margin_percent,
        expected_group_size=args.expected_group_size,
    )
    validation_summary = verify_validation(_load_jsonl(args.validation_jsonl))
    summary = {
        "status": "PASS",
        "margin_percent": args.margin_percent,
        "expected_group_size": args.expected_group_size,
        **rollout_summary,
        **validation_summary,
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "audit_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = WIDE_PAGE_CSS + "\n# Shortest-success reward audit\n\n"
    report += "| Metric | Value |\n| --- | ---: |\n"
    report += "\n".join(
        f"| {key} | {value} |" for key, value in summary.items() if key != "group_details"
    ) + "\n"
    report += "\n## Per-group reconstruction\n\n"
    report += (
        "| Group ID | Successes | Minimum success tokens | Inclusive threshold | "
        "Selected | Selected response lengths |\n"
        "| --- | ---: | ---: | ---: | ---: | --- |\n"
    )
    for group in summary["group_details"]:
        report += (
            f"| {group['group_id']} | {group['successes']} | "
            f"{group['minimum_success_tokens']} | {group['threshold_tokens']} | "
            f"{group['selected']} | {group['selected_response_tokens']} |\n"
        )
    (args.report_dir / "audit_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
