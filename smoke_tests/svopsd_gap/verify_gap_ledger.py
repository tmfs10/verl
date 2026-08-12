#!/usr/bin/env python3
"""Independently verify a full-batch steered-teacher gap ledger."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


WIDE_CSS = """<style>
body, main, article, .markdown-body, .rendered_html,
.jp-RenderedHTMLCommon, .jp-MarkdownOutput {
  max-width: none !important;
  width: min(98vw, 1800px) !important;
}
table { width: 100% !important; }
</style>"""


def read_records(step_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    summary = None
    paths = sorted(step_dir.glob("rank_*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no rank ledgers under {step_dir}")
    for index, path in enumerate(paths, start=1):
        print(f"progress file={index}/{len(paths)} path={path}", flush=True)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if int(record.get("prompt_tokens_logged", 0)) != 0:
                    raise AssertionError(f"prompt token leakage in {path}")
                if int(record.get("masked_or_padded_response_tokens_logged", 0)) != 0:
                    raise AssertionError(f"PAD token leakage in {path}")
                if record.get("record_type") == "sequence_gap_microbatch":
                    if record.get("teacher_source") != "production_steered_worker_forward":
                        raise AssertionError("gap ledger did not use the actual steered worker forward")
                    if record.get("crossfit_source_fold_for_target") != [1, 0]:
                        raise AssertionError("cross-fit vector provenance is not opposite-fold")
                    samples.extend(record.get("samples", []))
                elif record.get("record_type") == "sequence_gap_update_summary":
                    if summary is not None:
                        raise AssertionError("found multiple global update summaries")
                    summary = record
    if summary is None:
        raise AssertionError("missing sequence_gap_update_summary")
    return samples, summary


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def token_mean(rows: list[dict[str, Any]], key: str) -> float:
    tokens = sum(int(row["actual_response_token_count"]) for row in rows)
    return sum(
        float(row[key]) * int(row["actual_response_token_count"]) for row in rows
    ) / tokens


def recompute(samples: list[dict[str, Any]]) -> dict[str, float]:
    positive = [sample for sample in samples if bool(sample["correct"])]
    negative = [sample for sample in samples if not bool(sample["correct"])]
    if not positive or not negative:
        raise AssertionError("full batch does not contain both correctness classes")
    metrics: dict[str, float] = {
        "sample_count": float(len(samples)),
        "positive_count": float(len(positive)),
        "negative_count": float(len(negative)),
        "crossfit_available_count": float(
            sum(bool(sample["crossfit_available"]) for sample in samples)
        ),
    }
    for policy, key in (
        ("actor", "actor_logprob_mean"),
        ("teacher", "teacher_logprob_mean"),
    ):
        pos = mean(positive, key)
        neg = mean(negative, key)
        token_pos = token_mean(positive, key)
        token_neg = token_mean(negative, key)
        metrics[f"{policy}_positive_logprob_mean"] = pos
        metrics[f"{policy}_negative_logprob_mean"] = neg
        metrics[f"{policy}_gap"] = pos - neg
        metrics[f"{policy}_positive_token_weighted_logprob_mean"] = token_pos
        metrics[f"{policy}_negative_token_weighted_logprob_mean"] = token_neg
        metrics[f"{policy}_token_weighted_gap"] = token_pos - token_neg
    metrics["teacher_gap_lift"] = metrics["teacher_gap"] - metrics["actor_gap"]
    metrics["teacher_token_weighted_gap_lift"] = (
        metrics["teacher_token_weighted_gap"] - metrics["actor_token_weighted_gap"]
    )
    metrics["positive_response_length_mean"] = mean(
        positive, "actual_response_token_count"
    )
    metrics["negative_response_length_mean"] = mean(
        negative, "actual_response_token_count"
    )

    crossfit_positive = [row for row in positive if bool(row["crossfit_available"])]
    crossfit_negative = [row for row in negative if bool(row["crossfit_available"])]
    if crossfit_positive and crossfit_negative:
        pos = mean(crossfit_positive, "crossfit_teacher_logprob_mean")
        neg = mean(crossfit_negative, "crossfit_teacher_logprob_mean")
        actor_pos = mean(crossfit_positive, "actor_logprob_mean")
        actor_neg = mean(crossfit_negative, "actor_logprob_mean")
        token_pos = token_mean(
            crossfit_positive, "crossfit_teacher_logprob_mean"
        )
        token_neg = token_mean(
            crossfit_negative, "crossfit_teacher_logprob_mean"
        )
        actor_token_pos = token_mean(crossfit_positive, "actor_logprob_mean")
        actor_token_neg = token_mean(crossfit_negative, "actor_logprob_mean")
        metrics["crossfit_teacher_positive_logprob_mean"] = pos
        metrics["crossfit_teacher_negative_logprob_mean"] = neg
        metrics["crossfit_teacher_gap"] = pos - neg
        metrics["crossfit_actor_gap"] = actor_pos - actor_neg
        metrics["crossfit_teacher_gap_lift"] = (pos - neg) - (actor_pos - actor_neg)
        metrics["crossfit_teacher_positive_token_weighted_logprob_mean"] = token_pos
        metrics["crossfit_teacher_negative_token_weighted_logprob_mean"] = token_neg
        metrics["crossfit_teacher_token_weighted_gap"] = token_pos - token_neg
        metrics["crossfit_actor_token_weighted_gap"] = (
            actor_token_pos - actor_token_neg
        )
        metrics["crossfit_teacher_token_weighted_gap_lift"] = (
            (token_pos - token_neg) - (actor_token_pos - actor_token_neg)
        )

    prompts: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        prompts.setdefault(int(sample["prompt_group"]), []).append(sample)
    actor_gaps = []
    teacher_gaps = []
    crossfit_actor_gaps = []
    crossfit_teacher_gaps = []
    for rows in prompts.values():
        pos = [row for row in rows if bool(row["correct"])]
        neg = [row for row in rows if not bool(row["correct"])]
        if not pos or not neg:
            continue
        actor_gaps.append(mean(pos, "actor_logprob_mean") - mean(neg, "actor_logprob_mean"))
        teacher_gaps.append(
            mean(pos, "teacher_logprob_mean") - mean(neg, "teacher_logprob_mean")
        )
        if all(bool(row["crossfit_available"]) for row in rows):
            crossfit_actor_gaps.append(
                mean(pos, "actor_logprob_mean") - mean(neg, "actor_logprob_mean")
            )
            crossfit_teacher_gaps.append(
                mean(pos, "crossfit_teacher_logprob_mean")
                - mean(neg, "crossfit_teacher_logprob_mean")
            )
    metrics["mixed_prompt_count"] = float(len(actor_gaps))
    if actor_gaps:
        metrics["prompt_balanced_actor_gap"] = sum(actor_gaps) / len(actor_gaps)
        metrics["prompt_balanced_teacher_gap"] = sum(teacher_gaps) / len(teacher_gaps)
        metrics["prompt_balanced_teacher_gap_lift"] = (
            metrics["prompt_balanced_teacher_gap"]
            - metrics["prompt_balanced_actor_gap"]
        )
    metrics["crossfit_mixed_prompt_count"] = float(len(crossfit_actor_gaps))
    if crossfit_actor_gaps:
        metrics["prompt_balanced_crossfit_actor_gap"] = sum(crossfit_actor_gaps) / len(
            crossfit_actor_gaps
        )
        metrics["prompt_balanced_crossfit_teacher_gap"] = sum(
            crossfit_teacher_gaps
        ) / len(crossfit_teacher_gaps)
        metrics["prompt_balanced_crossfit_teacher_gap_lift"] = (
            metrics["prompt_balanced_crossfit_teacher_gap"]
            - metrics["prompt_balanced_crossfit_actor_gap"]
        )
    return metrics


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-samples", required=True, type=int)
    parser.add_argument("--expected-scale", default=0.5, type=float)
    parser.add_argument("--atol", default=1e-7, type=float)
    args = parser.parse_args()

    samples, summary = read_records(args.step_dir)
    if len(samples) != args.expected_samples:
        raise AssertionError(f"sample count {len(samples)} != {args.expected_samples}")
    indices = [int(sample["sample_index"]) for sample in samples]
    if len(set(indices)) != len(indices):
        raise AssertionError("sample indices are not unique")
    for sample in samples:
        for key in (
            "actor_logprob_mean",
            "teacher_logprob_mean",
            "teacher_minus_actor_logprob_mean",
        ):
            if not math.isfinite(float(sample[key])):
                raise AssertionError(f"non-finite {key} in sample {sample['sample_index']}")
        expected_delta = float(sample["teacher_logprob_mean"]) - float(
            sample["actor_logprob_mean"]
        )
        if not math.isclose(
            float(sample["teacher_minus_actor_logprob_mean"]),
            expected_delta,
            abs_tol=args.atol,
            rel_tol=args.atol,
        ):
            raise AssertionError("per-sample teacher-minus-actor arithmetic mismatch")

    fold_counts = summary.get("crossfit_global_fold_rollout_counts")
    if fold_counts is None or len(fold_counts) != 2:
        raise AssertionError("missing two-fold source counts")
    target_available = [
        bool(fold_counts[1][0] > 0 and fold_counts[1][1] > 0),
        bool(fold_counts[0][0] > 0 and fold_counts[0][1] > 0),
    ]
    for sample in samples:
        expected = target_available[int(sample["crossfit_fold"])]
        if bool(sample["crossfit_available"]) != expected:
            raise AssertionError("cross-fit availability disagrees with opposite-fold counts")

    vector_norms = summary.get("crossfit_vector_norms") or []
    if not vector_norms:
        raise AssertionError("missing cross-fit vector norms")
    for index, norm in enumerate(vector_norms):
        target_fold = index % 2
        expected_norm = 1.0 if target_available[target_fold] else 0.0
        if not math.isclose(float(norm), expected_norm, abs_tol=1e-6, rel_tol=1e-6):
            raise AssertionError(
                f"cross-fit vector norm {norm} != {expected_norm} at index {index}"
            )
        applied_norm = float(norm) * float(summary["steering_scale"])
        expected_applied = args.expected_scale if target_available[target_fold] else 0.0
        if not math.isclose(applied_norm, expected_applied, abs_tol=1e-6, rel_tol=1e-6):
            raise AssertionError("cross-fit applied vector norm mismatch")

    recomputed = recompute(samples)
    recorded = summary["metrics"]
    if set(recomputed) != set(recorded):
        raise AssertionError(
            f"summary metric keys differ: missing={set(recomputed)-set(recorded)} "
            f"extra={set(recorded)-set(recomputed)}"
        )
    for key, value in recomputed.items():
        if not math.isclose(float(recorded[key]), value, abs_tol=args.atol, rel_tol=args.atol):
            raise AssertionError(
                f"metric mismatch {key}: recorded={recorded[key]} recomputed={value}"
            )
    if summary.get("diagnostic_only") is not True or summary.get("gradient_route") != "none":
        raise AssertionError("cross-fit diagnostic is not marked gradient-free")

    report = {
        "status": "PASS",
        "step_dir": str(args.step_dir),
        "sample_count": len(samples),
        "fold_counts": fold_counts,
        "target_available": target_available,
        "crossfit_vector_norms": vector_norms,
        "steering_scale": summary["steering_scale"],
        "metrics": recomputed,
        "prompt_tokens_logged": 0,
        "masked_or_padded_response_tokens_logged": 0,
    }
    atomic_write(args.output_dir / "audit_report.json", json.dumps(report, indent=2, sort_keys=True))
    lines = [
        WIDE_CSS,
        "# Steered-teacher gap audit",
        "",
        "Status: **PASS**",
        "",
        f"- Samples: {len(samples)}",
        f"- Actor gap: `{recomputed['actor_gap']:.9g}`",
        f"- Production teacher gap: `{recomputed['teacher_gap']:.9g}`",
        f"- Production gap lift: `{recomputed['teacher_gap_lift']:.9g}`",
        f"- Cross-fit gap lift: `{recomputed.get('crossfit_teacher_gap_lift', float('nan')):.9g}`",
        "- Prompt tokens logged: `0`",
        "- PAD tokens logged: `0`",
        "- Cross-fit gradient route: `none`",
        "",
    ]
    atomic_write(args.output_dir / "audit_report.md", "\n".join(lines))
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
