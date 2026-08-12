#!/usr/bin/env python3
"""Summarize production-shape OPSD throughput smoke logs.

The input job manifest and accounting table are emitted by
``profile_opsd_throughput_cwdfw.sh``.  This parser deliberately uses VERL's
ordinary console timing metrics so profiling does not insert CUDA
synchronizations or enable the large OPSD audit ledger.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


STEP_RE = re.compile(r"(?:^|\s)step:(\d+)\s+-\s+")
METRIC_RE = re.compile(
    r"(?:^|\s-\s)([A-Za-z0-9_./@-]+):"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)
REQUIRED_VARIANTS = ("v1_vanilla_grpo", "v2_shared_teacher", "v3_separate_sft", "v4_separate_rlvr")
WALLTIME_SECONDS = 4 * 60 * 60
SAFETY_SECONDS = 15 * 60


@dataclass(frozen=True)
class JobRecord:
    variant: str
    job_id: str
    expname: str
    remote_output: str


@dataclass(frozen=True)
class VariantSummary:
    variant: str
    source: str
    job_id: str
    elapsed_seconds: float
    fixed_overhead_seconds: float
    cold_step_seconds: float
    steady_step_seconds_median: float
    steady_step_seconds_max: float
    checkpoint_seconds: float
    expected_iterations_4h: int
    conservative_iterations_4h: int
    expected_post_warmup_iterations_4h: int | None
    conservative_post_warmup_iterations_4h: int | None
    step_metrics: dict[str, dict[str, float]]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_jobs(path: Path) -> dict[str, JobRecord]:
    jobs: dict[str, JobRecord] = {}
    for row in read_tsv(path):
        record = JobRecord(**{key: row[key] for key in ("variant", "job_id", "expname", "remote_output")})
        if record.variant in jobs:
            raise ValueError(f"duplicate variant in {path}: {record.variant}")
        jobs[record.variant] = record
    missing = sorted(set(REQUIRED_VARIANTS) - set(jobs))
    if missing:
        raise ValueError(f"job manifest is missing variants: {missing}")
    return jobs


def read_elapsed_seconds(path: Path) -> dict[str, float]:
    elapsed: dict[str, float] = {}
    for row in read_tsv(path):
        if not row.get("job_id"):
            continue
        elapsed[row["job_id"]] = float(row["elapsed_raw"])
    return elapsed


def parse_step_metrics(lines: Iterable[str]) -> dict[int, dict[str, float]]:
    steps: dict[int, dict[str, float]] = {}
    for line in lines:
        step_match = STEP_RE.search(line)
        if step_match is None:
            continue
        step = int(step_match.group(1))
        metrics = {key: float(value) for key, value in METRIC_RE.findall(line)}
        metrics["step"] = float(step)
        steps[step] = metrics
    return steps


def find_main_log(log_root: Path, job_id: str) -> Path:
    candidates = sorted(log_root.rglob(f"main_*_{job_id}_srun.log"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one main srun log for job {job_id} below {log_root}, found {candidates}"
        )
    return candidates[0]


def iteration_capacity(
    *,
    fixed_overhead: float,
    checkpoint: float,
    cold_step: float,
    steady_step: float,
) -> int:
    cold_premium = max(0.0, cold_step - steady_step)
    available = WALLTIME_SECONDS - fixed_overhead - checkpoint - SAFETY_SECONDS - cold_premium
    return max(0, math.floor(available / steady_step))


def summarize_measured(
    *,
    job: JobRecord,
    elapsed_seconds: float,
    log_root: Path,
) -> VariantSummary:
    main_log = find_main_log(log_root, job.job_id)
    steps = parse_step_metrics(main_log.read_text(errors="replace", encoding="utf-8").splitlines())
    missing_steps = sorted(set(range(1, 6)) - set(steps))
    if missing_steps:
        raise ValueError(f"{job.variant} job {job.job_id} is missing steps {missing_steps} in {main_log}")
    for step in range(1, 6):
        if "timing_s/step" not in steps[step]:
            raise ValueError(f"{job.variant} step {step} lacks timing_s/step")
    checkpoint = steps[5].get("timing_s/save_checkpoint")
    if checkpoint is None or checkpoint <= 0:
        raise ValueError(f"{job.variant} step 5 lacks a positive timing_s/save_checkpoint")

    step_seconds = [steps[step]["timing_s/step"] for step in range(1, 6)]
    steady = step_seconds[1:4]
    steady_median = statistics.median(steady)
    steady_max = max(steady)
    fixed_overhead = max(0.0, elapsed_seconds - sum(step_seconds))
    encoded_steps = {str(step): steps[step] for step in range(1, 6)}
    return VariantSummary(
        variant=job.variant,
        source="measured",
        job_id=job.job_id,
        elapsed_seconds=elapsed_seconds,
        fixed_overhead_seconds=fixed_overhead,
        cold_step_seconds=step_seconds[0],
        steady_step_seconds_median=steady_median,
        steady_step_seconds_max=steady_max,
        checkpoint_seconds=checkpoint,
        expected_iterations_4h=iteration_capacity(
            fixed_overhead=fixed_overhead,
            checkpoint=checkpoint,
            cold_step=step_seconds[0],
            steady_step=steady_median,
        ),
        conservative_iterations_4h=iteration_capacity(
            fixed_overhead=fixed_overhead,
            checkpoint=checkpoint,
            cold_step=step_seconds[0],
            steady_step=steady_max,
        ),
        expected_post_warmup_iterations_4h=None,
        conservative_post_warmup_iterations_4h=None,
        step_metrics=encoded_steps,
    )


def extrapolate_warmup(source: VariantSummary) -> VariantSummary:
    expected_total = source.expected_iterations_4h
    conservative_total = source.conservative_iterations_4h
    return VariantSummary(
        variant="v5_separate_sft_warmup30",
        source="extrapolated conservatively from v3; warmup charged at full v3 steady-step cost",
        job_id=source.job_id,
        elapsed_seconds=source.elapsed_seconds,
        fixed_overhead_seconds=source.fixed_overhead_seconds,
        cold_step_seconds=source.cold_step_seconds,
        steady_step_seconds_median=source.steady_step_seconds_median,
        steady_step_seconds_max=source.steady_step_seconds_max,
        checkpoint_seconds=source.checkpoint_seconds,
        expected_iterations_4h=expected_total,
        conservative_iterations_4h=conservative_total,
        expected_post_warmup_iterations_4h=max(0, expected_total - 30),
        conservative_post_warmup_iterations_4h=max(0, conservative_total - 30),
        step_metrics={},
    )


def markdown_report(summaries: list[VariantSummary]) -> str:
    css = """<style>
body, main, article, .markdown-body, .rendered_html, .jp-RenderedHTMLCommon, .jp-MarkdownOutput {
  max-width: none !important;
  width: min(98vw, 1800px) !important;
}
table { width: 100% !important; }
</style>"""
    rows = []
    for item in summaries:
        post_warmup = "—"
        if item.expected_post_warmup_iterations_4h is not None:
            post_warmup = (
                f"{item.expected_post_warmup_iterations_4h} expected / "
                f"{item.conservative_post_warmup_iterations_4h} conservative"
            )
        rows.append(
            "| {variant} | {source} | {job} | {elapsed:.1f} | {fixed:.1f} | {cold:.1f} | "
            "{median:.1f} | {maximum:.1f} | {checkpoint:.1f} | {expected} | {conservative} | {post} |".format(
                variant=item.variant,
                source=item.source,
                job=item.job_id,
                elapsed=item.elapsed_seconds,
                fixed=item.fixed_overhead_seconds,
                cold=item.cold_step_seconds,
                median=item.steady_step_seconds_median,
                maximum=item.steady_step_seconds_max,
                checkpoint=item.checkpoint_seconds,
                expected=item.expected_iterations_4h,
                conservative=item.conservative_iterations_4h,
                post=post_warmup,
            )
        )
    return "\n".join(
        [
            css,
            "",
            "# OPSD production-shape throughput smoke",
            "",
            "The capacity calculation reserves 900 seconds, one measured checkpoint, fixed job/validation overhead, and the cold-step premium.",
            "",
            "| Variant | Source | Job | Slurm elapsed (s) | Fixed overhead (s) | Cold step (s) | Steady median (s) | Steady max (s) | Checkpoint (s) | Expected updates/4h | Conservative updates/4h | Post-warmup updates/4h |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            *rows,
            "",
            "Variant 5 is not directly timed. Its first 30 SFT-only updates omit the student backward, but are deliberately charged at the full variant-3 step cost.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--accounting", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = read_jobs(args.jobs)
    elapsed = read_elapsed_seconds(args.accounting)
    summaries = []
    for variant in REQUIRED_VARIANTS:
        job = jobs[variant]
        if job.job_id not in elapsed:
            raise ValueError(f"accounting table lacks root job {job.job_id}")
        summaries.append(
            summarize_measured(job=job, elapsed_seconds=elapsed[job.job_id], log_root=args.log_root)
        )
    summaries.append(extrapolate_warmup(next(item for item in summaries if item.variant == "v3_separate_sft")))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "throughput_report.json"
    md_path = args.output_dir / "throughput_report.md"
    json_path.write_text(json.dumps([asdict(item) for item in summaries], indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(summaries), encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
