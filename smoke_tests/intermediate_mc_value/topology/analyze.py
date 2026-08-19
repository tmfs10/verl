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

"""Validate, summarize, and rank collected intermediate-MC topology runs."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

EXPECTED_GPU_NAME = "A100-SXM4-80GB"
WIDE_CSS = """<style>
body, main, article, .markdown-body, .rendered_html, .jp-RenderedHTMLCommon, .jp-MarkdownOutput {
  max-width: none !important;
  width: 96vw !important;
}
table { width: 100% !important; }
</style>
"""


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        candidate = json.loads(line)
        candidate_id = candidate["candidate_id"]
        if candidate_id in result:
            raise ValueError(f"duplicate manifest candidate {candidate_id}")
        result[candidate_id] = candidate
    if not result:
        raise ValueError("candidate manifest is empty")
    return result


def _finite_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value))


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_median_ci95(values: list[float], *, seed: int = 20260819, resamples: int = 5000) -> list[float] | None:
    if len(values) < 3:
        return None
    rng = random.Random(seed)
    medians = [statistics.median(rng.choices(values, k=len(values))) for _ in range(resamples)]
    return [_percentile(medians, 0.025), _percentile(medians, 0.975)]


def _parse_gpu_names(snapshot: dict[str, Any]) -> list[str]:
    query = snapshot.get("nvidia_smi_query", {})
    if query.get("returncode") != 0:
        return []
    names: list[str] = []
    for line in str(query.get("stdout", "")).splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 3:
            names.append(fields[2])
    return names


def _validate_hardware(path: Path, expected_nodes: int) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    if not path.is_file():
        return [], ["missing ray_hardware_after.json"]
    snapshots = json.loads(path.read_text(encoding="utf-8"))
    if len(snapshots) != expected_nodes:
        errors.append(f"Ray saw {len(snapshots)} nodes, expected {expected_nodes}")
    hostnames = [str(snapshot.get("hostname", "")) for snapshot in snapshots]
    if len(set(hostnames)) != len(hostnames):
        errors.append(f"Ray hardware snapshots contain duplicate hostnames: {hostnames}")
    for snapshot in snapshots:
        names = _parse_gpu_names(snapshot)
        if len(names) != 8:
            errors.append(f"{snapshot.get('hostname')}: saw {len(names)} GPUs, expected 8")
        unexpected = [name for name in names if EXPECTED_GPU_NAME not in name]
        if unexpected:
            errors.append(f"{snapshot.get('hostname')}: unexpected GPU names {unexpected}")
    return hostnames, errors


def _load_metrics(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record.get("data"), dict) or not isinstance(record.get("step"), int):
            raise ValueError(f"{path}:{line_number}: malformed VeRL file-logger record")
        records.append(record)
    return records


def _median_metric(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record["data"][key]) for record in records if _finite_number(record["data"].get(key))]
    return statistics.median(values) if values else None


def analyze_candidate(candidate: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    candidate_id = candidate["candidate_id"]
    errors: list[str] = []
    if (run_dir / "failed.json").exists():
        errors.append("training entrypoint wrote failed.json")
    if not (run_dir / "completed.json").is_file():
        errors.append("missing completed.json")
    hostnames, hardware_errors = _validate_hardware(run_dir / "ray_hardware_after.json", candidate["topology"]["nodes"])
    errors.extend(hardware_errors)

    metrics_path = run_dir / "metrics.jsonl"
    records: list[dict[str, Any]] = []
    if not metrics_path.is_file():
        errors.append("missing metrics.jsonl")
    else:
        try:
            records = _load_metrics(metrics_path)
        except Exception as error:  # noqa: BLE001 - report every independent rejection reason
            errors.append(f"invalid metrics JSONL: {error}")

    expected_steps = list(range(1, int(candidate["total_steps"]) + 1))
    actual_steps = [record["step"] for record in records]
    if actual_steps != expected_steps:
        errors.append(f"metric steps {actual_steps} != expected {expected_steps}")
    measured = [record for record in records if record["step"] > int(candidate["stabilization_steps"])]
    if len(measured) != int(candidate["measured_steps"]):
        errors.append(f"found {len(measured)} measured steps, expected {candidate['measured_steps']}")

    num_critiques = int(candidate["workload"]["num_critiques"])
    for record in records:
        step = record["step"]
        data = record["data"]
        requirements = {
            "training/global_step": step,
            "intermediate_mc/warmup": 0.0,
            "intermediate_mc/bundles": 512,
            "intermediate_mc/critiques": 512 * num_critiques,
        }
        for key, expected in requirements.items():
            if data.get(key) != expected:
                errors.append(f"step {step}: {key}={data.get(key)!r}, expected {expected!r}")
        for key in (
            "timing_s/step",
            "actor/grad_norm",
            "critic/grad_norm",
            "intermediate_mc/tokens/generation_output",
            "intermediate_mc/tokens/critic_input",
            "intermediate_mc/tokens/actor_train",
        ):
            if not _finite_number(data.get(key)) or float(data[key]) <= 0.0:
                errors.append(f"step {step}: missing, non-finite, or non-positive {key}")
        if num_critiques == 0:
            for key in ("intermediate_mc/tokens/critique_input", "intermediate_mc/tokens/critique_output"):
                if data.get(key) != 0.0:
                    errors.append(f"step {step}: M0 requires {key}=0, got {data.get(key)!r}")
        selected = data.get("intermediate_mc/selected_marks")
        attempts = data.get("intermediate_mc/continuation_attempts")
        if not _finite_number(selected) or not 0 < float(selected) <= 512:
            errors.append(f"step {step}: selected mark count is invalid: {selected!r}")
        if selected != attempts:
            errors.append(f"step {step}: continuation attempts {attempts!r} != selected marks {selected!r}")

    step_seconds = [
        float(record["data"]["timing_s/step"])
        for record in measured
        if _finite_number(record["data"].get("timing_s/step"))
    ]
    generation_output = [
        float(record["data"]["intermediate_mc/tokens/generation_output"])
        for record in measured
        if _finite_number(record["data"].get("intermediate_mc/tokens/generation_output"))
    ]
    median_step = statistics.median(step_seconds) if step_seconds else None
    iterations_per_hour = 3600.0 / median_step if median_step and median_step > 0 else None
    median_generation_output = statistics.median(generation_output) if generation_output else None
    tokens_per_hour = (
        3600.0
        * statistics.median(tokens / seconds for tokens, seconds in zip(generation_output, step_seconds, strict=False))
        if generation_output and step_seconds
        else None
    )
    gpus = int(candidate["topology"]["nodes"]) * 8
    phase_keys = (
        "timing_s/gen",
        "timing_s/intermediate_mc_continuations",
        "timing_s/values",
        "timing_s/update_critic",
        "timing_s/update_actor",
        "timing_s/update_weights",
        "timing_s/reward",
    )
    phase_medians = {key: _median_metric(measured, key) for key in phase_keys}
    phase_fractions = {
        key: value / median_step
        for key, value in phase_medians.items()
        if value is not None and median_step is not None and median_step > 0
    }
    max_reserved = _median_metric(measured, "perf/max_memory_reserved_gb")
    summary = {
        "candidate_id": candidate_id,
        "workload_id": candidate["workload_id"],
        "phase": candidate["phase"],
        "seed": candidate["seed"],
        "nodes": candidate["topology"]["nodes"],
        "gpus": gpus,
        "topology_id": candidate["topology"]["topology_id"],
        "profile_id": candidate["batch_profile"]["profile_id"],
        "topology": candidate["topology"],
        "batch_profile": candidate["batch_profile"],
        "num_critiques": num_critiques,
        "valid": not errors,
        "rejection_reasons": sorted(set(errors)),
        "hostnames": hostnames,
        "measured_steps": len(measured),
        "measured_step_seconds": step_seconds,
        "median_step_seconds": median_step,
        "p95_step_seconds": _percentile(step_seconds, 0.95) if step_seconds else None,
        "iterations_per_hour": iterations_per_hour,
        "gpu_hours_per_iteration": gpus * median_step / 3600.0 if median_step else None,
        "median_generation_output_tokens": median_generation_output,
        "generation_output_tokens_per_hour": tokens_per_hour,
        "phase_median_seconds": phase_medians,
        "phase_fractions": phase_fractions,
        "max_memory_reserved_gb": max_reserved,
        "estimated_memory_headroom_gb": 80.0 - max_reserved if max_reserved is not None else None,
    }
    return summary


def _config_key(result: dict[str, Any]) -> tuple[object, ...]:
    return (result["workload_id"], result["nodes"], result["topology_id"], result["profile_id"])


def aggregate_repeats(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        if result["valid"]:
            groups[_config_key(result)].append(result)
    aggregated: list[dict[str, Any]] = []
    for key, group in groups.items():
        step_values = [float(item["median_step_seconds"]) for item in group]
        all_step_values = [value for item in group for value in item["measured_step_seconds"]]
        token_values = [float(item["median_generation_output_tokens"]) for item in group]
        median_step = statistics.median(step_values)
        aggregated.append(
            {
                "workload_id": key[0],
                "nodes": key[1],
                "topology_id": key[2],
                "profile_id": key[3],
                "topology": group[0]["topology"],
                "batch_profile": group[0]["batch_profile"],
                "runs": len(group),
                "seeds": sorted(item["seed"] for item in group),
                "median_step_seconds": median_step,
                "p95_run_median_step_seconds": _percentile(step_values, 0.95),
                "p95_step_seconds": _percentile(all_step_values, 0.95),
                "median_step_seconds_ci95": _bootstrap_median_ci95(step_values),
                "iterations_per_hour": 3600.0 / median_step,
                "median_generation_output_tokens": statistics.median(token_values),
                "candidate_ids": [item["candidate_id"] for item in group],
            }
        )
    return sorted(aggregated, key=lambda item: (item["workload_id"], item["nodes"], -item["iterations_per_hour"]))


def _complexity_key(item: dict[str, Any]) -> tuple[object, ...]:
    topology = item["topology"]
    profile = item["batch_profile"]
    return (
        int(topology["rollout_tp"]) > 1,
        int(topology["sequence_parallel_size"]),
        int(topology["actor_fsdp_size"]) != int(topology["critic_fsdp_size"]),
        topology["strategy"] != "fsdp2",
        profile["profile_id"] not in {"P01", "P02"},
        item["topology_id"],
        item["profile_id"],
    )


def _confidence_intervals_overlap(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_ci = first["median_step_seconds_ci95"]
    second_ci = second["median_step_seconds_ci95"]
    if first_ci is None or second_ci is None:
        return True
    return not (first_ci[1] < second_ci[0] or second_ci[1] < first_ci[0])


def _recommendations(aggregated: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in aggregated:
        groups[(item["workload_id"], int(item["nodes"]))].append(item)
    recommendations = []
    for (workload_id, nodes), group in sorted(groups.items()):
        raw_best = max(group, key=lambda item: item["iterations_per_hour"])
        contenders = [
            item
            for item in group
            if item["iterations_per_hour"] >= 0.97 * raw_best["iterations_per_hour"]
            and _confidence_intervals_overlap(item, raw_best)
        ]
        selected = min(
            contenders,
            key=lambda item: (item["p95_step_seconds"], _complexity_key(item)),
        )
        recommendations.append(
            {
                "workload_id": workload_id,
                "nodes": nodes,
                "selected_topology_id": selected["topology_id"],
                "selected_profile_id": selected["profile_id"],
                "selected_iterations_per_hour": selected["iterations_per_hour"],
                "raw_best_topology_id": raw_best["topology_id"],
                "raw_best_profile_id": raw_best["profile_id"],
                "tie_contenders": [
                    {"topology_id": item["topology_id"], "profile_id": item["profile_id"]}
                    for item in sorted(contenders, key=_complexity_key)
                ],
                "selection_rule": "within 3% with overlapping/insufficient CI, then p95 latency and simplicity",
            }
        )
    return recommendations


def _apply_token_drift_rejection(results: list[dict[str, Any]]) -> None:
    groups: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        if result["valid"] and result["median_generation_output_tokens"]:
            groups[(result["workload_id"], result["seed"], result["nodes"])].append(result)
    for group in groups.values():
        if len(group) < 2:
            continue
        reference = statistics.median(float(item["median_generation_output_tokens"]) for item in group)
        for result in group:
            deviation = abs(float(result["median_generation_output_tokens"]) - reference) / reference
            result["generation_token_deviation_from_group_median"] = deviation
            if deviation > 0.10:
                result["valid"] = False
                result["rejection_reasons"].append(
                    f"generation output token volume differs {deviation:.1%} from matched group median"
                )


def _critique_overheads(aggregated: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pair: dict[tuple[object, ...], dict[int, dict[str, Any]]] = defaultdict(dict)
    for item in aggregated:
        prefix, m_value = item["workload_id"].rsplit("-m", 1)
        by_pair[(prefix, item["nodes"], item["topology_id"], item["profile_id"])][int(m_value)] = item
    overheads = []
    for key, modes in by_pair.items():
        if set(modes) != {0, 4}:
            continue
        overheads.append(
            {
                "workload_prefix": key[0],
                "nodes": key[1],
                "topology_id": key[2],
                "profile_id": key[3],
                "m0_iterations_per_hour": modes[0]["iterations_per_hour"],
                "m4_iterations_per_hour": modes[4]["iterations_per_hour"],
                "critique_overhead_fraction": 1.0 - modes[4]["iterations_per_hour"] / modes[0]["iterations_per_hour"],
            }
        )
    return sorted(overheads, key=lambda item: (item["workload_prefix"], item["nodes"], item["topology_id"]))


def _scaling_decisions(aggregated: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_workload: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for item in aggregated:
        by_workload[item["workload_id"]][int(item["nodes"])].append(item)
    decisions = []
    for workload_id, by_nodes in by_workload.items():
        if not by_nodes[2] or not by_nodes[4]:
            continue
        best_two = max(by_nodes[2], key=lambda item: item["iterations_per_hour"])
        best_four = max(by_nodes[4], key=lambda item: item["iterations_per_hour"])
        improvement = best_four["iterations_per_hour"] / best_two["iterations_per_hour"] - 1.0
        decisions.append(
            {
                "workload_id": workload_id,
                "best_two_node": best_two,
                "best_four_node": best_four,
                "raw_iteration_improvement_fraction": improvement,
                "select_four_nodes": improvement >= 0.05,
            }
        )
    return decisions


def _markdown(payload: dict[str, Any]) -> str:
    lines = [WIDE_CSS, "# Intermediate-MC topology benchmark", ""]
    lines.append(
        "Only valid runs are ranked. M0 is the feature-enabled, no-self-critique baseline: it still trains the "
        "unconditioned critic and still generates one continuation at one selected mark."
    )
    lines.extend(["", "## Ranked configurations", ""])
    lines.append("| Workload | Nodes | Topology | Profile | Runs | Median s/iter | Iterations/hour | p95 run median |")
    lines.append("|---|---:|---|---|---:|---:|---:|---:|")
    for item in payload["aggregated"]:
        lines.append(
            f"| {item['workload_id']} | {item['nodes']} | {item['topology_id']} | {item['profile_id']} | "
            f"{item['runs']} | {item['median_step_seconds']:.3f} | {item['iterations_per_hour']:.3f} | "
            f"{item['p95_run_median_step_seconds']:.3f} |"
        )
    lines.extend(["", "## Rejected or incomplete runs", ""])
    rejected = [item for item in payload["runs"] if not item["valid"]]
    if not rejected:
        lines.append("None.")
    else:
        lines.append("| Candidate | Reasons |")
        lines.append("|---|---|")
        for item in rejected:
            lines.append(f"| {item['candidate_id']} | {'; '.join(item['rejection_reasons'])} |")
    lines.extend(["", "## Per-workload recommendations", ""])
    if not payload["recommendations"]:
        lines.append("No valid configuration is available.")
    else:
        lines.append("| Workload | Nodes | Selected topology | Profile | Iterations/hour | Tie contenders |")
        lines.append("|---|---:|---|---|---:|---|")
        for item in payload["recommendations"]:
            contenders = ", ".join(
                f"{candidate['topology_id']}/{candidate['profile_id']}" for candidate in item["tie_contenders"]
            )
            lines.append(
                f"| {item['workload_id']} | {item['nodes']} | {item['selected_topology_id']} | "
                f"{item['selected_profile_id']} | {item['selected_iterations_per_hour']:.3f} | {contenders} |"
            )
    lines.extend(["", "## Matched M4 versus M0 overhead", ""])
    if not payload["critique_overheads"]:
        lines.append("No complete matched M0/M4 configuration pairs were collected.")
    else:
        lines.append("| Workload | Nodes | Topology | Profile | M0 iter/h | M4 iter/h | M4 overhead |")
        lines.append("|---|---:|---|---|---:|---:|---:|")
        for item in payload["critique_overheads"]:
            lines.append(
                f"| {item['workload_prefix']} | {item['nodes']} | {item['topology_id']} | {item['profile_id']} | "
                f"{item['m0_iterations_per_hour']:.3f} | {item['m4_iterations_per_hour']:.3f} | "
                f"{item['critique_overhead_fraction']:.1%} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--collected-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    results = [
        analyze_candidate(candidate, args.collected_root / candidate_id) for candidate_id, candidate in manifest.items()
    ]
    _apply_token_drift_rejection(results)
    aggregated = aggregate_repeats(results)
    payload = {
        "runs": results,
        "aggregated": aggregated,
        "recommendations": _recommendations(aggregated),
        "critique_overheads": _critique_overheads(aggregated),
        "scaling_decisions": _scaling_decisions(aggregated),
        "acceptance": {
            "four_node_minimum_raw_iteration_improvement_fraction": 0.05,
            "token_volume_rerun_threshold_fraction": 0.10,
            "tie_threshold_fraction": 0.03,
            "expected_gpu": EXPECTED_GPU_NAME,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "runs": len(results),
                "valid": sum(item["valid"] for item in results),
                "rejected": sum(not item["valid"] for item in results),
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
