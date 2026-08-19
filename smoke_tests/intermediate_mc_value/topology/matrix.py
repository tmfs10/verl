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

"""Define and materialize the intermediate-MC topology search space."""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROMPTS_PER_ITERATION = 64
ROLLOUTS_PER_PROMPT = 8
SOLUTIONS_PER_ITERATION = PROMPTS_PER_ITERATION * ROLLOUTS_PER_PROMPT
PROMPT_TOKENS = 4096
SOLUTION_TOKENS = 8192
CRITIQUE_TOKENS = 8192
ROLLOUT_MAX_MODEL_LEN = 24576
MEASURED_STEPS = 2
STABILIZATION_STEPS = 1
TOTAL_STEPS = STABILIZATION_STEPS + MEASURED_STEPS


@dataclass(frozen=True)
class Workload:
    model: str
    model_path: str
    recipe: str
    critic_head: str
    mark_selector: str
    num_critiques: int

    @property
    def workload_id(self) -> str:
        return f"{self.model}-{self.recipe}-m{self.num_critiques}"


@dataclass(frozen=True)
class Topology:
    topology_id: str
    nodes: int
    strategy: str
    actor_fsdp_size: int
    critic_fsdp_size: int
    rollout_tp: int
    sequence_parallel_size: int
    source_topology: str | None = None

    @property
    def world_size(self) -> int:
        return self.nodes * 8

    def validate(self) -> None:
        if self.nodes not in {2, 4}:
            raise ValueError(f"{self.topology_id}: only two- and four-node candidates are supported")
        if self.strategy not in {"fsdp", "fsdp2"}:
            raise ValueError(f"{self.topology_id}: unsupported strategy {self.strategy}")
        for role, size in (("actor", self.actor_fsdp_size), ("critic", self.critic_fsdp_size)):
            if size <= 0 or size > self.world_size or self.world_size % size:
                raise ValueError(f"{self.topology_id}: invalid {role} FSDP size {size}")
        if self.rollout_tp <= 0 or self.world_size % self.rollout_tp:
            raise ValueError(f"{self.topology_id}: rollout TP must divide world size")
        if self.sequence_parallel_size <= 0 or self.world_size % self.sequence_parallel_size:
            raise ValueError(f"{self.topology_id}: sequence parallel size must divide world size")
        dp_size = self.world_size // self.sequence_parallel_size
        global_minibatch = PROMPTS_PER_ITERATION * ROLLOUTS_PER_PROMPT
        if global_minibatch % dp_size:
            raise ValueError(f"{self.topology_id}: global minibatch does not divide actor/critic DP size")


@dataclass(frozen=True)
class BatchProfile:
    profile_id: str
    actor_dynamic: bool
    critic_dynamic: bool
    actor_token_cap: int
    critic_token_cap: int
    rollout_batched_tokens: int
    rollout_max_num_seqs: int
    rollout_gpu_memory_utilization: float
    rollout_enforce_eager: bool
    gradient_checkpointing: bool
    reshard_after_forward: bool
    gated_by_memory_headroom: bool = False

    def validate(self) -> None:
        if self.actor_token_cap < ROLLOUT_MAX_MODEL_LEN or self.critic_token_cap < ROLLOUT_MAX_MODEL_LEN:
            raise ValueError(f"{self.profile_id}: dynamic token caps must fit the longest configured context")
        if self.rollout_batched_tokens < ROLLOUT_MAX_MODEL_LEN:
            raise ValueError(f"{self.profile_id}: vLLM batched-token cap must fit max_model_len")
        if self.rollout_max_num_seqs <= 0:
            raise ValueError(f"{self.profile_id}: rollout_max_num_seqs must be positive")
        if not 0.0 < self.rollout_gpu_memory_utilization < 1.0:
            raise ValueError(f"{self.profile_id}: invalid vLLM memory utilization")


@dataclass(frozen=True)
class Candidate:
    phase: str
    workload: Workload
    topology: Topology
    batch_profile: BatchProfile
    seed: int
    stabilization_steps: int = STABILIZATION_STEPS
    measured_steps: int = MEASURED_STEPS

    @property
    def total_steps(self) -> int:
        return self.stabilization_steps + self.measured_steps

    @property
    def candidate_id(self) -> str:
        return (
            f"{self.workload.workload_id}-{self.topology.topology_id.lower()}-"
            f"{self.batch_profile.profile_id.lower()}-s{self.seed}"
        )

    def validate(self) -> None:
        self.topology.validate()
        self.batch_profile.validate()
        if self.workload.num_critiques not in {0, 4}:
            raise ValueError(f"{self.candidate_id}: topology study supports only M0 and M4")
        if self.workload.recipe == "scalar_random":
            if (self.workload.critic_head, self.workload.mark_selector) != ("scalar", "random"):
                raise ValueError(f"{self.candidate_id}: invalid scalar_random recipe")
        elif self.workload.recipe == "beta_variance":
            if (self.workload.critic_head, self.workload.mark_selector) != ("beta", "variance"):
                raise ValueError(f"{self.candidate_id}: invalid beta_variance recipe")
        else:
            raise ValueError(f"{self.candidate_id}: unsupported recipe")
        if self.stabilization_steps < 1 or self.measured_steps < 1:
            raise ValueError(f"{self.candidate_id}: at least one stabilization and measured step are required")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload = dataclasses.asdict(self)
        payload["candidate_id"] = self.candidate_id
        payload["workload_id"] = self.workload.workload_id
        payload["total_steps"] = self.total_steps
        payload["contract"] = {
            "prompts_per_iteration": PROMPTS_PER_ITERATION,
            "rollouts_per_prompt": ROLLOUTS_PER_PROMPT,
            "solutions_per_iteration": SOLUTIONS_PER_ITERATION,
            "prompt_tokens": PROMPT_TOKENS,
            "solution_tokens": SOLUTION_TOKENS,
            "critique_tokens": CRITIQUE_TOKENS,
            "continuations_per_mark": 1,
            "max_marks": 1,
            "temperature": 1.0,
            "synchronous": True,
            "feature_enabled": True,
        }
        return payload


WORKLOADS = tuple(
    Workload(model, model_path, recipe, critic_head, selector, critiques)
    for model, model_path in (
        ("qwen3-1p7b", "/hf_models/Qwen3-1.7B"),
        ("qwen3-4b", "/hf_models/Qwen3-4B"),
    )
    for recipe, critic_head, selector in (
        ("scalar_random", "scalar", "random"),
        ("beta_variance", "beta", "variance"),
    )
    for critiques in (0, 4)
)


TWO_NODE_TOPOLOGIES = (
    Topology("T01", 2, "fsdp2", 8, 8, 1, 1),
    Topology("T02", 2, "fsdp2", 8, 8, 1, 2),
    Topology("T03", 2, "fsdp2", 8, 8, 1, 4),
    Topology("T04", 2, "fsdp2", 16, 16, 1, 1),
    Topology("T05", 2, "fsdp2", 16, 16, 1, 2),
    Topology("T06", 2, "fsdp2", 16, 16, 1, 4),
    Topology("T07", 2, "fsdp2", 8, 16, 1, 2),
    Topology("T08", 2, "fsdp2", 16, 8, 1, 2),
    Topology("T09", 2, "fsdp2", 8, 8, 2, 2),
    Topology("T10", 2, "fsdp2", 16, 16, 2, 2),
    Topology("T11", 2, "fsdp", 8, 8, 1, 2),
    Topology("T12", 2, "fsdp", 16, 16, 1, 2),
)

BETA_CORE_TOPOLOGY_IDS = frozenset({"T01", "T02", "T04", "T05", "T09", "T10", "T11"})

BATCH_PROFILES = {
    "P01": BatchProfile("P01", True, True, 24576, 24576, 32768, 256, 0.60, True, True, True),
    "P02": BatchProfile("P02", False, False, 24576, 24576, 32768, 256, 0.60, True, True, True),
    "P03": BatchProfile("P03", True, True, 49152, 49152, 65536, 512, 0.70, False, True, True),
    "P04": BatchProfile("P04", True, True, 49152, 49152, 65536, 512, 0.70, True, True, True),
    "P05": BatchProfile("P05", True, True, 49152, 49152, 65536, 512, 0.65, False, False, False, True),
}


def workload_by_id(workload_id: str) -> Workload:
    matches = [workload for workload in WORKLOADS if workload.workload_id == workload_id]
    if len(matches) != 1:
        raise ValueError(f"unknown workload id {workload_id!r}")
    return matches[0]


def topology_by_id(topology_id: str) -> Topology:
    matches = [topology for topology in TWO_NODE_TOPOLOGIES if topology.topology_id == topology_id]
    if len(matches) != 1:
        raise ValueError(f"unknown two-node topology id {topology_id!r}")
    return matches[0]


def four_node_variants(source: Topology) -> tuple[Topology, ...]:
    if source.nodes != 2:
        raise ValueError("four-node promotion requires a two-node source topology")
    preserved = Topology(
        topology_id=f"N4_{source.topology_id}_P",
        nodes=4,
        strategy=source.strategy,
        actor_fsdp_size=source.actor_fsdp_size,
        critic_fsdp_size=source.critic_fsdp_size,
        rollout_tp=source.rollout_tp,
        sequence_parallel_size=source.sequence_parallel_size,
        source_topology=source.topology_id,
    )
    expanded = Topology(
        topology_id=f"N4_{source.topology_id}_X",
        nodes=4,
        strategy=source.strategy,
        actor_fsdp_size=min(32, source.actor_fsdp_size * 2),
        critic_fsdp_size=min(32, source.critic_fsdp_size * 2),
        rollout_tp=source.rollout_tp,
        sequence_parallel_size=source.sequence_parallel_size,
        source_topology=source.topology_id,
    )
    return (preserved,) if preserved == expanded else (preserved, expanded)


def two_node_core(seed: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    for workload in WORKLOADS:
        topologies = TWO_NODE_TOPOLOGIES
        if workload.recipe == "beta_variance":
            topologies = tuple(t for t in topologies if t.topology_id in BETA_CORE_TOPOLOGY_IDS)
        candidates.extend(
            Candidate("two_node_core", workload, topology, BATCH_PROFILES["P01"], seed) for topology in topologies
        )
    return candidates


def batching_expansion(workload_ids: Iterable[str], topology_ids: Iterable[str], seed: int) -> list[Candidate]:
    return [
        Candidate("batching_expansion", workload_by_id(workload_id), topology_by_id(topology_id), profile, seed)
        for workload_id in workload_ids
        for topology_id in topology_ids
        for profile in (BATCH_PROFILES["P02"], BATCH_PROFILES["P03"], BATCH_PROFILES["P04"])
    ]


def finalist_repeats(
    workload_id: str,
    topology_id: str,
    profile_id: str,
    seeds: Iterable[int],
) -> list[Candidate]:
    return [
        Candidate(
            "two_node_finalist",
            workload_by_id(workload_id),
            topology_by_id(topology_id),
            BATCH_PROFILES[profile_id],
            seed,
            stabilization_steps=1,
            measured_steps=4,
        )
        for seed in seeds
    ]


def memory_expansion(workload_id: str, topology_id: str, seed: int) -> list[Candidate]:
    return [
        Candidate(
            "memory_expansion",
            workload_by_id(workload_id),
            topology_by_id(topology_id),
            BATCH_PROFILES["P05"],
            seed,
        )
    ]


def four_node_expansion(
    workload_id: str,
    topology_ids: Iterable[str],
    profile_id: str,
    seed: int,
) -> list[Candidate]:
    return [
        Candidate("four_node_expansion", workload_by_id(workload_id), topology, BATCH_PROFILES[profile_id], seed)
        for source_id in topology_ids
        for topology in four_node_variants(topology_by_id(source_id))
    ]


def four_node_finalist_repeats(
    workload_id: str,
    source_topology_id: str,
    variant: str,
    profile_id: str,
    seeds: Iterable[int],
) -> list[Candidate]:
    suffix = {"preserved": "_P", "expanded": "_X"}[variant]
    topologies = [
        topology
        for topology in four_node_variants(topology_by_id(source_topology_id))
        if topology.topology_id.endswith(suffix)
    ]
    if len(topologies) != 1:
        raise ValueError(f"source topology {source_topology_id} has no unique {variant} four-node variant")
    return [
        Candidate(
            "four_node_finalist",
            workload_by_id(workload_id),
            topologies[0],
            BATCH_PROFILES[profile_id],
            seed,
            stabilization_steps=1,
            measured_steps=4,
        )
        for seed in seeds
    ]


def write_manifest(path: Path, candidates: Iterable[Candidate]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    seen: set[str] = set()
    with path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            if candidate.candidate_id in seen:
                raise ValueError(f"duplicate candidate id {candidate.candidate_id}")
            seen.add(candidate.candidate_id)
            handle.write(json.dumps(candidate.to_dict(), sort_keys=True) + "\n")
            count += 1
    return count


def _csv(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    core = subparsers.add_parser("two-node-core")
    core.add_argument("--seed", type=int, default=1234)
    core.add_argument("--output", type=Path, required=True)

    batching = subparsers.add_parser("batching-expansion")
    batching.add_argument("--workloads", type=_csv, required=True)
    batching.add_argument("--topologies", type=_csv, required=True)
    batching.add_argument("--seed", type=int, default=1234)
    batching.add_argument("--output", type=Path, required=True)

    finalists = subparsers.add_parser("finalists")
    finalists.add_argument("--workload", required=True)
    finalists.add_argument("--topology", required=True)
    finalists.add_argument("--profile", choices=sorted(BATCH_PROFILES), required=True)
    finalists.add_argument(
        "--seeds", type=lambda value: [int(item) for item in _csv(value)], default=[1234, 2345, 3456]
    )
    finalists.add_argument("--output", type=Path, required=True)

    memory = subparsers.add_parser("memory-expansion")
    memory.add_argument("--workload", required=True)
    memory.add_argument("--topology", required=True)
    memory.add_argument("--seed", type=int, default=1234)
    memory.add_argument("--output", type=Path, required=True)

    four = subparsers.add_parser("four-node")
    four.add_argument("--workload", required=True)
    four.add_argument("--topologies", type=_csv, required=True)
    four.add_argument("--profile", choices=sorted(BATCH_PROFILES), required=True)
    four.add_argument("--seed", type=int, default=1234)
    four.add_argument("--output", type=Path, required=True)

    four_finalists = subparsers.add_parser("four-node-finalists")
    four_finalists.add_argument("--workload", required=True)
    four_finalists.add_argument("--source-topology", required=True)
    four_finalists.add_argument("--variant", choices=("preserved", "expanded"), required=True)
    four_finalists.add_argument("--profile", choices=sorted(BATCH_PROFILES), required=True)
    four_finalists.add_argument(
        "--seeds",
        type=lambda value: [int(item) for item in _csv(value)],
        default=[1234, 2345, 3456],
    )
    four_finalists.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "two-node-core":
        candidates = two_node_core(args.seed)
    elif args.command == "batching-expansion":
        candidates = batching_expansion(args.workloads, args.topologies, args.seed)
    elif args.command == "finalists":
        candidates = finalist_repeats(args.workload, args.topology, args.profile, args.seeds)
    elif args.command == "memory-expansion":
        candidates = memory_expansion(args.workload, args.topology, args.seed)
    elif args.command == "four-node":
        candidates = four_node_expansion(args.workload, args.topologies, args.profile, args.seed)
    else:
        candidates = four_node_finalist_repeats(
            args.workload,
            args.source_topology,
            args.variant,
            args.profile,
            args.seeds,
        )
    count = write_manifest(args.output, candidates)
    print(json.dumps({"manifest": str(args.output), "candidates": count}, sort_keys=True))


if __name__ == "__main__":
    main()
