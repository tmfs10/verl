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

"""Verify structural invariants from an intermediate-MC GPU smoke run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

STATE_FILENAME = "intermediate_mc_value_state.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-file", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--recipe", choices=("scalar_random", "beta_variance"), required=True)
    parser.add_argument("--num-critiques", type=int, required=True)
    parser.add_argument("--expected-critic-updates", type=int, required=True)
    parser.add_argument("--max-reward", type=float, default=1.0)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_jsonl(path: Path) -> list[dict[str, object]]:
    require(path.is_file(), f"missing audit file: {path}")
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise AssertionError(f"invalid audit JSON at {path}:{line_number}: {error}") from error
            require(isinstance(row, dict), f"audit row {line_number} is not an object")
            rows.append(row)
    require(rows, f"empty audit file: {path}")
    return rows


def latest_feature_state(checkpoint_root: Path) -> tuple[Path, dict[str, object]]:
    candidates = list(checkpoint_root.glob(f"global_step_*/{STATE_FILENAME}"))
    require(bool(candidates), f"no {STATE_FILENAME} beneath {checkpoint_root}")

    def step(path: Path) -> int:
        return int(path.parent.name.removeprefix("global_step_"))

    path = max(candidates, key=step)
    with path.open(encoding="utf-8") as handle:
        state = json.load(handle)
    require(isinstance(state, dict), f"invalid checkpoint state: {path}")
    return path, state


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.audit_file)
    by_event: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_event.setdefault(str(row.get("event")), []).append(row)

    warmup = by_event.get("warmup", [])
    require(warmup, "warmup phase was not audited")
    require(all(row.get("continuations") == 0 for row in warmup), "warmup requested a continuation")

    actor_batches = by_event.get("actor_batch", [])
    require(actor_batches, "post-warmup actor update was not audited")
    for row in actor_batches:
        require(row.get("continuations") == 0, "a continuation entered an actor batch")
        solutions = int(row["solutions"])
        critiques = int(row["critiques"])
        require(critiques == solutions * args.num_critiques, "actor critique multiplicity is incorrect")

    selections = by_event.get("mark_selection", [])
    require(selections, "no nonterminal mark was selected")
    if args.recipe == "scalar_random":
        require(all(row.get("reason") == "random" for row in selections), "scalar recipe used a non-random mark")
    else:
        require(
            all(row.get("reason") in {"variance", "random_fallback"} for row in selections),
            "Beta recipe used an unknown selection reason",
        )
        for row in selections:
            variance = float(row["variance"])
            require(math.isfinite(variance) and variance >= 0.0, "invalid selected variance")

    continuations = by_event.get("continuation", [])
    require(continuations, "no continuation completed")
    for row in continuations:
        reward = float(row["reward"])
        require(0.0 <= reward <= args.max_reward and math.isfinite(reward), "continuation reward is out of range")

    targets = by_event.get("critic_targets", [])
    require(targets, "post-warmup critic targets were not audited")
    require(any(int(row["dense_token_labels"]) > 0 for row in targets), "no dense continuation labels survived")
    for row in targets:
        terminal = int(row["terminal_token"])
        require(
            all(1 <= int(mark) < terminal for mark in row.get("selected_marks", [])),
            "a selected mark was not a one-indexed nonterminal prefix",
        )
    require(
        all(int(row["terminal_token"]) >= 1 for row in targets),
        "a terminal label did not use the final valid response token",
    )

    state_path, state = latest_feature_state(args.checkpoint_root)
    require(
        state.get("critic_update_count") == args.expected_critic_updates,
        f"wrong critic update count in {state_path}: {state.get('critic_update_count')}",
    )
    contract = state.get("contract")
    require(isinstance(contract, dict), f"missing checkpoint contract in {state_path}")
    feature = contract.get("feature")
    require(isinstance(feature, dict), f"missing feature contract in {state_path}")
    require(feature.get("recipe") == args.recipe, f"checkpoint recipe mismatch in {state_path}")
    require(feature.get("num_critiques") == args.num_critiques, f"checkpoint critique count mismatch in {state_path}")

    print(
        "verified",
        {
            "audit_rows": len(rows),
            "actor_batches": len(actor_batches),
            "continuations": len(continuations),
            "latest_state": str(state_path),
            "critic_updates": state["critic_update_count"],
        },
    )


if __name__ == "__main__":
    main()
