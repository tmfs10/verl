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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-file", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--critic-head", choices=("scalar", "beta"), required=True)
    parser.add_argument("--mark-selector", choices=("random", "ema", "variance"), required=True)
    parser.add_argument("--num-critiques", type=int, required=True)
    parser.add_argument("--expected-global-step", type=int, required=True)
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
    require(bool(rows), f"empty audit file: {path}")
    return rows


def latest_native_checkpoint(checkpoint_root: Path) -> tuple[int, Path]:
    candidates = [path for path in checkpoint_root.glob("global_step_*") if path.is_dir()]
    require(bool(candidates), f"no native global_step checkpoint beneath {checkpoint_root}")

    def step(path: Path) -> int:
        return int(path.name.removeprefix("global_step_"))

    path = max(candidates, key=step)
    return step(path), path


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.audit_file)
    by_event: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_event.setdefault(str(row.get("event")), []).append(row)

    warmup = by_event.get("warmup", [])
    require(bool(warmup), "critic-only warmup was not audited")
    require(all(row.get("continuations") == 0 for row in warmup), "warmup requested a continuation")

    actor_batches = by_event.get("actor_batch", [])
    require(bool(actor_batches), "post-warmup actor update was not audited")
    for row in actor_batches:
        require(row.get("continuations") == 0, "a continuation entered an actor batch")
        require(row.get("padding") == 0, "actor optimizer batch contains dummy padding")
        solutions = int(row["solutions"])
        critiques = int(row["critiques"])
        require(critiques == solutions * args.num_critiques, "actor critique multiplicity is incorrect")

    selections = [row for row in by_event.get("mark_selection", []) if row.get("reason") != "ema_summary"]
    require(bool(selections), "no nonterminal mark was selected")
    if args.mark_selector == "random":
        require(all(row.get("reason") == "random" for row in selections), "random selector used another reason")
    elif args.mark_selector == "ema":
        require(
            all(row.get("reason") in {"ema_up", "ema_down"} for row in selections),
            "EMA selector used an unknown reason",
        )
        require(all(math.isfinite(float(row["ratio"])) for row in selections), "EMA ratio is non-finite")
    else:
        require(args.critic_head == "beta", "variance selection did not use a Beta critic")
        require(
            all(row.get("reason") in {"variance", "random_fallback"} for row in selections),
            "variance selector used an unknown reason",
        )
        require(
            all(math.isfinite(float(row["variance"])) and float(row["variance"]) >= 0 for row in selections),
            "selected variance is invalid",
        )

    continuations = by_event.get("continuation", [])
    require(bool(continuations), "no continuation completed")
    for row in continuations:
        reward = float(row["reward"])
        require(0.0 <= reward <= args.max_reward and math.isfinite(reward), "continuation reward is invalid")

    targets = [row for row in by_event.get("critic_targets", []) if int(row.get("global_step", 0)) > 1]
    require(bool(targets), "post-warmup critic targets were not audited")
    require(any(int(row["dense_token_labels"]) > 0 for row in targets), "no dense continuation labels survived")
    for row in targets:
        terminal = int(row["terminal_token"])
        require(
            all(1 <= int(mark) < terminal for mark in row.get("selected_marks", [])),
            "a selected mark is not a one-indexed nonterminal prefix",
        )
        initial = float(row["initial_state_target"])
        require(0.0 <= initial <= args.max_reward and math.isfinite(initial), "invalid trained V(s0) target")

    step, checkpoint = latest_native_checkpoint(args.checkpoint_root)
    require(
        step == args.expected_global_step, f"expected native checkpoint step {args.expected_global_step}, got {step}"
    )
    require((checkpoint / "actor").is_dir(), f"missing native actor checkpoint in {checkpoint}")
    require((checkpoint / "critic").is_dir(), f"missing native critic checkpoint in {checkpoint}")
    require((checkpoint / "data.pt").is_file(), f"missing native dataloader state in {checkpoint}")
    require(
        not (checkpoint / "intermediate_mc_value_state.json").exists(),
        "obsolete feature-owned checkpoint state was written",
    )

    print(
        "verified",
        {
            "audit_rows": len(rows),
            "actor_batches": len(actor_batches),
            "continuations": len(continuations),
            "latest_checkpoint": str(checkpoint),
            "global_step": step,
        },
    )


if __name__ == "__main__":
    main()
