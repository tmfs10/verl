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

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smoke_tests.branch_revision_grpo.submit_oci_iad import _extra_args, build_command
from smoke_tests.branch_revision_grpo.verify_smoke import verify


def test_rendered_smoke_contract_is_synchronous_temperature_one_and_wandb_free(tmp_path: Path) -> None:
    command, remote_output = build_command(
        run_tag="unit",
        dry_run=False,
        python=Path("/python"),
        launcher=Path("/launcher"),
        verl_root=Path("/verl"),
        reward_file=Path("/reward.py"),
        config_dir=tmp_path,
    )
    rendered = " ".join(command)
    assert remote_output == "/output/smoke_tests/branch_revision_grpo/unit"
    assert "algorithm.branch_revision_grpo.enable=true" in rendered
    assert "algorithm.intermediate_mc_value.enable=false" in rendered
    assert "actor_rollout_ref.rollout.temperature=1.0" in rendered
    assert "actor_rollout_ref.rollout.val_kwargs.temperature=1.0" in rendered
    assert "actor_rollout_ref.actor.policy_loss.loss_mode=dppo_tv" in rendered
    assert "reward.reward_model.launch_reward_fn_async=false" in rendered
    assert "--enable_wandb" not in command
    assert "--no_requeue" in command
    assert "--add_interactive" in command


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture(root: Path, *, include_continuation: bool = True) -> None:
    _write_json(root / "completed.json", {"status": "completed", "wall_seconds": 2.0})
    events = []
    continuation_count = 0
    for original_index in range(2):
        rollout = f"p:{original_index}"
        for critique_index in range(2):
            valid = include_continuation and original_index == 0 and critique_index == 0
            events.append(
                {
                    "event": "critique",
                    "rollout_id": rollout,
                    "critique_index": critique_index,
                    "reward": 0.5 if valid else -0.5,
                    "continuation_outcome": 1.0 if valid else 0.0,
                    "prompt_pass_at_1": 0.5,
                    "parse_reason": "valid" if valid else "tag_count",
                    "branch": "bad" if valid else "",
                    "new_continuation": "good" if valid else "",
                }
            )
            if valid:
                continuation_count += 1
                events.append(
                    {
                        "event": "continuation",
                        "rollout_id": rollout,
                        "critique_index": critique_index,
                        "reward": 1.0,
                        "revised_prefix_ids": [1],
                        "continuation_ids": [2],
                        "continuation_log_probs": [-0.2],
                    }
                )
    rows = 16 + 4 + continuation_count
    padding = (-rows) % 8
    events.extend(
        [
            {
                "event": "actor_batch",
                "rows": rows,
                "original": 16,
                "critiques": 4,
                "continuations": continuation_count,
                "padding": padding,
                "policy_loss_mode": "dppo_tv",
            },
            {
                "event": "iteration",
                "originals": 16,
                "incorrect": 2,
                "original_rewards": [0.0, 0.0] + [1.0] * 14,
            },
        ]
    )
    _write_jsonl(root / "audit" / "step_00000001.jsonl", events)
    _write_jsonl(
        root / "metrics.jsonl",
        [
            {
                "step": 1,
                "data": {
                    "branch_revision/originals": 16.0,
                    "branch_revision/incorrect_originals": 2.0,
                    "branch_revision/critiques": 4.0,
                    "branch_revision/valid_edits": float(continuation_count),
                    "branch_revision/continuations": float(continuation_count),
                    "branch_revision/policy_loss_is_dppo_tv": 1.0,
                    "actor/grad_norm": 1.0,
                },
            }
        ],
    )


def test_verifier_accepts_complete_live_contract(tmp_path: Path) -> None:
    _fixture(tmp_path)
    result = verify(tmp_path)
    assert result["status"] == "verified"
    assert result["valid_edits"] == 1


def test_verifier_rejects_smoke_without_a_valid_revision(tmp_path: Path) -> None:
    _fixture(tmp_path, include_continuation=False)
    with pytest.raises(ValueError, match="no strictly parsed"):
        verify(tmp_path)


def test_extra_args_contains_no_async_or_critic_training() -> None:
    rendered = _extra_args("/output/evidence")
    assert "critic.enable=false" in rendered
    assert "launch_reward_fn_async=false" in rendered
    assert "actor_rollout_ref.rollout.temperature=1.0" in rendered
