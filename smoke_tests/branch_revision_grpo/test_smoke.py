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
    assert "algorithm.branch_revision_grpo.num_critiques=4" in rendered
    assert "actor_rollout_ref.rollout.n=4" in rendered
    assert "algorithm.branch_revision_grpo.min_continuation_tokens=128" in rendered
    assert "data.max_response_length=2048" in rendered
    assert "reward.reward_model.launch_reward_fn_async=false" in rendered
    assert "--enable_wandb" not in command
    assert "--no_requeue" in command
    assert "--add_interactive" in command


def test_rendered_smoke_scales_dataset_batch_rollouts_and_critiques_together(tmp_path: Path) -> None:
    command, _ = build_command(
        run_tag="scaled",
        dry_run=False,
        python=Path("/python"),
        launcher=Path("/launcher"),
        verl_root=Path("/verl"),
        reward_file=Path("/reward.py"),
        config_dir=tmp_path,
        n_prompts=32,
        n_samples=4,
        num_critiques=6,
        seed=47,
    )
    rendered = " ".join(command)
    assert "--n_prompts 32" in rendered
    assert "--n_samples 4" in rendered
    assert "--seed 47" in rendered
    assert "data.train_batch_size=32" in rendered
    assert "++data.gen_batch_size=32" in rendered
    assert "actor_rollout_ref.rollout.n=4" in rendered
    assert "algorithm.branch_revision_grpo.num_critiques=6" in rendered


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"n_prompts": 0}, "n_prompts"),
        ({"n_samples": 0}, "n_samples"),
        ({"num_critiques": 1}, "at least 2"),
        ({"seed": -1}, "nonnegative"),
    ],
)
def test_rendered_smoke_rejects_invalid_scale(overrides: dict[str, int], match: str, tmp_path: Path) -> None:
    kwargs = {
        "run_tag": "invalid",
        "dry_run": False,
        "python": Path("/python"),
        "launcher": Path("/launcher"),
        "verl_root": Path("/verl"),
        "reward_file": Path("/reward.py"),
        "config_dir": tmp_path,
        **overrides,
    }
    with pytest.raises(ValueError, match=match):
        build_command(**kwargs)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture(root: Path, *, include_continuation: bool = True) -> None:
    _write_json(root / "completed.json", {"status": "completed", "wall_seconds": 2.0})
    _write_json(
        root / "resolved_config.json",
        {
            "algorithm": {"branch_revision_grpo": {"num_critiques": 2, "min_continuation_tokens": 128}},
            "data": {"train_batch_size": 8},
            "actor_rollout_ref": {"rollout": {"n": 2}},
        },
    )
    events = []
    original_rewards = [0.0, 1.0, 0.0, 1.0] + [1.0] * 12
    actor_rows = [
        {
            "kind": "original",
            "group_id": f"solution:prompt-{index // 2}",
            "reward": original_rewards[index],
        }
        for index in range(16)
    ]
    continuation_count = 0
    for original_index in range(2):
        rollout = f"p:{original_index}"
        prompt_group_id = f"prompt-{original_index}"
        for critique_index in range(2):
            valid = include_continuation and original_index == 0 and critique_index == 0
            events.append(
                {
                    "event": "critique",
                    "rollout_id": rollout,
                    "prompt_group_id": prompt_group_id,
                    "critique_index": critique_index,
                    "reward": 0.5 if valid else -0.5,
                    "continuation_outcome": 1.0 if valid else 0.0,
                    "prompt_pass_at_1": 0.5,
                    "parse_reason": "valid" if valid else "tag_count",
                    "branch": "bad" if valid else "",
                    "new_continuation": "good" if valid else "",
                    "critique_prompt_ids": [10, 11, original_index],
                    "critique_ids": [20, critique_index],
                    "critique_log_probs": [-0.2, -0.3],
                }
            )
            actor_rows.append(
                {
                    "kind": "critique",
                    "group_id": f"critique:{rollout}",
                    "reward": 0.5 if valid else -0.5,
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
                        "continuation_max_tokens": 128,
                    }
                )
                actor_rows.append(
                    {
                        "kind": "continuation",
                        "group_id": f"solution:{prompt_group_id}",
                        "reward": 1.0,
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
                "actor_rows": actor_rows,
            },
            {
                "event": "iteration",
                "originals": 16,
                "incorrect": 2,
                "original_rewards": original_rewards,
                "prompt_pass_at_1": {
                    "prompt-0": 0.5,
                    "prompt-1": 0.5,
                    **{f"prompt-{index}": 1.0 for index in range(2, 8)},
                },
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
                    "actor/pg_loss": 0.1,
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


def test_verifier_rejects_zero_signal_optimizer_step(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["data"]["actor/grad_norm"] = 0.0
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="positive learning signal"):
        verify(tmp_path)


def test_verifier_rejects_centered_critique_reward_on_revised_solution_row(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "audit" / "step_00000001.jsonl"
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    actor_batch = next(event for event in events if event["event"] == "actor_batch")
    continuation = next(row for row in actor_batch["actor_rows"] if row["kind"] == "continuation")
    continuation["reward"] = 0.5
    _write_jsonl(path, events)
    with pytest.raises(ValueError, match="continuation actor reward"):
        verify(tmp_path)


def test_verifier_rejects_critique_baseline_from_wrong_prompt_group(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "audit" / "step_00000001.jsonl"
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    critique = next(event for event in events if event["event"] == "critique")
    critique["prompt_group_id"] = "prompt-7"
    _write_jsonl(path, events)
    with pytest.raises(ValueError, match="wrong original prompt group"):
        verify(tmp_path)


def test_extra_args_contains_no_async_or_critic_training() -> None:
    rendered = _extra_args("/output/evidence")
    assert "critic.enable=false" in rendered
    assert "launch_reward_fn_async=false" in rendered
    assert "actor_rollout_ref.rollout.temperature=1.0" in rendered
    assert "critique_max_response_length=2560" in rendered
    assert "min_continuation_tokens=128" in rendered
