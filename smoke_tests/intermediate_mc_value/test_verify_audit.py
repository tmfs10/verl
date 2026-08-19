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

import json

import pytest

from smoke_tests.intermediate_mc_value import verify_audit


def _write_fixture(tmp_path, *, actor_continuations=0):
    audit_dir = tmp_path / "audit"
    checkpoint_root = tmp_path / "checkpoints"
    state_dir = checkpoint_root / "global_step_2"
    (state_dir / "actor").mkdir(parents=True)
    (state_dir / "critic").mkdir()
    (state_dir / "data.pt").touch()
    audit_dir.mkdir()
    rows = [
        {"event": "warmup", "global_step": 1, "continuations": 0},
        {
            "event": "actor_batch",
            "global_step": 2,
            "solutions": 1,
            "critiques": 2,
            "continuations": actor_continuations,
            "padding": 0,
        },
        {"event": "mark_selection", "global_step": 2, "rollout_id": "r", "token": 2, "reason": "random"},
        {"event": "continuation", "global_step": 2, "rollout_id": "r", "mark": 2, "reward": 1.0},
        {
            "event": "critic_targets",
            "global_step": 2,
            "rollout_id": "r",
            "selected_marks": [2],
            "surviving_marks": [2],
            "dense_token_labels": 2,
            "initial_state_target": 1.0,
            "terminal_token": 4,
        },
    ]
    audit_path = audit_dir / "intermediate_mc_value.jsonl"
    audit_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return audit_path, checkpoint_root


def _argv(audit_path, checkpoint_root):
    return [
        "verify_audit.py",
        "--audit-file",
        str(audit_path),
        "--checkpoint-root",
        str(checkpoint_root),
        "--critic-head",
        "scalar",
        "--mark-selector",
        "random",
        "--num-critiques",
        "2",
        "--expected-global-step",
        "2",
    ]


def test_positive_smoke_audit_and_native_checkpoint(tmp_path, monkeypatch, capsys) -> None:
    audit_path, checkpoint_root = _write_fixture(tmp_path)
    monkeypatch.setattr("sys.argv", _argv(audit_path, checkpoint_root))
    verify_audit.main()
    assert "verified" in capsys.readouterr().out


def test_verifier_rejects_continuation_actor_membership(tmp_path, monkeypatch) -> None:
    audit_path, checkpoint_root = _write_fixture(tmp_path, actor_continuations=1)
    monkeypatch.setattr("sys.argv", _argv(audit_path, checkpoint_root))
    with pytest.raises(AssertionError, match="continuation entered"):
        verify_audit.main()


def test_verifier_rejects_feature_owned_checkpoint_state(tmp_path, monkeypatch) -> None:
    audit_path, checkpoint_root = _write_fixture(tmp_path)
    (checkpoint_root / "global_step_2" / "intermediate_mc_value_state.json").touch()
    monkeypatch.setattr("sys.argv", _argv(audit_path, checkpoint_root))
    with pytest.raises(AssertionError, match="obsolete"):
        verify_audit.main()
