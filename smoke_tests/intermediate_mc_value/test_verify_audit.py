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


def _write_fixture(tmp_path, *, actor_continuations=0, num_critiques=2, mark_selector="random"):
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
            "event": "critic_batch",
            "global_step": 1,
            "solutions": 1,
            "contexts": max(1, num_critiques),
            "critiques": num_critiques,
        },
        {
            "event": "actor_batch",
            "global_step": 2,
            "solutions": 1,
            "critiques": num_critiques,
            "continuations": actor_continuations,
            "padding": 0,
        },
        {
            "event": "critic_batch",
            "global_step": 2,
            "solutions": 1,
            "contexts": max(1, num_critiques),
            "critiques": num_critiques,
        },
        {"event": "critic_scored", "global_step": 2, "contexts": max(1, num_critiques), "solutions": 1},
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
    selection = {
        "event": "mark_selection",
        "global_step": 2,
        "rollout_id": "r",
        "token": 2,
        "reason": "random",
    }
    if mark_selector == "ema":
        selection.update({"reason": "ema_up", "value": 0.7, "ema": 0.7, "reference": 0.2, "ratio": 3.5})
    elif mark_selector == "variance":
        selection.update({"reason": "variance", "variance": 0.1, "draw": 0.5, "scope": "r"})
    scored_index = next(index for index, row in enumerate(rows) if row["event"] == "critic_scored")
    rows.insert(scored_index + 1, selection)
    if num_critiques > 0:
        rows.append(
            {
                "event": "critique_credit",
                "global_step": 2,
                "rollout_id": "r",
                "rewards": [0.5] * num_critiques,
                "advantages": [0.0] * num_critiques,
            }
        )
    audit_path = audit_dir / "intermediate_mc_value.jsonl"
    audit_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return audit_path, checkpoint_root


def _argv(
    audit_path,
    checkpoint_root,
    *,
    num_critiques=2,
    mark_selector="random",
    critic_head="scalar",
    expected_global_step=2,
):
    return [
        "verify_audit.py",
        "--audit-file",
        str(audit_path),
        "--checkpoint-root",
        str(checkpoint_root),
        "--critic-head",
        critic_head,
        "--mark-selector",
        mark_selector,
        "--num-critiques",
        str(num_critiques),
        "--critic-warmup",
        "1",
        "--expected-global-step",
        str(expected_global_step),
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


def test_verifier_rejects_actor_batch_during_critic_warmup(tmp_path, monkeypatch) -> None:
    audit_path, checkpoint_root = _write_fixture(tmp_path)
    rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    actor_batch = next(row for row in rows if row["event"] == "actor_batch")
    rows.insert(rows.index(actor_batch), {**actor_batch, "global_step": 1})
    audit_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr("sys.argv", _argv(audit_path, checkpoint_root))
    with pytest.raises(AssertionError, match="actor should be frozen"):
        verify_audit.main()


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "nonconsecutive"))
def test_verifier_requires_exact_critic_warmup_steps(tmp_path, monkeypatch, mutation) -> None:
    audit_path, checkpoint_root = _write_fixture(tmp_path)
    rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    warmup = next(row for row in rows if row["event"] == "warmup")
    if mutation == "missing":
        rows.remove(warmup)
    elif mutation == "duplicate":
        rows.insert(rows.index(warmup), warmup.copy())
    else:
        warmup["global_step"] = 2
    audit_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr("sys.argv", _argv(audit_path, checkpoint_root))
    with pytest.raises(AssertionError, match="warmup"):
        verify_audit.main()


def test_verifier_requires_post_warmup_actor_batch(tmp_path, monkeypatch) -> None:
    audit_path, checkpoint_root = _write_fixture(tmp_path)
    rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row["event"] != "actor_batch"]
    audit_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr("sys.argv", _argv(audit_path, checkpoint_root))
    with pytest.raises(AssertionError, match="post-warmup actor update"):
        verify_audit.main()


def test_resume_verifier_requires_actor_batch_at_resumed_step(tmp_path, monkeypatch) -> None:
    audit_path, checkpoint_root = _write_fixture(tmp_path)
    (checkpoint_root / "global_step_2").rename(checkpoint_root / "global_step_3")
    monkeypatch.setattr("sys.argv", _argv(audit_path, checkpoint_root, expected_global_step=3))
    with pytest.raises(AssertionError, match="expected actor audit steps"):
        verify_audit.main()


def test_verifier_rejects_feature_owned_checkpoint_state(tmp_path, monkeypatch) -> None:
    audit_path, checkpoint_root = _write_fixture(tmp_path)
    (checkpoint_root / "global_step_2" / "intermediate_mc_value_state.json").touch()
    monkeypatch.setattr("sys.argv", _argv(audit_path, checkpoint_root))
    with pytest.raises(AssertionError, match="obsolete"):
        verify_audit.main()


def test_verifier_accepts_no_self_critique_contract(tmp_path, monkeypatch, capsys) -> None:
    audit_path, checkpoint_root = _write_fixture(tmp_path, num_critiques=0)
    monkeypatch.setattr("sys.argv", _argv(audit_path, checkpoint_root, num_critiques=0))
    verify_audit.main()
    assert "verified" in capsys.readouterr().out


def test_verifier_rejects_synthetic_no_self_critique_credit(tmp_path, monkeypatch) -> None:
    audit_path, checkpoint_root = _write_fixture(tmp_path, num_critiques=0)
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "critique_credit", "global_step": 2}) + "\n")
    monkeypatch.setattr("sys.argv", _argv(audit_path, checkpoint_root, num_critiques=0))
    with pytest.raises(AssertionError, match="synthetic critique credit"):
        verify_audit.main()


def test_verifier_requires_ema_selection_after_critic_scoring(tmp_path, monkeypatch) -> None:
    audit_path, checkpoint_root = _write_fixture(tmp_path, mark_selector="ema")
    rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    selection = next(row for row in rows if row.get("reason") == "ema_up")
    rows.remove(selection)
    scored_index = next(index for index, row in enumerate(rows) if row["event"] == "critic_scored")
    rows.insert(scored_index, selection)
    audit_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr("sys.argv", _argv(audit_path, checkpoint_root, mark_selector="ema"))
    with pytest.raises(AssertionError, match="before critic scoring"):
        verify_audit.main()
